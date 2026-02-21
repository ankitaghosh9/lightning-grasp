# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lygra Common
from lygra.robot import build_robot
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.mesh import get_urdf_mesh, get_urdf_mesh_decomposed, get_urdf_mesh_for_projection, trimesh_to_open3d, load_material_from_mtl
from lygra.mesh_analyzer import get_support_point_mask
from lygra.memory import IKGPUBufferPool
from lygra.utils.geom_utils import MeshObject
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.transform_utils import batch_object_transform
from lygra.utils.isaac_utils import generate_geodesic_confidence_mask, load_obj_rot_from_isaac, save_grasps_to_json_isaac_converted
from lygra.utils.save_utils import print_dict_structure, load_results_from_json


# Lygra Pipeline
from lygra.pipeline.module.object_placement import sample_object_pose, get_object_pose_sampling_args, get_bounded_object_pose
from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
from lygra.pipeline.module.contact_collection import sample_pose_and_contact_from_interaction
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.kinematics import batch_ik, batch_contact_adjustment
from lygra.pipeline.module.collision import batch_filter_collision
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter, batch_generate_bg_surface

# Common 
import torch 
import trimesh
from tqdm import tqdm 
import numpy as np 
import open3d as o3d
import argparse
import time 
import random
import sys
from scipy.spatial.transform import Rotation as R

object_mesh_path = "assets/testing_tool_4.obj"
object = MeshObject(object_mesh_path)
                    
robot = build_robot("shadow_dex")

# Robot Structure.
tree = build_kinematics_tree(
    urdf_path=robot.urdf_path,
    active_joint_names=robot.get_active_joints()
)
# print("JOINT", tree.active_joints)
# print("JOINT IDS: ", [tree.get_joint_id_from_active_joint_id(i) for i in range(len(tree.active_joints))])

# Robot Mesh Data
mesh_data = get_urdf_mesh(
    urdf_path=robot.urdf_path,
    tree=tree,
    mesh_scale=robot.get_mesh_scale()
)

mesh_data_for_ik = get_urdf_mesh_for_projection(
    urdf_path=robot.urdf_path,
    tree=tree,
    config=robot.get_contact_field_config(),
    mesh_scale=robot.get_mesh_scale()
)

decomposed_static_mesh_data = get_urdf_mesh_decomposed(
    urdf_path=robot.urdf_path,
    tree=tree,
    override_link_names=robot.get_static_links(),
    mesh_scale=robot.get_mesh_scale()
)

decomposed_mesh_data = get_urdf_mesh_decomposed(
    urdf_path=robot.urdf_path,
    tree=tree,
    mesh_scale=robot.get_mesh_scale()
)

# Robot Collision & Kinematics Metadata
self_collision_link_pairs = tree.get_self_collision_check_link_pairs(
    link_body_id=decomposed_mesh_data['link_body_id'],
    whitelist_link=[]
)

self_collision_link_pairs = torch.from_numpy(self_collision_link_pairs).cuda().int()

viewer = RobotVisualizer(robot)

# Offsets from URDF
forearm_to_wrist = np.array([0, -0.010, 0.213])
wrist_to_palm = np.array([0, 0, 0.034])

wrist_mesh = trimesh.load("assets/hand/shadow_dex/meshes/wrist.obj")
wrist_transform = np.eye(4)
wrist_transform[:3, 3] = -1 * wrist_to_palm
wrist_mesh.apply_transform(wrist_transform)
wrist_mesh = trimesh_to_open3d(wrist_mesh)

forearm_mesh = trimesh.load("assets/hand/shadow_dex/meshes/forearm.obj")
forearm_transform = np.eye(4)
forearm_transform[:3, 3] = -1 * (wrist_to_palm  + forearm_to_wrist)
forearm_mesh.apply_transform(forearm_transform)
forearm_mesh = trimesh_to_open3d(forearm_mesh)

result = load_results_from_json("assets/testing_lygra_results.json")
n_result = len(result['q'])
idx = 0
while idx < n_result:
    print(f"Solution Number {idx+1}:")

    # 1. Get Robot Mesh (Static)
    q = result['q'][idx:idx+1].detach().cpu().numpy()
    robot_mesh, robot_poses_dict = viewer.get_mesh_fk(q, visual=True, extra_meshes=[wrist_mesh, forearm_mesh])
    robot_link_poses = np.array(list(robot_poses_dict.values()))
    robot_link_names = list(robot_poses_dict.keys())

    # Initialize list for ALWAYS visible items (Robot, Contacts, Table)
    geometries = []
    geometries.extend(robot_mesh) # Add robot parts

    # 2. Prepare Data for Contacts
    object_mesh_data = object.mesh.copy() # Copy original mesh data
    object_pose = result['object_pose'][idx].cpu().numpy()
    contact_link_id = result["contact_link_id"][idx].cpu().numpy()
    contact_link_poses = robot_link_poses[contact_link_id]
    local_contact_pos = result['contact_pos'][idx].cpu().numpy()
    contact_pos_world = (contact_link_poses[:, :3, :3] @ local_contact_pos[..., None]).squeeze(-1) + contact_link_poses[:, :3, 3]

    # 3. Create Textured Object (Toggle Option A)
    object_mesh_textured = object.mesh.copy()
    object_mesh_textured.apply_transform(object_pose)
    object_mesh_o3d = trimesh_to_open3d(object_mesh_textured)
    material = load_material_from_mtl(object_mesh_path.replace(".obj", ".mtl"))
    textured_element = {"name": 'object', "geometry": object_mesh_o3d, "material": material}
    geometries.append(textured_element)

    # 4. Create Heatmap Object (Toggle Option B)
    contact_link_names = [robot_link_names[idx] for idx in contact_link_id]
    conf, heat_mesh_geom = generate_geodesic_confidence_mask(object_mesh_data, contact_pos_world, object_pose, contact_link_names)
    heat_mesh_geom.apply_transform(object_pose)
    heat_mesh_o3d = trimesh_to_open3d(heat_mesh_geom)
    heatmap_element = {"name": "heatmap", "geometry": heat_mesh_o3d, "material": None, "is_visible": False}
    geometries.append(heatmap_element)

    # 5. Create Contact Spheres (Static)
    # Create visuals for final link contact points (Red Spheres)
    contact_material = o3d.visualization.rendering.MaterialRecord()
    contact_material.base_color = [1.0, 0.0, 0.0, 1.0]  # Red
    contact_material.shader = "defaultLit"
    for i, point in enumerate(contact_pos_world):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=20)
        sphere.compute_vertex_normals()
        sphere.translate(point)
        sphere_dict = {
            "name": f"Contact_{i}",
            "geometry": sphere,
            "material": contact_material
        }
        geometries.append(sphere_dict)

    # # Create visuals for target contact points (Green Spheres)
    # target_material = o3d.visualization.rendering.MaterialRecord()
    # target_material.base_color = [0.0, 1.0, 0.0, 1.0]  # Green
    # target_material.shader = "defaultLit"
    # for i, point in enumerate(target_pos):
    #     # Create sphere
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003, resolution=20)
    #     sphere.compute_vertex_normals()
    #     sphere.translate(point)
    #     sphere_dict = {
    #         "name": f"Target_{i}",
    #         "geometry": sphere,
    #         "material": target_material
    #     }
    #     geometries.append(sphere_dict)

    
    # 6. Create Table/Surface (Static)
    if result['surface_points'] is not None:
        surface_points = result['surface_points'][idx].cpu().numpy()
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)
        surface_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        surface_material = o3d.visualization.rendering.MaterialRecord()
        surface_material.shader = "defaultLit"
        surface_material.point_size = 4.0
        surface_dict = {
            "name": "plane",
            "geometry": surface_pcd,
            "material": surface_material
        }
        geometries.append(surface_dict)

    # 2. The One-Liner Visualization
    # This opens a window with a sidebar. 
    # Click the "Scene" or "List" icon on the right to see checkboxes for your objects.
    o3d.visualization.draw(geometries, title=f"Solution {idx+1}")

    if input("Continue? (Y/n)") in ['n', 'N']:
        break
    idx+= 1