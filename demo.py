# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Lygra Common
from lygra.robot import build_robot
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.mesh import get_urdf_mesh, get_urdf_mesh_decomposed, get_urdf_mesh_for_projection, trimesh_to_open3d
from lygra.mesh_analyzer import get_support_point_mask
from lygra.utils.geom_utils import MeshObject
from lygra.memory import IKGPUBufferPool
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.transform_utils import batch_object_transform

# Lygra Pipeline
from lygra.pipeline.module.object_placement import sample_object_pose, get_object_pose_sampling_args
from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
from lygra.pipeline.module.contact_collection import sample_pose_and_contact_from_interaction
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.kinematics import batch_ik, batch_contact_adjustment
from lygra.pipeline.module.collision import batch_filter_collision
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter

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

import open3d as o3d
import numpy as np
import torch

def visualize_with_open3d(pos, normal, mask=None):
    """
    Args:
        pos: [N, 3] tensor or array
        normal: [N, 3] tensor or array
        mask: Optional [N,] boolean mask for "support points"
    """
    # Convert to numpy
    if isinstance(pos, torch.Tensor): pos = pos.cpu().numpy()
    if isinstance(normal, torch.Tensor): normal = normal.cpu().numpy()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.normals = o3d.utility.Vector3dVector(normal)

    # Colorize
    if mask is not None:
        if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
        # Initialize colors as Gray
        colors = np.full((pos.shape[0], 3), [0.5, 0.5, 0.5]) 
        # Support points = Green, Others = Red
        colors[mask] = [0, 1, 0]
        colors[~mask] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color based on normal direction (RGB = XYZ)
        pcd.orient_normals_consistent_tangent_plane(10)
        pcd.paint_uniform_color([0.1, 0.7, 1.0]) # Light Blue

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lightning Grasp - Point Cloud Debug")
    vis.add_geometry(pcd)
    
    # Render normals as lines
    opt = vis.get_render_option()
    opt.point_show_normal = True # Press 'N' to toggle this in the window
    
    vis.run()
    vis.destroy_window()

import numpy as np

def get_translation_statistics(poses):
    """
    Args:
        poses: np.ndarray of shape (N, 4, 4)
    """
    # 1. Extract Translation Vectors
    # Slicing: [All items, First 3 rows, 4th column]
    # Result shape: (N, 3) where each row is [x, y, z]
    translations = poses[:, :3, 3]
    
    # 2. Calculate Statistics along axis 0 (across the N samples)
    stats = {
        "mean": np.mean(translations, axis=0),
        "std":  np.std(translations, axis=0),
        "min":  np.min(translations, axis=0),
        "max":  np.max(translations, axis=0),
    }
    
    # 3. (Optional) Euclidean Distance from Origin (Norm)
    # Useful to see the average "radius" or distance
    norms = np.linalg.norm(translations, axis=1)
    stats["mean_norm"] = np.mean(norms)
    
    return stats

def get_args():
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument('--robot', type=str, default="allegro", help='Robot Name')
    parser.add_argument('--batch_size_outer', type=int, default=128, help='Outer batch size (Object Pose)')
    parser.add_argument('--batch_size_inner', type=int, default=128, help='Inner batch size (Contact Domain Variants)')
    parser.add_argument('--n_contact', type=int, default=3, help='Number of non-static contacts to optimize')
    parser.add_argument('--n_sample_point', type=int, default=2048, help='Number of sampled object points')
    parser.add_argument('--ik_finetune_iter', type=int, default=5, help='Number of IK finetune iterations')
    parser.add_argument('--zo_lr_sigma', type=float, default=5, help='Sigma of the Zeroth-order Optimizer')

    parser.add_argument('--cf_accel', type=str, default='lbvhs2', help='Contact Field Acceleration Structure')
    parser.add_argument('--object_pose_sampling_strategy', type=str, default='canonical', help='Object pose sampling strategy')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/object/ycb/013_apple/textured.obj", help='Path to the object mesh')

    args = parser.parse_args()
    return args


# In gratitude to the NJU ICS-PA, HKUST RI, and Berkeley CS267 teaching teams.
# This program embodies lessons and memories that came alive during its creation.
#                                                               -- Zhao-Heng Yin  
#                                                                       Nov 2025
def main(args):
    batch_size_outer = args.batch_size_outer
    batch_size_inner = args.batch_size_inner
    n_contact = args.n_contact
    n_sample_point = args.n_sample_point
    ik_finetune_iter = args.ik_finetune_iter
    cf_accel = args.cf_accel
    object_pose_sampling_strategy = args.object_pose_sampling_strategy
    visualize = args.visualize
    object_mesh_path = args.object_mesh_path
    zo_lr_sigma = args.zo_lr_sigma

    # -----------------
    # Preparation Stage 
    # -----------------
    robot = build_robot(args.robot)

    # Robot Structure.
    tree = build_kinematics_tree(
        urdf_path=robot.urdf_path,
        active_joint_names=robot.get_active_joints()
    )

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

    contact_field = robot.get_contact_field()
    # print("Contact Field", contact_field)
    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])

    contact_parent_links = contact_field.get_all_parent_link_names()
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()

    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets)
    dependency_matrix = dependency_matrix.cuda()

#################### PRINTING RESULTS ###########################
    # print("mesh data", mesh_data['v'].shape, mesh_data['f'].shape, mesh_data['n'].shape,
    #         mesh_data['vi'].shape, mesh_data['fi'].shape)
    
    # #robot_mesh = []
    # for i in range(len(mesh_data['vi'])-1):
    #     mesh_o3d = o3d.geometry.TriangleMesh()
    #     vertices = mesh_data['v'][mesh_data['vi'][i]:mesh_data['vi'][i+1]]
    #     faces = mesh_data['f'][mesh_data['fi'][i]:mesh_data['fi'][i+1]]
    #     face_normals = mesh_data['n'][mesh_data['fi'][i]:mesh_data['fi'][i+1]]
    #     print(vertices.shape, faces.shape, face_normals.shape)
    #     mesh_o3d.vertices=o3d.utility.Vector3dVector(vertices)
    #     mesh_o3d.triangles=o3d.utility.Vector3iVector(faces)
    #     #mesh_o3d.compute_vertex_normals()
    #     mesh_o3d.triangle_normals = o3d.utility.Vector3dVector(face_normals)
    #     mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
    #         #robot_mesh.append(mesh_o3d)
        
    #     o3d.visualization.draw_geometries(
    #         [mesh_o3d],
    #         #mesh_show_back_face=True
    #     )

    # #print("mesh data for ik", mesh_data_for_ik)
    # #print("decomposed static mesh data", decomposed_static_mesh_data)
    # #print("decomposed mesh data", decomposed_mesh_data)

    # # viewer = RobotVisualizer(robot)
    # # viewer.show([mesh_o3d])

    # assert False
##################################################################

    # Contact Field Acceleration Data Structure (LBVH-S2Bundle)
    accel_structure = contact_field.generate_acceleration_structure(method=cf_accel)

    # Object Data.
    object = MeshObject(object_mesh_path)
    
    object_mask = np.load('assets/object/testing.npy')
    #print("mask.shape: ", object_mask.shape)
    object.create_masked_submesh(object_mask)
    object_area = object.get_area()
    #print("object area", object_area)
    zo_lr = ((object_area / n_sample_point) ** 0.5) * zo_lr_sigma
    #print("zo_lr", zo_lr)
    points, normals = object.sample_point_and_normal(count=n_sample_point, return_submesh=True)
    #print("points, normals", points.shape, normals.shape)
    #visualize_with_open3d(points, normals)
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()

    # Filtering
    support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0] #remove convex (difficult to grasp) points
    points = points_all[torch.where(support_point_mask)]            # good grasp point.
    normals = normals_all[torch.where(support_point_mask)]          # good_grasp_point.
    #print("points, normals", points.shape, normals.shape)
    #visualize_with_open3d(points, normals)

    # IK GPU buffer. 
    gpu_memory_pool = IKGPUBufferPool(
        n_dof=tree.n_dof(), 
        n_link=tree.n_link(), 
        max_batch=min([batch_size_outer * batch_size_inner, 65536]), 
        retry=10
    )

    # ---------------
    # Inference Stage 
    # ---------------
    # TODO: Refactor Args/Returns (Nov 10)
    # I should have a class to wrap these arg/return values below as people did for professional graphics engines.
    # But I am too lazy to move, python dict is so comforting for prototyping.

    print("Launch Inference")
    with torch.no_grad():
        # Object Placement
        # Note: We put extra contact points in condition.
        # condition: [B, 1, 3] contact pos/normal, [B, 1] mask 
        object_poses, condition = sample_object_pose(
            n=batch_size_outer, 
            points=points, 
            normals=normals, 
            contact_field=contact_field, 
            tree=tree, 
            mesh_data=decomposed_static_mesh_data,
            sampling_args=get_object_pose_sampling_args(object_pose_sampling_strategy, robot)
        )
        print("OBJECT POSES", object_poses.shape, "CONDITION", 
              "extra_contact_pos", condition['extra_contact_pos'].shape, 
              "extra_contact_normal", condition['extra_contact_normal'].shape,
              "extra_contact_mask", condition["extra_contact_mask"].shape)
        
        trans_res = get_translation_statistics(object_poses.cpu().numpy())
        print("Translation Statistics (X, Y, Z):")
        print(f"Mean: {trans_res['mean']}")
        print(f"Std:  {trans_res['std']}")
        print(f"Min:  {trans_res['min']}")
        print(f"Max:  {trans_res['max']}")
        assert False

        # Contact Field BVH Traversal
        interaction_matrix_hand_point_idx = batch_object_all_contact_fields_interaction(
            object_pos=points, 
            object_normal=normals, 
            object_pose=object_poses, 
            accel_structure=accel_structure
        )

        interaction_matrix = (interaction_matrix_hand_point_idx >= 0).int()
        link_interaction_matrix = contact_field.reduce_link_interaction(interaction_matrix)

        # Get Contact Domain
        contact_domain_pos, contact_domain_normal, contact_domain_point_idx, \
        object_poses, contact_link_ids, condition, valid_outer_idx = \
        sample_pose_and_contact_from_interaction(
            n_contact=n_contact,
            interaction_matrix=link_interaction_matrix, 
            dependency_matrix=dependency_matrix, 
            object_points=points, 
            object_normals=normals, 
            object_poses=object_poses,
            condition=condition
        )

        # Search Contact Points in Contact Domain
        target_contact_pos, target_contact_normal, target_contact_point_idx, \
        object_poses, target_contact_link_ids, target_batch_outer_ids = \
        search_contact_point(
            contact_domain_pos=contact_domain_pos, 
            contact_domain_normal=contact_domain_normal, 
            contact_domain_point_idx=contact_domain_point_idx,
            object_poses=object_poses, 
            contact_ids=contact_link_ids,
            batch_size=batch_size_inner,
            return_hand_frame=True,
            condition=condition,
            zo_lr=zo_lr
        )

        contact_ids, local_contact_ids = contact_field.sample_contact_ids(
            interaction_matrix=interaction_matrix[valid_outer_idx], 
            interaction_matrix_hand_point_idx=interaction_matrix_hand_point_idx[valid_outer_idx],
            target_batch_outer_ids=target_batch_outer_ids, 
            target_contact_link_ids=target_contact_link_ids, 
            target_contact_point_idx=target_contact_point_idx
        )

        contact_pos_in_linkf, contact_normal_in_linkf = contact_field.sample_contact_geometry(contact_ids, local_contact_ids)

        # Kinematics Optimization (I)
        # Coarse IK. Might not align well.
        result = batch_ik(
            tree=tree,
            contact_ids=contact_ids,
            contact_parent_ids=contact_parent_ids,
            contact_pos_in_linkf=contact_pos_in_linkf.float(),
            contact_normal_in_linkf=contact_normal_in_linkf.float(),
            target_contact_pos=target_contact_pos.float(),
            target_contact_normal=target_contact_normal.float(),
            object_pose=object_poses.float(),
            gpu_memory_pool=gpu_memory_pool
        )
        
        # Kinematics Optimization (II)
        # Finegrained Finger Pose by Iterative Projection + IK Adjustment.
        result = batch_contact_adjustment(
            tree=tree,
            mesh=mesh_data_for_ik,
            q_init=result["q"],
            q_mask=result["q_mask"],
            contact_ids=contact_ids,
            contact_link_ids=result["contact_link_id"],
            contact_pos_in_linkf=result["contact_pos"],
            contact_normal_in_linkf=result["contact_normal"],
            target_contact_pos=result["target_pos"],
            target_contact_normal=result["target_normal"],
            object_pose=result["object_pose"],
            n_iter=ik_finetune_iter,
            gpu_memory_pool=gpu_memory_pool,
            ret_mesh_buffer=True
        )

        # Postprocessing: 
        # Search Free Finger Configuration & Remove Invalid Results (collision).
        # Hand-to-hand  -- AABB broad phase + GJK narrow phase 
        # Hand-to-point -- AABB broad phase + halfplane-test narrow phase
        result = batch_assign_free_finger_and_filter(
            tree=tree,
            result=result,
            object_point=points_all,
            self_collision_link_pairs=self_collision_link_pairs,
            decomposed_mesh_data=decomposed_mesh_data
        )

    n_result = len(result['q'])
    print("Number of Solutions:", n_result)
   
    # -----------------
    # Visualize Results
    # -----------------
    if not visualize:
        sys.exit(0)

    viewer = RobotVisualizer(robot)

    while True:
        idx = random.randint(0, n_result - 1)
        robot_mesh = viewer.get_mesh_fk(result['q'][idx:idx+1].detach().cpu().numpy(), visual=True)
        #print("robot mesh", robot_mesh)

        object_mesh = object.mesh.copy()
        object_mesh.apply_transform(result['object_pose'][idx].cpu().numpy())
        object_mesh_o3d = trimesh_to_open3d(object_mesh)
        
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [245 / 256, 162 / 256, 98 / 256, 0.8]
        material.base_metallic = 0.0
        material.base_roughness = 1.0
        object_mesh = {"name": 'object', "geometry": object_mesh_o3d, "material": material}
        viewer.show(robot_mesh + [object_mesh])

        if input("Continue? (Y/n)") in ['n', 'N']:
            break


if __name__ == '__main__':
    args = get_args()
    main(args)