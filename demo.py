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
from lygra.utils.geom_utils import MeshObject
from lygra.memory import IKGPUBufferPool
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.transform_utils import batch_object_transform

# Lygra Pipeline
from lygra.pipeline.module.object_placement import sample_object_pose, get_object_pose_sampling_args, get_bounded_object_pose
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
from scipy.spatial.transform import Rotation as R

def get_args():
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument('--robot', type=str, default="allegro", help='Robot Name')
    parser.add_argument('--batch_size_outer', type=int, default=64, help='Outer batch size (Object Pose)')
    parser.add_argument('--batch_size_inner', type=int, default=64, help='Inner batch size (Contact Domain Variants)')
    parser.add_argument('--n_contact_links', nargs='*', default=['ffdistal', 'ffmiddle', 'thdistal'], 
                        choices = ['rfdistal', 'lfdistal', 'ffmiddle', 'lfmiddle', 'ffdistal', 'rfmiddle', 'thdistal', 'mfmiddle', 'mfdistal'], 
                        help='Name of the links to be involved in the grasp')
    #parser.add_argument('--n_contact', type=int, default=3, help='Number of non-static contacts to optimize')
    parser.add_argument('--n_sample_point', type=int, default=2048, help='Number of sampled object points')
    parser.add_argument('--ik_finetune_iter', type=int, default=5, help='Number of IK finetune iterations')
    parser.add_argument('--zo_lr_sigma', type=float, default=5, help='Sigma of the Zeroth-order Optimizer')

    parser.add_argument('--cf_accel', type=str, default='lbvhs2', help='Contact Field Acceleration Structure')
    parser.add_argument('--object_pose_sampling_strategy', type=str, default='canonical', help='Object pose sampling strategy')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/object/testing.obj", help='Path to the object mesh')
    parser.add_argument('--object_mask_path', type=str, default=None, 
                        help='Path to the object mesh mask of graspable area' )

    args = parser.parse_args()
    return args


# In gratitude to the NJU ICS-PA, HKUST RI, and Berkeley CS267 teaching teams.
# This program embodies lessons and memories that came alive during its creation.
#                                                               -- Zhao-Heng Yin  
#                                                                       Nov 2025
def main(args):
    batch_size_outer = args.batch_size_outer
    batch_size_inner = args.batch_size_inner
    n_contact_links = args.n_contact_links
    #n_contact = args.n_contact
    n_sample_point = args.n_sample_point
    ik_finetune_iter = args.ik_finetune_iter
    cf_accel = args.cf_accel
    object_pose_sampling_strategy = args.object_pose_sampling_strategy
    visualize = args.visualize
    object_mesh_path = args.object_mesh_path
    object_mask_path = args.object_mask_path
    zo_lr_sigma = args.zo_lr_sigma

    n_contact = set()
    for l in n_contact_links:
        n_contact.add(l[0:2])
    n_contact = len(n_contact)

    # -----------------
    # Preparation Stage 
    # -----------------
    robot = build_robot(args.robot)

    # Robot Structure.
    tree = build_kinematics_tree(
        urdf_path=robot.urdf_path,
        active_joint_names=robot.get_active_joints()
    )
    #print("TREE", tree.link_name_to_id)

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
    # print("contact_parent_links", contact_parent_links)
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    # print("contact_parent_ids", contact_parent_ids)
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()

    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets)
    #print("dependency_matrix", dependency_matrix.shape)
    dependency_matrix = dependency_matrix.cuda()

    # Contact Field Acceleration Data Structure (LBVH-S2Bundle)
    accel_structure = contact_field.generate_acceleration_structure(method=cf_accel)
    print(len(accel_structure.contact_lbvhs))

    # Object Data. #If no mask provided then submesh defaults to original mesh
    object = MeshObject(object_mesh_path, object_mask_path)
    object_area = object.get_area()
    zo_lr = ((object_area / n_sample_point) ** 0.5) * zo_lr_sigma

    #Complete Object Points and Normals
    points, normals = object.sample_point_and_normal(count=n_sample_point, return_submesh=False)
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()
    support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0] #remove convex (difficult to grasp) points
    points = points_all[torch.where(support_point_mask)]            # good grasp point.
    normals = normals_all[torch.where(support_point_mask)]          

    #Graspable Object Points and Normals
    grasp_points, grasp_normals = object.sample_point_and_normal(count=n_sample_point, return_submesh=True)
    grasp_points_all = torch.from_numpy(grasp_points).cuda().float()
    grasp_normals_all = torch.from_numpy(grasp_normals).cuda().float()
    grasp_support_point_mask = get_support_point_mask(grasp_points_all, grasp_normals_all, [0.01])[0] #remove convex (difficult to grasp) points
    grasp_points = grasp_points_all[torch.where(grasp_support_point_mask)]            # good grasp point.
    grasp_normals = grasp_normals_all[torch.where(grasp_support_point_mask)] 

    #Object Data
    object_data = {'mesh_points': points, 
                   'mesh_normals': normals, 
                   'grasp_points': grasp_points,
                   'grasp_normals': grasp_normals}

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
    # Re: I agree with the original author and have decided to keep it this way too.

    print("Launch Inference")
    with torch.no_grad():

        # Object Placement: Fixed Rotation and Bounded Translation
        object_poses, condition = get_bounded_object_pose(
            n=batch_size_outer, 
            object_data=object_data,
            tree=tree, 
            mesh_data=decomposed_static_mesh_data,
            rot_vec=np.array([0.0, 0.0, 0.0]))
        print("object poses: ", object_poses.shape)

        # Contact Field BVH Traversal: poses x BVH patch x obj points (0 or -1)
        interaction_matrix_hand_point_idx = batch_object_all_contact_fields_interaction(
            object_pos=grasp_points, 
            object_normal=grasp_normals, 
            object_pose=object_poses, 
            accel_structure=accel_structure
        )    

        # Filtering to have only desired links involved in the grasp
        interaction_matrix = (interaction_matrix_hand_point_idx >= 0).int()
        link_interaction_matrix = contact_field.reduce_link_interaction(interaction_matrix)
        link_mask = torch.tensor(np.isin(contact_field.all_contact_link_names, n_contact_links), dtype=torch.int, device='cuda')
        link_interaction_matrix = link_interaction_matrix * link_mask[None, :, None]

        # Get Contact Domain
        contact_domain_pos, contact_domain_normal, contact_domain_point_idx, \
        object_poses, contact_link_ids, condition, valid_outer_idx = \
        sample_pose_and_contact_from_interaction(
            n_contact=n_contact,
            interaction_matrix=link_interaction_matrix, 
            dependency_matrix=dependency_matrix, 
            object_points=grasp_points, 
            object_normals=grasp_normals, 
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

    idx = 0
    while idx < n_result:
        print(f"Solution Number {idx+1}:")

        robot_mesh, robot_link_poses = viewer.get_mesh_fk(result['q'][idx:idx+1].detach().cpu().numpy(), visual=True)
        robot_link_poses = np.array(list(robot_link_poses.values()))
        geometries = []

        object_mesh = object.mesh.copy()
        res = result['object_pose'][idx].cpu().numpy()
        target_pos = result['target_pos'][idx].cpu().numpy()
        contact_link_id = result["contact_link_id"][idx].cpu().numpy()
        contact_link_poses = robot_link_poses[contact_link_id]
        local_contact_pos = result['contact_pos'][idx].cpu().numpy()
        contact_pos_world = (contact_link_poses[:, :3, :3] @ local_contact_pos[..., None]).squeeze(-1) + contact_link_poses[:, :3, 3]
        # print("Target Positions: ", target_pos)
        # print("Contact Positions: ", contact_pos_world)
        
        r = R.from_matrix(res[0:3, 0:3])
        print("Object Rotation Values", r.as_euler('xyz', degrees=True))
        print("Object Translation Values", res[:3, 3])
        
        object_mesh.apply_transform(result['object_pose'][idx].cpu().numpy())
        object_mesh_o3d = trimesh_to_open3d(object_mesh)

        material = load_material_from_mtl(object_mesh_path.replace(".obj", ".mtl"))
        object_mesh = {"name": 'object', "geometry": object_mesh_o3d, "material": material}
        geometries.append(object_mesh)

        # # Create visuals for target contact points (Green Spheres)
        # target_material = o3d.visualization.rendering.MaterialRecord()
        # target_material.base_color = [0.0, 1.0, 0.0, 1.0]  # Red
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

        # Create visuals for final contact points (Red Spheres)
        contact_material = o3d.visualization.rendering.MaterialRecord()
        contact_material.base_color = [1.0, 0.0, 0.0, 1.0]  # Red
        contact_material.shader = "defaultLit"
        for i, point in enumerate(contact_pos_world):
            # Create sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003, resolution=20)
            sphere.compute_vertex_normals()
            sphere.translate(point)
            sphere_dict = {
                "name": f"Contact_{i}",
                "geometry": sphere,
                "material": contact_material
            }
            geometries.append(sphere_dict)

        viewer.show(robot_mesh + geometries)

        if input("Continue? (Y/n)") in ['n', 'N']:
            break
        idx+= 1


if __name__ == '__main__':
    args = get_args()
    main(args)