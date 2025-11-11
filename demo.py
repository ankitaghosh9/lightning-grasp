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
    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])

    contact_parent_links = contact_field.get_all_parent_link_names()
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()

    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets)
    dependency_matrix = dependency_matrix.cuda()

    # Contact Field Acceleration Data Structure (LBVH-S2Bundle)
    accel_structure = contact_field.generate_acceleration_structure(method=cf_accel)

    # Object Data.
    object = MeshObject(object_mesh_path)
    object_area = object.get_area()
    zo_lr = ((object_area / n_sample_point) ** 0.5) * zo_lr_sigma
    points, normals = object.sample_point_and_normal(count=n_sample_point)
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()

    # Filtering
    support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0]
    points = points_all[torch.where(support_point_mask)]            # good grasp point.
    normals = normals_all[torch.where(support_point_mask)]          # good_grasp_point.

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