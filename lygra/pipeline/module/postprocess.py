# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm 
from lygra.pipeline.module.collision import batch_filter_collision
import torch
from lygra.utils.transform_utils import batch_object_transform
from lygra.utils.geom_utils import get_plane_surface_points_batch
import trimesh
import numpy as np

def generate_translated_clouds(ply_path, object_poses):
    """
    Loads a PLY point cloud and applies M translations extracted from 
    object_poses matrices.
    
    Args:
        ply_path (str): Path to the .ply file.
        object_poses (np.ndarray): Array of shape (M, 4, 4).
        
    Returns:
        np.ndarray: Array of shape (M, N, 3).
    """
    
    # 1. Load the PLY file
    # trimesh.load handles .ply files automatically
    pcd = trimesh.load(ply_path)
    
    # Ensure we have the vertices as a numpy array (N, 3)
    # Note: Depending on the file, it might load as a Scene or PointCloud. 
    # This ensures we get the vertices regardless.
    if isinstance(pcd, trimesh.Scene):
        # If it loads as a scene, dump all geometry vertices into one array
        vertices = np.vstack([g.vertices for g in pcd.geometry.values()])
    else:
        # If it loads as a PointCloud or Trimesh
        vertices = np.array(pcd.vertices)
    
    print(f"Loaded {vertices.shape[0]} vertices.")

    # 2. Extract Translation Vectors
    # Take all M rows, first 3 rows, 4th column -> Shape (M, 3)
    translations = object_poses[:, :3, 3]
    
    # 3. Apply Translations using Broadcasting
    # We want to add (M, 3) translations to (N, 3) vertices to get (M, N, 3)
    
    # Reshape vertices to (1, N, 3)
    vertices_expanded = vertices[None, :, :]
    
    # Reshape translations to (M, 1, 3)
    translations_expanded = translations[:, None, :]
    
    # NumPy broadcasts dimensions 1 and M against each other
    translated_clouds = vertices_expanded + translations_expanded
    
    return translated_clouds

def batch_generate_bg_surface(
   result,
   object_point,     
):
    object_pose = result["object_pose"]
    #print("object pose shape", object_pose.shape)
    batch_size = object_pose.shape[0]
    #print("batch size", batch_size)
    object_point = batch_object_transform(object_pose, object_point)["pos"]
    #min_vals = torch.amin(object_point[:, :, 1], dim=1)

    min_indices = torch.argmin(object_point[:, :, 1], dim=1)
    m_indices = torch.arange(object_point.shape[0], device=object_point.device)
    min_points = object_point[m_indices, min_indices]
    
    surface_points = get_plane_surface_points_batch(a=torch.zeros(batch_size), b=torch.ones(batch_size), 
                                                    c=torch.zeros(batch_size), d=-1 * min_points[:,1], 
                                                    center_point=min_points)
    return surface_points


def batch_assign_free_finger_and_filter(
    tree,
    result,
    object_point,
    self_collision_link_pairs,
    decomposed_mesh_data,
    n_assign_retry=5
):
    assigned_results = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            assigned_results[k] = []
        else:
            assigned_results[k] = v 

    joint_limit_lower, joint_limit_upper = tree.get_active_joint_limit()
    joint_limit_lower = torch.from_numpy(joint_limit_lower).to(object_point.device)
    joint_limit_upper = torch.from_numpy(joint_limit_upper).to(object_point.device)
    
    tmp_result = {k: v for k, v in result.items()}

    n_initial = 0

    for i in tqdm(range(n_assign_retry), desc="Postprocessing"):
        q = tmp_result["q"]
        q_mask = tmp_result["q_mask"].float()
       
        free_q = torch.zeros_like(q) #torch.rand_like(q) * (joint_limit_upper - joint_limit_lower) + joint_limit_lower
        tmp_result["q"] = free_q * (1 - q_mask) + q * q_mask # rand_value * unassigned_joints + fixed_value * assigned_joints
        
        success_mask = batch_filter_collision(
            tree,
            tmp_result["q"],
            tmp_result["object_pose"],
            object_point,
            self_collision_link_pairs,
            decomposed_mesh_data,
            ret_mask_only=True,
            surface_points=tmp_result["surface_points"]
        )
       
        for k, v in tmp_result.items():
            if isinstance(v, torch.Tensor):
                assigned_results[k].append(tmp_result[k][torch.where(success_mask)])
                tmp_result[k] = tmp_result[k][torch.where(success_mask.logical_not())]

        if len(tmp_result["q"]) == 0:
            break

        if q_mask.bool().all():
            break 

        n_success = success_mask.int().sum().item()
        if i == 0:
            n_initial = n_success
        else:
            if n_success < 0.1 * n_initial:
                # benefit from extra rounds is marginal.
                break

    for k, v in assigned_results.items():
        if isinstance(result[k], torch.Tensor):
            assigned_results[k] = torch.cat(assigned_results[k], dim=0)

    return assigned_results