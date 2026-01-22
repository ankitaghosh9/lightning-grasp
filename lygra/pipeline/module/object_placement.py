# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np 
from lygra.utils.geom_utils import get_tangent_plane
from lygra.pipeline.module.collision import batch_object_hand_collision_check
from scipy.spatial.transform import Rotation as R

def get_object_pose_sampling_args(strategy, robot):
    args = {
        "strategy": strategy
    }

    if strategy == 'canonical':
        bmin, bmax = robot.get_canonical_space()
        args['bmin'] = bmin 
        args['bmax'] = bmax

    return args


def get_align_transform(obj_pos, obj_normal, robot_pos, robot_normal):
    obj_transform = np.eye(4)
    obj_x, obj_y = get_tangent_plane(obj_normal)
    obj_transform[:3, 0] = obj_x 
    obj_transform[:3, 1] = obj_y 
    obj_transform[:3, 2] = obj_normal
    obj_transform[:3, 3] = obj_pos 

    robot_transform = np.eye(4)
    robot_x, robot_y = get_tangent_plane(-robot_normal)
    robot_transform[:3, 0] = robot_x
    robot_transform[:3, 1] = robot_y 
    robot_transform[:3, 2] = -robot_normal 
    robot_transform[:3, 3] = robot_pos
    return robot_transform @ np.linalg.inv(obj_transform)


def sample_object_pose_kernel(
    n, 
    points, 
    normals, 
    contact_field,
    tree, 
    mesh_data, 
    sampling_args,
    use_static_prob=0.5
):
    """
    Args:
        points:  [N_point, 3]
        normals: [N_point, 3]
        ...

    Returns:
        ...
    """

    result = []
    result_extra_contact_pos = []
    result_extra_contact_normal = []
    result_extra_contact_mask = []

    n_static = int(n * use_static_prob)
    n_non_static = n - n_static

    robot_contact = contact_field.sample_spatial_contact(n_non_static, sampling_args)
    
    sample_idx = np.random.randint(0, len(points), n_non_static)
    sampled_object_contact_points = points[sample_idx]
    sampled_object_contact_normals = normals[sample_idx]

    """
    TODO: rewrite all these nonsense
    """


    if isinstance(sampled_object_contact_normals, torch.Tensor):
        sampled_object_contact_points = sampled_object_contact_points.detach().cpu().numpy()
        sampled_object_contact_normals = sampled_object_contact_normals.detach().cpu().numpy()

    for i in range(n_non_static):
        transform = get_align_transform(
            sampled_object_contact_points[i],
            sampled_object_contact_normals[i],
            robot_contact[i, :3],
            robot_contact[i, 3:]
        )
        result.append(transform)
        result_extra_contact_mask.append(np.array([0.0]))
        result_extra_contact_pos.append(np.zeros((1, 3)))
        result_extra_contact_normal.append(np.zeros((1, 3)))

    # we also need to filter these.
    if n_static > 0 and contact_field.sample_static_contact_geometry(1) is not None:
        sample_idx = np.random.randint(0, len(points), n_static)
        sampled_object_contact_points = points[sample_idx]
        sampled_object_contact_normals = normals[sample_idx]

        if isinstance(sampled_object_contact_normals, torch.Tensor):
            sampled_object_contact_points = sampled_object_contact_points.detach().cpu().numpy()
            sampled_object_contact_normals = sampled_object_contact_normals.detach().cpu().numpy()

        robot_contact = contact_field.sample_static_contact_geometry(n_static)

        for i in range(n_static):
            result.append(
                get_align_transform(
                    sampled_object_contact_points[i],
                    sampled_object_contact_normals[i],
                    robot_contact[i, :3],
                    robot_contact[i, 3:]
                )
            )
            result_extra_contact_mask.append(np.array([1.0]))
            result_extra_contact_pos.append(robot_contact[i:i+1, :3])
            result_extra_contact_normal.append(robot_contact[i:i+1, 3:])

    # Ensure no penetration
    result = torch.from_numpy(np.array(result)).cuda().float() # [n, 4, 4]
    
    points = points.unsqueeze(0).expand(result.shape[0], -1, -1)
    rotation = result[:, :3, :3]
    translation = result[:, :3, 3]  # [n, 3]

    point_transformed = torch.bmm(points, rotation.transpose(-1, -2)) + translation.unsqueeze(1)    # [n, n_point, 3]

    no_collision = batch_object_hand_collision_check(
        tree=tree,
        mesh=mesh_data,
        object_point=point_transformed
    )

    selected_idx = torch.where(no_collision)
    out = result[selected_idx]

    condition = {
        "extra_contact_pos": torch.from_numpy(np.array(result_extra_contact_pos)).cuda()[selected_idx].float(),
        "extra_contact_normal": torch.from_numpy(np.array(result_extra_contact_normal)).cuda()[selected_idx].float(),
        "extra_contact_mask": torch.from_numpy(np.array(result_extra_contact_mask)).cuda()[selected_idx].float()
    }
    return out, condition


def sample_object_pose(
    n, 
    points, 
    normals, 
    contact_field, 
    tree, 
    mesh_data, 
    sampling_args, 
    use_static_prob=0.5
):
    """
    TODO(): batch it please.
    Args:
        points:  [N_point, 3]
        normals: [N_point, 3]
    
    Returns:
        ...
    """

    n_sol = 0
    all_result_pose = []
    all_result_condition = None

    while n_sol < n:
        pose, condition = sample_object_pose_kernel(
            n, points, normals, 
            contact_field, tree, mesh_data, 
            sampling_args=sampling_args,
            use_static_prob=use_static_prob
        )

        all_result_pose.append(pose)
        if all_result_condition is None:
            all_result_condition = condition
        else:
            for k, v in all_result_condition.items():
                all_result_condition[k] = torch.cat((v, condition[k]), dim=0)

        n_sol += len(pose)

    all_result_pose = torch.cat(all_result_pose, dim=0)
    return all_result_pose, all_result_condition

def get_bounded_object_pose(n, object_data, tree, mesh_data, 
                            rot_vec=None, 
                            bounds={'x': [-0.03, 0.07], 
                                    'y': [-0.13, -0.03], 
                                    'z': [0.05, 0.15]}):
    """
    Generates 'n' valid object poses by sampling random translations within bounds 
    and performing collision checks.

    Args:
        n (int): Desired number of valid poses.
        object_data: Dictionary containing object mesh data (must have 'object_points').
        tree: Robot kinematic tree (for collision check).
        mesh_data: Robot mesh data
        rot_vec: Rotation input (fixed for all samples). Can be:
                 - None (defaults to Identity)
                 - Euler angles (1x3) [roll, pitch, yaw]
                 - Quaternion (1x4)
                 - Rotation Matrix (3x3)
        bounds: Dictionary or list defining sampling ranges.
                Format: {'x': [min, max], 'y': [min, max], 'z': [min, max]}
                Defaults to a small box above the origin if None.

    Returns:
        batch_poses (torch.Tensor): [n, 4, 4] CUDA tensor of valid poses.
        condition (dict): Dummy condition dictionary (zeros).
    """

    # --- 1. Setup Rotation (Fixed for all samples) ---
    if rot_vec is None:
        r_matrix = np.eye(3)
    else:
        if isinstance(rot_vec, torch.Tensor):
            rot_vec = rot_vec.cpu().numpy()
        rot_val = np.array(rot_vec)
        
        # Determine rotation format
        if rot_val.shape == (3, 3):
            r_matrix = rot_val
        elif rot_val.size == 4:
            r_matrix = R.from_quat(rot_val.flatten()).as_matrix()
        elif rot_val.size == 3:
            r_matrix = R.from_euler('xyz', rot_val.flatten()).as_matrix()
        else:
            raise ValueError(f"rot_vec has unsupported shape: {rot_val.shape}")

    # Convert rotation to CUDA tensor once
    r_tensor = torch.from_numpy(r_matrix).float()
    if torch.cuda.is_available():
        r_tensor = r_tensor.cuda()

    # --- 2. Setup Bounds ---
    # Extract limits into tensors for fast batch sampling
    b_min = torch.tensor([bounds['x'][0], bounds['y'][0], bounds['z'][0]]).float()
    b_max = torch.tensor([bounds['x'][1], bounds['y'][1], bounds['z'][1]]).float()
    
    if torch.cuda.is_available():
        b_min = b_min.cuda()
        b_max = b_max.cuda()

    # --- 3. Sampling & Collision Loop ---
    # We loop until we have collected exactly 'n' valid poses
    collected_poses = []
    num_collected = 0
    
    # Get object points for collision check [N_points, 3]
    obj_points = object_data['mesh_points']
    if isinstance(obj_points, np.ndarray):
        obj_points = torch.from_numpy(obj_points).float()
    if torch.cuda.is_available():
        obj_points = obj_points.cuda()

    # Safety to prevent infinite loops if bounds are invalid (e.g., inside the palm)
    max_retries = 10
    retry_count = 0
    while num_collected < n and retry_count < max_retries:
        # How many more do we need? Oversample slightly (1.5x) to account for collisions
        needed = n - num_collected
        batch_size = int(needed * 1.5) if needed > 10 else needed * 2
        
        # A. Sample Random Translations [batch_size, 3]
        # Formula: rand * (max - min) + min
        t_samples = torch.rand(batch_size, 3, device=b_min.device) * (b_max - b_min) + b_min

        # B. Construct Poses [batch_size, 4, 4]
        current_poses = torch.eye(4, device=b_min.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        current_poses[:, :3, :3] = r_tensor  # Set rotation
        current_poses[:, :3, 3] = t_samples  # Set translation

        # C. Transform Object Points for Collision Check
        # Points: [1, N_pts, 3] -> [B, N_pts, 3]
        # R_transpose is needed because we are transforming points: p_new = R * p + t
        # Equivalent to: (p @ R.T) + t
        rot_batch = current_poses[:, :3, :3] # [B, 3, 3]
        trans_batch = current_poses[:, :3, 3].unsqueeze(1) # [B, 1, 3]
        
        # Batch Matmul: [B, N_pts, 3]
        transformed_pts = torch.matmul(obj_points.unsqueeze(0), rot_batch.transpose(-1, -2)) + trans_batch
        #point_transformed = torch.bmm(points, rotation.transpose(-1, -2)) + translation.unsqueeze(1)

        # D. Perform Collision Check
        # Returns a boolean mask of valid poses
        no_collision = batch_object_hand_collision_check(
            tree=tree,
            mesh=mesh_data,
            object_point=transformed_pts
        )

        # E. Filter and Collect
        selected_idx = torch.where(no_collision)
        valid_batch = current_poses[selected_idx]
        
        if len(valid_batch) > 0:
            collected_poses.append(valid_batch)
            num_collected += len(valid_batch)
        
        retry_count += 1

    # --- 4. Finalize Output ---
    if num_collected < n:
        raise RuntimeError(f"Could only find {num_collected}/{n} valid poses after {max_retries} retries. "
                           "Your bounds might be overlapping the hand mesh too much.")

    # Concatenate all valid batches and slice to exact size 'n'
    final_poses = torch.cat(collected_poses, dim=0)[:n]

    # Create dummy conditions (must match final batch size n)
    device = final_poses.device
    condition = {
        "extra_contact_pos": torch.zeros((n, 1, 3), device=device, dtype=torch.float32),
        "extra_contact_normal": torch.zeros((n, 1, 3), device=device, dtype=torch.float32),
        "extra_contact_mask": torch.zeros((n, 1), device=device, dtype=torch.float32)
    }

    return final_poses, condition