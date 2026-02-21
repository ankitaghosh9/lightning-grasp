# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
import torch.nn.functional as F
import trimesh
import numpy as np


class MeshObject:
    def __init__(self, obj_path, object_mask_path=None, scale=1.0):
        self.mesh = trimesh.load(obj_path, process=False)
        if isinstance(self.mesh, trimesh.Scene):
            print(f"Merging {len(self.mesh.geometry)} sub-meshes into one...")
            # 'concatenate=True' merges all sub-meshes into one single Trimesh object
            self.mesh = self.mesh.dump(concatenate=True)
        scale_matrix = trimesh.transformations.scale_matrix(scale, [0,0,0])
        self.mesh.apply_transform(scale_matrix)

        if object_mask_path is not None:
            self.mask = np.load(object_mask_path)
            valid_face_mask = np.all(~self.mask[self.mesh.faces], axis=1)  
            self.submesh = self.mesh.submesh([valid_face_mask], append=True)
        else:
            self.mask = None
            self.submesh = self.mesh.copy()

    def sample_point_and_normal(self, count=2000, return_submesh=False):
        if return_submesh:
            points, face_indices = trimesh.sample.sample_surface(self.submesh, count=count)
            normals = self.submesh.face_normals[face_indices]
        else:
            points, face_indices = trimesh.sample.sample_surface(self.mesh, count=count)
            normals = self.mesh.face_normals[face_indices]
        return points, normals 

    def get_area(self, return_submesh=False):
        if return_submesh:
            return self.submesh.area
        else:
            return self.mesh.area 


def get_tangent_plane(batch_vector):
    ''' Get the tangent planes of vectors.
    
    Args:
        batch_vector: [..., 3] (torch.Tensor or np.ndarray)
    
    Returns:
        x:            [..., 3]
        y:            [..., 3]
    '''
    if isinstance(batch_vector, torch.Tensor):
        shape = batch_vector.shape[:-1]
        batch_vector = F.normalize(batch_vector.reshape(-1, 3), dim=-1)
        x = batch_vector + torch.ones_like(batch_vector) * 2
        y = torch.cross(batch_vector, F.normalize(x, dim=-1), dim=-1)
        x = torch.cross(y, batch_vector, dim=-1)

        return x.reshape(*shape, 3), y.reshape(*shape, 3)
    else:
        # numpy
        shape = batch_vector.shape[:-1]
        
        batch_vector = np.reshape(batch_vector, (-1, 3))
        norm = np.linalg.norm(batch_vector, axis=-1, keepdims=True)
        batch_vector = batch_vector / np.clip(norm, 1e-8, None)

        x = batch_vector + 2.0  # broadcast with scalar
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.clip(x_norm, 1e-8, None)

        y = np.cross(batch_vector, x)
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        y = y / np.clip(y_norm, 1e-8, None)

        x = np.cross(y, batch_vector)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.clip(x_norm, 1e-8, None)

        
        return x.reshape(*shape, 3), y.reshape(*shape, 3)

import torch

def get_plane_surface_points_batch(a, b, c, d, width=0.4, resolution=0.005, device='cuda'):
    """
    Generates a batch of dense point clouds representing planes defined by ax + by + cz + d = 0.
    
    Args:
        a, b, c, d: Tensors of shape (B,) or (B, 1) representing plane coefficients.
        width: Scalar, size of the plane.
        resolution: Scalar, spacing between points.
        device: Torch device.
        
    Returns:
        torch.Tensor: [B, N, 3] tensor of points, where B is batch size.
    """
    # 1. Standardization: Ensure inputs are 1D tensors (Batch Size)
    # We allow inputs to be lists or scalars, converting them to tensors
    def to_tensor(x):
        t = torch.as_tensor(x, device=device, dtype=torch.float32)
        return t.flatten()
    
    a, b, c, d = to_tensor(a), to_tensor(b), to_tensor(c), to_tensor(d)
    
    # Check consistent batch size
    batch_size = a.shape[0]
    assert b.shape[0] == batch_size and c.shape[0] == batch_size and d.shape[0] == batch_size
    
    # 2. Generate Base Grid (Z=0) - Created once, expanded later
    steps = int(width / resolution)
    range_t = torch.linspace(-width/2, width/2, steps, device=device)
    grid_x, grid_y = torch.meshgrid(range_t, range_t, indexing='xy')
    
    # Shape: [1, N, 3]
    base_points = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        torch.zeros_like(grid_x.flatten())
    ], dim=1).unsqueeze(0) 
    
    # 3. Process Normals
    # Stack coefficients -> [B, 3]
    normals = torch.stack([a, b, c], dim=1)
    norm_mags = torch.norm(normals, dim=1, keepdim=True)
    
    # Avoid division by zero
    norm_mags = torch.where(norm_mags < 1e-8, torch.ones_like(norm_mags), norm_mags)
    normals = normals / norm_mags  # [B, 3]
    
    # 4. Compute Rotation Matrices (Batch Rodrigues)
    # We align Z-axis (0,0,1) to the target Normals
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch_size, 3)
    
    # Cross Product: [B, 3]
    rot_axis = torch.linalg.cross(z_axis, normals)
    rot_sin = torch.norm(rot_axis, dim=1, keepdim=True) # [B, 1]
    rot_cos = torch.sum(z_axis * normals, dim=1, keepdim=True) # Dot product [B, 1]
    
    # -- Handling Singularities (Parallel vectors) --
    # Mask where sin(theta) is near zero (vectors are parallel or anti-parallel)
    is_singular = rot_sin.squeeze() < 1e-6
    
    # Normalize axis safely (avoid NaN for singular cases)
    # We add 1e-8 to denominator just for the math, but will overwrite these indices later
    axis_norm = rot_axis / (rot_sin + 1e-8)
    
    # Construct Skew-Symmetric Matrices K: [B, 3, 3]
    # K = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    zeros = torch.zeros_like(axis_norm[:, 0])
    K = torch.stack([
        torch.stack([zeros, -axis_norm[:, 2], axis_norm[:, 1]], dim=1),
        torch.stack([axis_norm[:, 2], zeros, -axis_norm[:, 0]], dim=1),
        torch.stack([-axis_norm[:, 1], axis_norm[:, 0], zeros], dim=1)
    ], dim=1)
    
    # Identity Matrix: [B, 3, 3]
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3)
    
    # Rodrigues Formula: R = I + sin*K + (1-cos)*K^2
    # [B, 3, 3]
    R = I + (rot_sin.unsqueeze(-1) * K) + ((1 - rot_cos.unsqueeze(-1)) * (K @ K))
    
    # -- Fix Singularities --
    # If cos > 0 (Parallel): R = Identity
    # If cos < 0 (Anti-Parallel): R = 180 deg flip (e.g. diagonal 1, -1, -1)
    if is_singular.any():
        # Indices where singular
        idx_sing = torch.where(is_singular)[0]
        
        # Check cosine sign for these indices
        cos_vals = rot_cos[idx_sing].squeeze()
        
        # Create Identity and Flip matrices
        I_sing = torch.eye(3, device=device).unsqueeze(0).expand(len(idx_sing), 3, 3)
        Flip_sing = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 
                                 dtype=torch.float32, device=device).unsqueeze(0).expand(len(idx_sing), 3, 3)
        
        # Choose based on cos > 0
        R_corrected = torch.where(cos_vals.view(-1, 1, 1) > 0, I_sing, Flip_sing)
        
        # Overwrite in main R tensor
        R[idx_sing] = R_corrected

    # 5. Rotate Points
    # Base: [1, N, 3] -> Expand to [B, N, 3]
    points_expanded = base_points.expand(batch_size, -1, -1)
    
    # Batch Matrix Multiplication: (B, N, 3) @ (B, 3, 3)^T
    # We transpose R to (B, 3, 3) -> (B, 3, 3) for correct multiplication on the right
    rotated_points = torch.bmm(points_expanded, R.transpose(1, 2))
    
    # 6. Translate Points
    # Distance from origin D = -d / |normal| (but normal is already normalized)
    # dist: [B, 1]
    d_vals = d.view(-1, 1)
    dist_from_origin = -d_vals / norm_mags
    
    # Translation vector: normal * dist
    # [B, 3] -> [B, 1, 3] for broadcasting
    translation = (normals * dist_from_origin).unsqueeze(1)
    
    final_points = rotated_points + translation
    
    return final_points

def get_plane_surface_points(a, b, c, d, width=0.4, resolution=0.005, device='cuda'):
    """
    Generates a dense point cloud representing a plane defined by ax + by + cz + d = 0.
    
    Args:
        a, b, c, d: Coefficients of the plane equation.
        width: The size of the plane (width x width) in meters.
        resolution: Spacing between points (smaller = denser collision check).
        device: Torch device (e.g., 'cuda' or 'cpu').
        
    Returns:
        torch.Tensor: [N, 3] tensor of points on the plane.
    """
    # 1. Generate a base grid on the XY plane (z=0) centered at origin
    steps = int(width / resolution)
    range_t = torch.linspace(-width/2, width/2, steps, device=device)
    grid_x, grid_y = torch.meshgrid(range_t, range_t, indexing='xy')
    
    # Flatten to create a list of points: [x, y, 0]
    base_points = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        torch.zeros_like(grid_x.flatten())
    ], dim=1)  # Shape: [N, 3]

    # 2. Compute the Normal Vector of the target plane
    normal = torch.tensor([a, b, c], dtype=torch.float32, device=device)
    norm_magnitude = torch.norm(normal)
    
    if norm_magnitude < 1e-6:
        raise ValueError("Plane normal vector (a, b, c) cannot be zero.")
    
    normal = normal / norm_magnitude  # Normalize
    
    # 3. Compute Rotation Matrix to align Z-axis (0,0,1) with Plane Normal
    # We want to rotate "up" (0,0,1) to match (a,b,c)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Axis of rotation = Cross Product (z_axis x normal)
    rot_axis = torch.cross(z_axis, normal, dim=0)
    rot_sin = torch.norm(rot_axis)
    rot_cos = torch.dot(z_axis, normal)
    
    # Rotation Matrix calculation
    if rot_sin < 1e-6:
        # Normal is parallel to Z-axis
        if rot_cos > 0:
            R = torch.eye(3, device=device) # Already aligned
        else:
            # 180 degree flip around X axis
            R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=device) 
    else:
        rot_axis = rot_axis / rot_sin
        K = torch.tensor([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ], device=device)
        
        # Rodrigues' rotation formula: I + sin(theta)K + (1-cos(theta))K^2
        R = torch.eye(3, device=device) + rot_sin * K + (1 - rot_cos) * (K @ K)

    # 4. Rotate the points
    # (Using matrix multiplication: Points @ R.T)
    rotated_points = base_points @ R.T

    # 5. Translate the plane
    # The distance from origin to the plane along the normal is -d / |(a,b,c)|
    # We shift the points along the normal vector by this distance.
    distance_from_origin = -d / norm_magnitude
    translation = normal * distance_from_origin
    
    final_points = rotated_points + translation

    return final_points