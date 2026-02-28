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
            valid_face_mask = np.all(self.mask[self.mesh.faces], axis=1)  
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

def get_plane_surface_points_batch(a, b, c, d, center_point=None, width=0.4, resolution=0.005, device='cuda'):
    """
    Generates a batch of dense point clouds representing planes.
    
    Args:
        a, b, c, d: Tensors of shape (B,) or (B, 1) representing plane coefficients.
        center_point: Optional Tensor of shape (B, 3) or (3,). If provided, this point
                      becomes the center of the generated plane, overriding 'd'.
        width: Scalar, size of the plane.
        resolution: Scalar, spacing between points.
        device: Torch device.
        
    Returns:
        torch.Tensor: [B, N, 3] tensor of points, where B is batch size.
    """
    # 1. Standardization: Ensure inputs are 1D tensors (Batch Size)
    def to_tensor(x):
        t = torch.as_tensor(x, device=device, dtype=torch.float32)
        return t.flatten()
    
    a, b, c, d = to_tensor(a), to_tensor(b), to_tensor(c), to_tensor(d)
    
    # Check consistent batch size
    batch_size = a.shape[0]
    # Note: We rely on a,b,c for batch size. d is checked only if we use it later.
    
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
    
    # Normalize axis safely
    axis_norm = rot_axis / (rot_sin + 1e-8)
    
    # Construct Skew-Symmetric Matrices K: [B, 3, 3]
    zeros = torch.zeros_like(axis_norm[:, 0])
    K = torch.stack([
        torch.stack([zeros, -axis_norm[:, 2], axis_norm[:, 1]], dim=1),
        torch.stack([axis_norm[:, 2], zeros, -axis_norm[:, 0]], dim=1),
        torch.stack([-axis_norm[:, 1], axis_norm[:, 0], zeros], dim=1)
    ], dim=1)
    
    # Identity Matrix: [B, 3, 3]
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3)
    
    # Rodrigues Formula: R = I + sin*K + (1-cos)*K^2
    R = I + (rot_sin.unsqueeze(-1) * K) + ((1 - rot_cos.unsqueeze(-1)) * (K @ K))
    
    # -- Fix Singularities --
    if is_singular.any():
        idx_sing = torch.where(is_singular)[0]
        cos_vals = rot_cos[idx_sing].squeeze()
        I_sing = torch.eye(3, device=device).unsqueeze(0).expand(len(idx_sing), 3, 3)
        Flip_sing = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 
                                 dtype=torch.float32, device=device).unsqueeze(0).expand(len(idx_sing), 3, 3)
        R_corrected = torch.where(cos_vals.view(-1, 1, 1) > 0, I_sing, Flip_sing)
        R[idx_sing] = R_corrected

    # 5. Rotate Points
    # Base: [1, N, 3] -> Expand to [B, N, 3]
    points_expanded = base_points.expand(batch_size, -1, -1)
    
    # Batch Matrix Multiplication
    rotated_points = torch.bmm(points_expanded, R.transpose(1, 2))
    
    # 6. Translate Points
    if center_point is not None:
        # --- NEW LOGIC ---
        # If a center point is provided, we translate the rotated grid (which is currently centered at 0,0,0)
        # to the specified center_point.
        
        # Ensure center_point is a tensor on the correct device
        if not isinstance(center_point, torch.Tensor):
             center_point = torch.tensor(center_point, device=device, dtype=torch.float32)
        
        # Handle shape: If (3,), reshape to (1, 3) then expand to (B, 3)
        if center_point.ndim == 1:
            center_point = center_point.view(1, 3).expand(batch_size, 3)
        
        # Expand for broadcasting: [B, 3] -> [B, 1, 3]
        translation = center_point.unsqueeze(1)
        
    else:
        # --- OLD LOGIC ---
        # Use 'd' to calculate distance from origin
        d_vals = d.view(-1, 1)
        dist_from_origin = -d_vals / norm_mags
        translation = (normals * dist_from_origin).unsqueeze(1)
    
    final_points = rotated_points + translation
    
    return final_points