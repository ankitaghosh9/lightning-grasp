import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from trimesh.visual import ColorVisuals
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

def lightning_to_isaac_transform(matrix):
    """
    Converts a matrix from Lightning Grasp (Y-up) to Isaac Sim (Z-up).
    Accepts either a 4x4 Transformation Matrix or a 3x3 Rotation Matrix.
    
    Args:
        matrix (np.ndarray): Shape (4,4) or (3,3).
        
    Returns:
        np.ndarray: Converted matrix of the same shape.
    """
    matrix = np.array(matrix) # Ensure input is numpy array
    
    # Rotation: +90 degrees around X-axis
    # (Y-up becomes Z-up)
    R_L_to_I = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    
    if matrix.shape == (3, 3):
        # Case 1: 3x3 Rotation Matrix
        return R_L_to_I @ matrix
        
    elif matrix.shape == (4, 4):
        # Case 2: 4x4 Transformation Matrix
        T_L_to_I = np.eye(4)
        T_L_to_I[:3, :3] = R_L_to_I
        return T_L_to_I @ matrix
        
    else:
        raise ValueError(f"Input must be 3x3 or 4x4 matrix. Got shape {matrix.shape}")


def isaac_to_lightning_transform(matrix):
    """
    Converts a matrix from Isaac Sim (Z-up) to Lightning Grasp (Y-up).
    Accepts either a 4x4 Transformation Matrix or a 3x3 Rotation Matrix.
    
    Args:
        matrix (np.ndarray): Shape (4,4) or (3,3).
        
    Returns:
        np.ndarray: Converted matrix of the same shape.
    """
    matrix = np.array(matrix) # Ensure input is numpy array
    
    # Rotation: -90 degrees around X-axis
    # (Z-up becomes Y-up)
    R_I_to_L = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ])
    
    if matrix.shape == (3, 3):
        # Case 1: 3x3 Rotation Matrix
        return R_I_to_L @ matrix
        
    elif matrix.shape == (4, 4):
        # Case 2: 4x4 Transformation Matrix
        T_I_to_L = np.eye(4)
        T_I_to_L[:3, :3] = R_I_to_L
        return T_I_to_L @ matrix
        
    else:
        raise ValueError(f"Input must be 3x3 or 4x4 matrix. Got shape {matrix.shape}")

def load_obj_rot_from_isaac(file_path):
    """
    Loads quaternion from JSON, converts to 3x3 matrix, 
    and subtracts 90 degrees from the X-axis rotation.
    """
    # 1. Load the R values from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract quaternion components [w, x, y, z]
    q_data = data["R"]
    quat = [q_data["x"], q_data["y"], q_data["z"], q_data["w"]] # SciPy uses [x, y, z, w]
    
    # 2. Convert to 3x3 Rotation Matrix
    # We create a Rotation object from the quaternion
    rotation_obj = R.from_quat(quat)
    matrix_3x3 = rotation_obj.as_matrix()
    
    # 3. Transform to lightning grasp orientation
    rot_matrix = isaac_to_lightning_transform(matrix_3x3)
    
    return rot_matrix

def load_obj_T_from_json(file_path):
    """
    Loads translation values from a JSON file and returns them as a 1x3 numpy array.
    
    Args:
        file_path (str): Path to the .json file.
        
    Returns:
        np.ndarray: A 1x3 array containing [x, y, z].
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract translation dictionary
    t_data = data["T"]
    # Create numpy array in order [x, y, z]
    translation_array = np.array([[t_data["x"], t_data["y"], t_data["z"]]])
    
    return translation_array

def generate_geodesic_confidence_mask(mesh, contact_pos_world, object_pose, contact_link_names, 
                                      sigma=0.05, max_dist=0.05, device='cuda'):
    """
    Generates a Geodesic confidence mask where confidence is:
    - 1.0 at the contact point.
    - Gaussian curve in between.
    - Exactly 0.0 at max_dist.
    - 0.0 everywhere beyond max_dist.
    """
    
    # --- 1. Preparation ---
    if isinstance(contact_pos_world, torch.Tensor):
        contact_pos_world = contact_pos_world.detach().cpu().numpy()
    if isinstance(object_pose, torch.Tensor):
        object_pose = object_pose.detach().cpu().numpy()

    # Transform contacts FROM World TO Object Frame
    R = object_pose[:3, :3]
    t = object_pose[:3, 3]
    R_inv = R.T
    contact_pos_local = (contact_pos_world - t) @ R_inv.T

    # --- 2. Snap Contacts to Nearest Mesh Vertices ---
    _, vertex_indices = mesh.nearest.vertex(contact_pos_local)

    # --- 3. Build the Mesh Graph ---
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    
    graph = sp.coo_matrix(
        (length, (edges[:, 0], edges[:, 1])), 
        shape=(len(mesh.vertices), len(mesh.vertices))
    )
    graph = graph + graph.T

    # --- 4. Run Dijkstra's Algorithm ---
    # We use limit=max_dist to stop searching early for performance
    distances_matrix = dijkstra(
        csgraph=graph, 
        directed=False, 
        indices=vertex_indices,
        limit=max_dist 
    )

    # --- 5. Find Min Distance for each Vertex ---
    min_geodesic_dists = distances_matrix

    # --- 6. Apply Normalized Gaussian Decay ---
    
    # Initialize all confidence to 0.0
    confidence_values = np.zeros_like(min_geodesic_dists)
    
    # Mask for valid points within range
    valid_mask = (min_geodesic_dists != np.inf) & (min_geodesic_dists <= max_dist)
    
    if np.any(valid_mask):
        # A. Calculate the "Floor" value (The Gaussian value at exactly max_dist)
        # We need to subtract this so the curve touches 0 at the limit
        floor_val = np.exp(-(max_dist**2) / (2 * sigma**2))
        
        # B. Calculate raw Gaussian for valid points
        raw_vals = np.exp(-(min_geodesic_dists[valid_mask]**2) / (2 * sigma**2))
        
        # C. Shift and Scale: (val - floor) / (1 - floor)
        # At dist=0: (1 - floor)/(1 - floor) = 1.0
        # At dist=max_dist: (floor - floor)/(1 - floor) = 0.0
        denominator = 1.0 - floor_val
        if denominator < 1e-8: # Avoid division by zero if sigma is huge
            denominator = 1.0
            
        scaled_vals = (raw_vals - floor_val) / denominator
        
        # Clip to ensure numerical stability (0.0 to 1.0)
        confidence_values[valid_mask] = np.clip(scaled_vals, 0.0, 1.0)

    if len(vertex_indices) > 1:
        confidence_values = np.max(confidence_values, axis=0)
    else:
        confidence_values = confidence_values
    
    # --- 7. Visualization ---
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    
    vertex_colors = cmap(confidence_values)[:, :3] * 255
    
    colored_mesh = mesh.copy()
    colored_mesh.visual = ColorVisuals(
        mesh=colored_mesh, 
        vertex_colors=vertex_colors.astype(np.uint8)
    )

    return torch.from_numpy(confidence_values).float().to(device), colored_mesh

def save_grasps_to_json_isaac_converted(results, json_file, output_file):
    """
    Saves grasp results to JSON, converting from Lightning (Y-up) 
    to Isaac Sim (Z-up) coordinate systems.
    """
    num_poses = results['q'].shape[0]
    output_data = []

    for i in range(num_poses):
        entry = {"id": i}

        # 1. q, q_mask (No spatial conversion needed)
        entry['q'] = results['q'][i].cpu().tolist()
        entry['q_mask'] = results['q_mask'][i].cpu().numpy().astype(int).tolist()

        # 2. Hand Global Pose
        # Apply transformations to create global pose for the robot hand
        hand_pose = np.eye(4)
        # Relative Translation Shifting in Lightning Grasp
        hand_pose[:3, 3] = -1 * results['object_pose'][i].cpu().numpy()[:3, 3]
        # Apply transformation: T_new = T_convert @ T_old
        hand_pose = lightning_to_isaac_transform(hand_pose)
         # Relative Rotation and Translation Shifting in Isaac Sim
        hand_pose[:3, :3] = np.eye(3)
        hand_pose[:3, 3] = hand_pose[:3, 3] + load_obj_T_from_json(json_file)
        entry['hand_pose'] = hand_pose.tolist()

        output_data.append(entry)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        print(f"Saved {len(output_data)} Isaac-converted poses to {output_file}")