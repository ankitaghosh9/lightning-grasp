# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json 
import pickle 
from pathlib import Path 
import trimesh
from trimesh.visual import ColorVisuals
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

def pathify(p):
    if not isinstance(p, Path):
        return Path(p)
    return p

def save_json(obj, filepath, indent=4):
    """Save a Python object as a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)


def load_json(filepath):
    """Load a JSON file into a Python object."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(obj, filepath):
    """Save a Python object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load a pickle file into a Python object."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def print_dict_structure(d, indent=0):
    """
    Recursively parses a nested dictionary and prints the key structure.
    """
    for key, value in d.items():
        # Print the current key with indentation
        print('  ' * indent + str(key))
        
        # If the value is another dictionary, recurse into it
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        # Optional: Handle lists of dictionaries
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    print_dict_structure(item, indent + 1)
        else:
            print(value.shape)

import trimesh
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from trimesh.visual import ColorVisuals

def generate_geodesic_confidence_mask(mesh, contact_pos_world, object_pose, sigma=0.05, max_dist=0.05, device='cuda'):
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
    if len(vertex_indices) > 1:
        min_geodesic_dists = np.min(distances_matrix, axis=0)
    else:
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