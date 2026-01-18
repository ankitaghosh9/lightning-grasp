import trimesh
import numpy as np

def check_transform(obj_path, matrix):
    # 1. Load the original mesh
    original_mesh = trimesh.load(obj_path)
    
    # 2. Create a copy and apply the transformation
    # .copy() ensures we don't modify the original object data in memory
    transformed_mesh = original_mesh.copy()
    transformed_mesh.apply_transform(matrix)
    
    # 3. Print Pose Information
    print("--- Original Pose ---")
    print(f"Centroid: {original_mesh.centroid}")
    print(f"Bounds: \n{original_mesh.bounds}")
    
    print("\n--- Transformed Pose ---")
    print(f"Centroid: {transformed_mesh.centroid}")
    print(f"Bounds: \n{transformed_mesh.bounds}")

    # 4. Visualization
    # We color the transformed mesh to distinguish it
    transformed_mesh.visual.face_colors = [255, 0, 0, 150] # Red with some transparency
    
    # Create a scene to view both simultaneously
    scene = trimesh.Scene([original_mesh, transformed_mesh])
    scene.show()

# Example Usage:
# Define a 4x4 Matrix (e.g., Translate by 5 on X-axis and Rotate 45 deg on Z)

T = np.array([[ 0.7683,  0.5738, -0.2835,  0.0218],
         [-0.3451,  0.7445,  0.5715, -0.1004],
         [ 0.5390, -0.3412,  0.7701,  0.1002],
         [ 0.0000,  0.0000,  0.0000,  1.0000]])
check_transform('assets/object/testing.obj', T)