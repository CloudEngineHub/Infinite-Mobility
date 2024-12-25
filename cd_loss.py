import trimesh
import numpy as np
from scipy.spatial import cKDTree

def normalize_mesh(mesh):
    """
    Normalize a mesh to fit within [-1, 1] in all dimensions.
    """
    # Get the bounding box of the mesh
    vertices = mesh.vertices
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    scale = (max_bound - min_bound).max() / 2.0  # Largest dimension as scale

    # Normalize vertices
    normalized_vertices = (vertices - center) / scale
    mesh.vertices = normalized_vertices
    return mesh

def chamfer_distance(points1, points2):
    """
    Compute the Chamfer Distance between two sets of points.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # For each point in points1, find the closest point in points2
    dist1, _ = tree1.query(points2)
    # For each point in points2, find the closest point in points1
    dist2, _ = tree2.query(points1)

    # Chamfer Distance is the sum of average nearest neighbor distances
    return np.mean(dist1) + np.mean(dist2)

def compute_cd_loss_with_normalization(mesh_file1, mesh_file2, num_samples=10000):
    """
    Compute the Chamfer Distance loss between two meshes after normalization.
    """
    # Load meshes
    mesh1 = trimesh.load(mesh_file1)
    mesh2 = trimesh.load(mesh_file2)

    # Normalize meshes to [-1, 1]
    mesh1 = normalize_mesh(mesh1)
    mesh2 = normalize_mesh(mesh2)

    # Sample points on the surface of each mesh
    points1 = mesh1.sample(num_samples)
    points2 = mesh2.sample(num_samples)

    # Compute Chamfer Distance
    cd = chamfer_distance(points1, points2)
    return cd

# Example usage
mesh1_path = "/home/pjlab/projects/infinigen_sep_part_urdf/outputs/ForkFactory/0/objs/0/Grid.001/Grid.001.obj"
mesh2_path = "/home/pjlab/projects/infinigen_sep_part_urdf/outputs/ForkFactory/0/objs/0/Grid.001.obj"
cd_loss = compute_cd_loss_with_normalization(mesh1_path, mesh2_path)
print(f"Chamfer Distance Loss: {cd_loss}")
