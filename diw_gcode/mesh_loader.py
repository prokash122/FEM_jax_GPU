"""STL mesh loading and preprocessing."""

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an STL file and return a trimesh object."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {path} as a single mesh")
    return mesh


def prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair, orient, and position mesh for printing.

    - Fixes winding and normals
    - Fills holes if possible
    - Centers XY on origin, places Z-min at Z=0
    """
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    # Move Z-min to 0
    bounds = mesh.bounds
    translation = np.array([
        -(bounds[0][0] + bounds[1][0]) / 2.0,
        -(bounds[0][1] + bounds[1][1]) / 2.0,
        -bounds[0][2],
    ])
    mesh.apply_translation(translation)

    if not mesh.is_watertight:
        print("WARNING: Mesh is not watertight. Results may have gaps.")

    dims = mesh.bounds[1] - mesh.bounds[0]
    print(f"Mesh dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm")
    print(f"Mesh is watertight: {mesh.is_watertight}")

    return mesh
