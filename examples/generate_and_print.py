"""Demo: Generate a test STL and produce continuous DIW G-code.

Usage:
    python examples/generate_and_print.py

This creates a simple test geometry (cube, cylinder, or complex shape),
slices it, optimizes the toolpath for continuous printing, and outputs G-code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from diw_gcode import (
    PrintConfig,
    prepare_mesh,
    slice_mesh,
    generate_all_toolpaths,
    write_gcode,
)
from diw_gcode.visualizer import print_stats


def create_test_cube(size=20.0) -> trimesh.Trimesh:
    """Create a simple cube STL."""
    mesh = trimesh.creation.box(extents=[size, size, size])
    return mesh


def create_test_cylinder(radius=10.0, height=20.0) -> trimesh.Trimesh:
    """Create a cylinder STL."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    return mesh


def create_test_torus(major_r=15.0, minor_r=5.0) -> trimesh.Trimesh:
    """Create a torus (donut) - tests disconnected cross-sections."""
    # Create via revolution
    theta = np.linspace(0, 2 * np.pi, 64)
    phi = np.linspace(0, 2 * np.pi, 32)
    theta, phi = np.meshgrid(theta, phi)

    x = (major_r + minor_r * np.cos(phi)) * np.cos(theta)
    y = (major_r + minor_r * np.cos(phi)) * np.sin(theta)
    z = minor_r * np.sin(phi)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Create faces
    faces = []
    rows, cols = theta.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + (j + 1)
            v2 = (i + 1) * cols + (j + 1)
            v3 = (i + 1) * cols + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    return mesh


def create_hollow_box(outer=20.0, inner=14.0, height=15.0) -> trimesh.Trimesh:
    """Create a hollow box - tests holes in cross-sections."""
    outer_box = trimesh.creation.box(extents=[outer, outer, height])
    inner_box = trimesh.creation.box(extents=[inner, inner, height + 2])
    hollow = outer_box.difference(inner_box)
    return hollow


def main():
    print("=" * 60)
    print("DIW Continuous Print G-code Generator - Demo")
    print("=" * 60)

    # Configuration for DIW printing
    config = PrintConfig(
        layer_height=0.4,
        nozzle_diameter=0.84,
        extrusion_width=0.9,
        print_speed=10.0,
        travel_speed=30.0,
        num_perimeters=2,
        infill_density=0.4,
        infill_pattern="zigzag",
        infill_angle=45.0,
        infill_angle_increment=90.0,
        optimize_path=True,
        extrusion_multiplier=1.0,
    )

    # Generate test geometries
    geometries = {
        "cube": create_test_cube(20.0),
        "cylinder": create_test_cylinder(10.0, 20.0),
    }

    os.makedirs("stl_files", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    for name, mesh in geometries.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {name}")
        print(f"{'=' * 60}")

        # Save STL
        stl_path = f"stl_files/{name}.stl"
        mesh.export(stl_path)
        print(f"STL saved to: {stl_path}")

        # Prepare mesh
        mesh = prepare_mesh(mesh)

        # Slice
        print(f"\nSlicing...")
        layers = slice_mesh(mesh, config)

        if not layers:
            print(f"  No layers generated for {name}, skipping.")
            continue

        # Generate toolpaths
        print(f"\nGenerating optimized continuous toolpaths...")
        toolpaths = generate_all_toolpaths(layers, config)

        # Print statistics
        print_stats(toolpaths)

        # Write G-code
        gcode_path = f"output/{name}_continuous.gcode"
        write_gcode(toolpaths, config, gcode_path)

    print(f"\n{'=' * 60}")
    print("Done! G-code files are in the output/ directory.")
    print(f"{'=' * 60}")
    print("\nTo generate G-code for your own STL:")
    print("  python -m diw_gcode your_model.stl -o output.gcode")
    print("\nFor help:")
    print("  python -m diw_gcode --help")


if __name__ == "__main__":
    main()
