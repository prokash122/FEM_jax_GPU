"""Command-line interface for DIW continuous G-code generator."""

import argparse
import sys

from .config import PrintConfig
from .mesh_loader import load_mesh, prepare_mesh
from .slicer import slice_mesh
from .toolpath import generate_all_toolpaths
from .gcode_writer import write_gcode
from .visualizer import plot_layer, plot_all_layers_3d, print_stats


def main():
    parser = argparse.ArgumentParser(
        description="Smart G-code generator for DIW continuous printing. "
                    "Generates continuous toolpaths from STL files, minimizing "
                    "travel moves for direct ink writing.",
    )

    parser.add_argument("input", help="Input STL file path")
    parser.add_argument("-o", "--output", default="output.gcode",
                        help="Output G-code file path (default: output.gcode)")

    # Print parameters
    parser.add_argument("--layer-height", type=float, default=0.4,
                        help="Layer height in mm (default: 0.4)")
    parser.add_argument("--nozzle-diameter", type=float, default=0.84,
                        help="Nozzle diameter in mm (default: 0.84)")
    parser.add_argument("--extrusion-width", type=float, default=0.9,
                        help="Extrusion width in mm (default: 0.9)")
    parser.add_argument("--print-speed", type=float, default=10.0,
                        help="Print speed in mm/s (default: 10)")
    parser.add_argument("--travel-speed", type=float, default=30.0,
                        help="Travel speed in mm/s (default: 30)")

    # Extrusion
    parser.add_argument("--extrusion-multiplier", type=float, default=1.0,
                        help="Extrusion multiplier (default: 1.0)")
    parser.add_argument("--filament-diameter", type=float, default=1.75,
                        help="Filament diameter in mm (default: 1.75)")
    parser.add_argument("--pressure-mode", action="store_true",
                        help="Use pressure-based extrusion (no E values)")
    parser.add_argument("--pressure", type=float, default=0.0,
                        help="Pressure in PSI for pneumatic systems")

    # Perimeters and infill
    parser.add_argument("--num-perimeters", type=int, default=2,
                        help="Number of perimeter shells (default: 2)")
    parser.add_argument("--infill-density", type=float, default=0.5,
                        help="Infill density 0.0-1.0 (default: 0.5)")
    parser.add_argument("--infill-pattern", choices=["zigzag", "spiral", "hilbert"],
                        default="zigzag",
                        help="Infill pattern (default: zigzag)")
    parser.add_argument("--infill-angle", type=float, default=45.0,
                        help="Initial infill angle in degrees (default: 45)")

    # Optimization
    parser.add_argument("--no-optimize", action="store_true",
                        help="Disable path optimization")

    # Visualization
    parser.add_argument("--preview", action="store_true",
                        help="Show toolpath preview")
    parser.add_argument("--preview-layer", type=int, default=None,
                        help="Preview a specific layer (0-indexed)")
    parser.add_argument("--save-preview", type=str, default=None,
                        help="Save preview image to path")

    args = parser.parse_args()

    # Build config
    config = PrintConfig(
        layer_height=args.layer_height,
        nozzle_diameter=args.nozzle_diameter,
        extrusion_width=args.extrusion_width,
        print_speed=args.print_speed,
        travel_speed=args.travel_speed,
        extrusion_multiplier=args.extrusion_multiplier,
        filament_diameter=args.filament_diameter,
        use_pressure_mode=args.pressure_mode,
        pressure=args.pressure,
        num_perimeters=args.num_perimeters,
        infill_density=args.infill_density,
        infill_pattern=args.infill_pattern,
        infill_angle=args.infill_angle,
        optimize_path=not args.no_optimize,
    )

    # Pipeline
    print(f"Loading mesh: {args.input}")
    mesh = load_mesh(args.input)
    mesh = prepare_mesh(mesh)

    print(f"\nSlicing with layer height {config.layer_height}mm...")
    layers = slice_mesh(mesh, config)

    if not layers:
        print("ERROR: No layers generated. Check mesh and layer height.")
        sys.exit(1)

    print(f"\nGenerating optimized toolpaths...")
    toolpaths = generate_all_toolpaths(layers, config)

    # Statistics
    print_stats(toolpaths)

    # Write G-code
    print(f"\nWriting G-code...")
    write_gcode(toolpaths, config, args.output)

    # Visualization
    if args.preview or args.preview_layer is not None or args.save_preview:
        if args.preview_layer is not None:
            idx = args.preview_layer
            if 0 <= idx < len(toolpaths):
                plot_layer(toolpaths[idx], save_path=args.save_preview,
                          show=args.preview)
            else:
                print(f"Layer {idx} out of range (0-{len(toolpaths)-1})")
        else:
            # Show first, middle, and last layer, plus 3D
            if toolpaths:
                plot_layer(toolpaths[0], title="First Layer",
                          save_path=args.save_preview, show=args.preview)
            if len(toolpaths) > 2:
                mid = len(toolpaths) // 2
                plot_layer(toolpaths[mid], title="Middle Layer",
                          show=args.preview)
            plot_all_layers_3d(toolpaths, show=args.preview)


if __name__ == "__main__":
    main()
