"""G-code generation for DIW (Direct Ink Writing) printers."""

import math
from datetime import datetime

import numpy as np

from .config import PrintConfig
from .toolpath import LayerToolpath, ToolpathSegment


def write_gcode(
    toolpaths: list[LayerToolpath],
    config: PrintConfig,
    output_path: str,
) -> None:
    """Write complete G-code file from toolpaths."""
    lines = []

    # Header
    lines.extend(_generate_header(config, toolpaths))

    # Body: layer by layer
    e_total = 0.0
    for tp in toolpaths:
        layer_lines, e_total = _generate_layer(tp, config, e_total)
        lines.extend(layer_lines)

    # Footer
    lines.extend(_generate_footer(config))

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')

    total_travel = sum(tp.travel_distance for tp in toolpaths)
    total_print = sum(tp.print_distance for tp in toolpaths)
    print(f"\nG-code written to: {output_path}")
    print(f"Total lines: {len(lines)}")
    print(f"Total print distance: {total_print:.1f} mm")
    print(f"Total travel distance: {total_travel:.1f} mm")
    print(f"Number of layers: {len(toolpaths)}")


def _generate_header(config: PrintConfig, toolpaths: list[LayerToolpath]) -> list[str]:
    """Generate G-code header with machine setup."""
    lines = [
        f"; DIW Continuous Print G-code",
        f"; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"; Layer height: {config.layer_height} mm",
        f"; Nozzle diameter: {config.nozzle_diameter} mm",
        f"; Print speed: {config.print_speed} mm/s",
        f"; Infill pattern: {config.infill_pattern}",
        f"; Infill density: {config.infill_density * 100:.0f}%",
        f"; Layers: {len(toolpaths)}",
        f"; Optimized path: {config.optimize_path}",
        ";",
        "; --- Machine Setup ---",
        "G21 ; Metric units",
        "G90 ; Absolute positioning",
    ]

    if config.use_pressure_mode:
        lines.append("M83 ; Relative extrusion (pressure mode)")
    else:
        lines.append("M82 ; Absolute extrusion")

    lines.append("G28 ; Home all axes")

    if config.bed_temp > 0:
        lines.append(f"M190 S{config.bed_temp:.0f} ; Wait for bed temp")

    if config.nozzle_temp > 0:
        lines.append(f"M109 S{config.nozzle_temp:.0f} ; Wait for nozzle temp")

    if config.use_pressure_mode and config.pressure > 0:
        lines.append(f"{config.pressure_on_code} P{config.pressure:.1f} ; Set pressure")

    lines.extend([
        "G92 E0 ; Reset extruder",
        f"G0 F{config.travel_speed * 60:.0f} ; Set travel feedrate",
        "",
    ])

    return lines


def _generate_layer(
    toolpath: LayerToolpath,
    config: PrintConfig,
    e_total: float,
) -> tuple[list[str], float]:
    """Generate G-code for a single layer."""
    lines = [
        f"; --- Layer {toolpath.z_height:.2f} mm ---",
        f"G0 Z{toolpath.z_height:.3f} F{config.travel_speed * 60:.0f}",
    ]

    for segment in toolpath.segments:
        seg_lines, e_total = _generate_segment(segment, config, e_total, toolpath.z_height)
        lines.extend(seg_lines)

    lines.append("")
    return lines, e_total


def _generate_segment(
    segment: ToolpathSegment,
    config: PrintConfig,
    e_total: float,
    z_height: float,
) -> tuple[list[str], float]:
    """Generate G-code for a single toolpath segment."""
    lines = []
    points = segment.points

    if len(points) < 2:
        return lines, e_total

    if segment.segment_type == "travel":
        # Travel move (no extrusion)
        x, y = points[-1]
        if config.z_hop > 0:
            lines.append(f"G0 Z{z_height + config.z_hop:.3f} F{config.travel_speed * 60:.0f}")
        lines.append(f"G0 X{x:.3f} Y{y:.3f} F{config.travel_speed * 60:.0f}")
        if config.z_hop > 0:
            lines.append(f"G0 Z{z_height:.3f} F{config.travel_speed * 60:.0f}")
        return lines, e_total

    # Print move
    speed_mmmin = config.print_speed * 60.0

    # Move to start point without extrusion
    x0, y0 = points[0]
    lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{config.travel_speed * 60:.0f}")

    for i in range(1, len(points)):
        x, y = points[i]
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            continue

        if config.use_pressure_mode:
            lines.append(f"G1 X{x:.3f} Y{y:.3f} F{speed_mmmin:.0f}")
        else:
            # Calculate E value: cross-section area * distance / filament area
            cross_section = config.layer_height * config.extrusion_width
            filament_area = math.pi * (config.filament_diameter / 2.0) ** 2
            e_per_mm = cross_section / filament_area * config.extrusion_multiplier
            e_total += dist * e_per_mm
            lines.append(f"G1 X{x:.3f} Y{y:.3f} E{e_total:.5f} F{speed_mmmin:.0f}")

    return lines, e_total


def _generate_footer(config: PrintConfig) -> list[str]:
    """Generate G-code footer."""
    lines = [
        "; --- End ---",
    ]

    if config.use_pressure_mode and config.pressure > 0:
        lines.append(f"{config.pressure_off_code} ; Turn off pressure")

    lines.extend([
        "G91 ; Relative positioning",
        "G0 Z10 F600 ; Lift nozzle",
        "G90 ; Absolute positioning",
        "G0 X0 Y0 F3000 ; Move to home",
        "M84 ; Disable steppers",
        "; End of G-code",
    ])

    return lines
