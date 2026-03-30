"""Toolpath assembly: combines contours and infill into optimized per-layer paths."""

from dataclasses import dataclass

import numpy as np

from .config import PrintConfig
from .slicer import LayerSlice
from .contour import extract_contours, get_infill_region, Contour
from .infill import generate_infill
from .graph_optimizer import optimize_toolpath


@dataclass
class ToolpathSegment:
    """A single segment of the toolpath."""
    points: np.ndarray  # (N, 2) XY coordinates
    segment_type: str  # "perimeter", "infill", "travel"


@dataclass
class LayerToolpath:
    """Complete toolpath for a single layer."""
    z_height: float
    segments: list[ToolpathSegment]
    travel_distance: float = 0.0
    print_distance: float = 0.0


def generate_layer_toolpath(
    layer: LayerSlice,
    config: PrintConfig,
) -> LayerToolpath:
    """Generate optimized toolpath for a single layer.

    1. Extract perimeter contours
    2. Generate infill for interior region
    3. Combine all path segments
    4. Optimize ordering for continuity (Chinese Postman / greedy)
    5. Identify travel moves between disconnected segments
    """
    # Step 1: Extract contours
    contours = extract_contours(layer, config)

    # Step 2: Generate infill
    infill_paths = []
    for polygon in layer.polygons:
        infill_region = get_infill_region(polygon, config)
        if infill_region is not None:
            paths = generate_infill(infill_region, config, layer.index)
            infill_paths.extend(paths)

    # Step 3: Collect all printable segments
    all_segments = []
    segment_types = []

    for contour in contours:
        if len(contour.points) >= 2:
            all_segments.append(contour.points)
            segment_types.append("perimeter")

    for infill_path in infill_paths:
        if len(infill_path) >= 2:
            all_segments.append(infill_path)
            segment_types.append("infill")

    if not all_segments:
        return LayerToolpath(z_height=layer.z_height, segments=[])

    # Step 4: Optimize segment ordering
    if config.optimize_path and len(all_segments) > 1:
        optimized = optimize_toolpath(all_segments, config)
    else:
        optimized = all_segments

    # Step 5: Build final toolpath with travel identification
    toolpath_segments = []
    total_travel = 0.0
    total_print = 0.0
    prev_end = None

    for seg in optimized:
        if len(seg) < 2:
            continue

        # Check if we need a travel move to reach this segment
        if prev_end is not None:
            gap = np.linalg.norm(seg[0] - prev_end)
            if gap > 0.01:  # More than 0.01mm gap = travel move
                travel_seg = np.array([prev_end, seg[0]])
                toolpath_segments.append(ToolpathSegment(
                    points=travel_seg,
                    segment_type="travel",
                ))
                total_travel += gap

        # Determine segment type (try to preserve original type)
        seg_type = _classify_segment(seg, all_segments, segment_types)
        toolpath_segments.append(ToolpathSegment(
            points=seg,
            segment_type=seg_type,
        ))

        # Calculate print distance
        diffs = np.diff(seg, axis=0)
        total_print += float(np.sum(np.linalg.norm(diffs, axis=1)))

        prev_end = seg[-1].copy()

    return LayerToolpath(
        z_height=layer.z_height,
        segments=toolpath_segments,
        travel_distance=total_travel,
        print_distance=total_print,
    )


def _classify_segment(
    segment: np.ndarray,
    original_segments: list[np.ndarray],
    segment_types: list[str],
) -> str:
    """Try to identify if a segment is perimeter or infill."""
    # Check if this segment matches any original segment
    for orig, stype in zip(original_segments, segment_types):
        if len(segment) == len(orig):
            if np.allclose(segment, orig, atol=0.01) or np.allclose(segment, orig[::-1], atol=0.01):
                return stype
    return "infill"


def generate_all_toolpaths(
    layers: list[LayerSlice],
    config: PrintConfig,
) -> list[LayerToolpath]:
    """Generate optimized toolpaths for all layers."""
    toolpaths = []
    total_travel = 0.0
    total_print = 0.0

    for i, layer in enumerate(layers):
        tp = generate_layer_toolpath(layer, config)
        toolpaths.append(tp)
        total_travel += tp.travel_distance
        total_print += tp.print_distance

        if (i + 1) % 10 == 0 or i == len(layers) - 1:
            print(f"  Layer {i + 1}/{len(layers)}: "
                  f"travel={tp.travel_distance:.1f}mm, print={tp.print_distance:.1f}mm")

    print(f"\nTotal print distance: {total_print:.1f} mm")
    print(f"Total travel distance: {total_travel:.1f} mm")
    if total_print > 0:
        print(f"Travel ratio: {total_travel / total_print * 100:.1f}%")

    return toolpaths
