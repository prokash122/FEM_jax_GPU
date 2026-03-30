"""Continuous infill pattern generation for DIW printing.

All patterns are designed to produce maximally continuous paths
(single connected polylines) to minimize travel moves.
"""

import math

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, linemerge

from .config import PrintConfig


def generate_infill(
    polygon: Polygon,
    config: PrintConfig,
    layer_index: int = 0,
) -> list[np.ndarray]:
    """Generate infill paths for a polygon region.

    Returns list of point arrays. Ideally a single continuous path.
    """
    if polygon is None or polygon.is_empty or polygon.area < 1e-6:
        return []

    if config.infill_density <= 0:
        return []

    # Rotate angle per layer for cross-hatching
    angle = config.infill_angle + layer_index * config.infill_angle_increment

    if config.infill_pattern == "zigzag":
        return zigzag_infill(polygon, config, angle)
    elif config.infill_pattern == "spiral":
        return spiral_infill(polygon, config)
    elif config.infill_pattern == "hilbert":
        return hilbert_infill(polygon, config)
    else:
        return zigzag_infill(polygon, config, angle)


def zigzag_infill(
    polygon: Polygon,
    config: PrintConfig,
    angle: float = 45.0,
) -> list[np.ndarray]:
    """Generate connected zigzag infill.

    Lines are connected at alternating ends to form a single continuous path.
    """
    spacing = config.extrusion_width / max(config.infill_density, 0.01)
    spacing = min(spacing, 100.0)  # sanity cap

    # Handle MultiPolygon
    if isinstance(polygon, MultiPolygon):
        results = []
        for geom in polygon.geoms:
            results.extend(zigzag_infill(geom, config, angle))
        return results

    # Rotate polygon to align infill lines with X-axis, then generate horizontal lines
    angle_rad = math.radians(angle)
    from shapely import affinity
    rotated = affinity.rotate(polygon, -angle, origin='centroid', use_radians=False)

    bounds = rotated.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds

    # Generate horizontal scan lines
    y_values = np.arange(miny + spacing / 2.0, maxy, spacing)

    if len(y_values) == 0:
        return []

    # Intersect each scan line with the polygon
    segments = []
    for y in y_values:
        line = LineString([(minx - 1, y), (maxx + 1, y)])
        intersection = rotated.intersection(line)

        if intersection.is_empty:
            continue

        if isinstance(intersection, LineString):
            coords = np.array(intersection.coords)
            if len(coords) >= 2:
                segments.append(coords)
        elif isinstance(intersection, MultiLineString):
            for seg in intersection.geoms:
                coords = np.array(seg.coords)
                if len(coords) >= 2:
                    segments.append(coords)

    if not segments:
        return []

    # Connect segments into a zigzag: alternate direction for consecutive lines
    zigzag_points = []
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            seg = seg[::-1]  # Reverse every other segment

        if zigzag_points:
            # Connect end of previous segment to start of this one
            zigzag_points.append(seg[0])

        for pt in seg:
            zigzag_points.append(pt)

    if not zigzag_points:
        return []

    zigzag = np.array(zigzag_points)

    # Rotate back to original orientation
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = polygon.centroid.x, polygon.centroid.y

    # Translate to centroid of rotated polygon, rotate, translate back
    rcx, rcy = rotated.centroid.x, rotated.centroid.y
    zigzag_shifted = zigzag - np.array([rcx, rcy])
    rotated_back = np.column_stack([
        zigzag_shifted[:, 0] * cos_a - zigzag_shifted[:, 1] * sin_a,
        zigzag_shifted[:, 0] * sin_a + zigzag_shifted[:, 1] * cos_a,
    ])
    zigzag_final = rotated_back + np.array([cx, cy])

    return [zigzag_final]


def spiral_infill(
    polygon: Polygon,
    config: PrintConfig,
) -> list[np.ndarray]:
    """Generate spiral/contour-offset infill.

    Progressively offsets the polygon boundary inward, connecting each
    ring to form a continuous spiral path.
    """
    if isinstance(polygon, MultiPolygon):
        results = []
        for geom in polygon.geoms:
            results.extend(spiral_infill(geom, config))
        return results

    spacing = config.extrusion_width
    all_points = []

    current = polygon
    offset = spacing / 2.0

    while True:
        buffered = current.buffer(-offset) if offset == spacing / 2.0 else current.buffer(-spacing)

        if buffered.is_empty:
            break

        polys = [buffered] if isinstance(buffered, Polygon) else list(buffered.geoms)

        for poly in polys:
            if poly.is_empty or poly.area < 1e-6:
                continue
            ring_coords = np.array(poly.exterior.coords)

            if all_points:
                # Connect previous ring to this one at nearest point
                last_pt = all_points[-1]
                dists = np.linalg.norm(ring_coords - last_pt, axis=1)
                nearest_idx = np.argmin(dists)
                # Reorder ring to start at nearest point
                ring_coords = np.roll(ring_coords[:-1], -nearest_idx, axis=0)
                ring_coords = np.vstack([ring_coords, ring_coords[0]])
                all_points.extend(ring_coords.tolist())
            else:
                all_points.extend(ring_coords.tolist())

        current = buffered
        offset = spacing

    if not all_points:
        return []

    return [np.array(all_points)]


def hilbert_infill(
    polygon: Polygon,
    config: PrintConfig,
) -> list[np.ndarray]:
    """Generate Hilbert curve infill.

    A space-filling curve that covers the area in a single continuous path.
    """
    if isinstance(polygon, MultiPolygon):
        results = []
        for geom in polygon.geoms:
            results.extend(hilbert_infill(geom, config))
        return results

    spacing = config.extrusion_width / max(config.infill_density, 0.01)
    bounds = polygon.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    size = max(width, height)

    # Determine Hilbert curve order
    order = max(1, int(math.ceil(math.log2(size / spacing)))) if spacing > 0 else 3
    order = min(order, 7)  # cap to avoid excessive points

    # Generate Hilbert curve points
    n = 2 ** order
    points = []
    for i in range(n * n):
        x, y = _hilbert_d2xy(n, i)
        px = bounds[0] + (x + 0.5) / n * size
        py = bounds[1] + (y + 0.5) / n * size
        points.append([px, py])

    if not points:
        return []

    curve = np.array(points)

    # Clip to polygon: keep only points inside
    from shapely.geometry import Point
    inside_mask = np.array([polygon.contains(Point(p)) for p in curve])

    if not np.any(inside_mask):
        return []

    # Extract connected segments of points inside the polygon
    result_points = curve[inside_mask]

    if len(result_points) < 2:
        return []

    return [result_points]


def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert curve index to (x, y) coordinates."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y
