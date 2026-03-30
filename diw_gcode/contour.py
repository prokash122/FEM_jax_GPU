"""Contour extraction and ordering from sliced polygons."""

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon, LinearRing

from .config import PrintConfig
from .slicer import LayerSlice


@dataclass
class Contour:
    """A single printable contour (perimeter shell)."""
    points: np.ndarray  # (N, 2) XY coordinates
    is_outer: bool = True
    is_hole: bool = False
    depth: int = 0  # 0 = outermost, increases inward


def extract_contours(layer: LayerSlice, config: PrintConfig) -> list[Contour]:
    """Extract offset perimeter contours from a layer's polygons.

    Generates num_perimeters inward-offset shells for each polygon.
    Orders inside-out (inner shells first) for better dimensional accuracy.
    """
    all_contours = []

    for polygon in layer.polygons:
        contours = _offset_contours(polygon, config.num_perimeters, config.extrusion_width)
        all_contours.extend(contours)

    return all_contours


def _offset_contours(
    polygon: Polygon,
    num_perimeters: int,
    width: float,
) -> list[Contour]:
    """Generate inward-offset perimeter shells for a polygon.

    Returns contours ordered inside-out (innermost first).
    """
    contours = []

    for i in range(num_perimeters):
        offset = width / 2.0 + i * width
        buffered = polygon.buffer(-offset)

        if buffered.is_empty:
            break

        polys = [buffered] if isinstance(buffered, Polygon) else list(buffered.geoms)

        for poly in polys:
            if poly.is_empty or poly.area < 1e-6:
                continue

            # Exterior ring
            coords = np.array(poly.exterior.coords)
            contours.append(Contour(
                points=coords,
                is_outer=(i == num_perimeters - 1),
                is_hole=False,
                depth=i,
            ))

            # Interior rings (holes)
            for interior in poly.interiors:
                hole_coords = np.array(interior.coords)
                contours.append(Contour(
                    points=hole_coords,
                    is_outer=False,
                    is_hole=True,
                    depth=i,
                ))

    # Sort inside-out: innermost (highest depth) first
    contours.sort(key=lambda c: -c.depth)

    return contours


def get_infill_region(polygon: Polygon, config: PrintConfig) -> Polygon:
    """Get the region that should be filled with infill.

    This is the polygon offset inward by all perimeters plus half an extrusion width.
    """
    total_offset = config.num_perimeters * config.extrusion_width + config.extrusion_width / 2.0
    region = polygon.buffer(-total_offset)

    if region.is_empty:
        return None

    if isinstance(region, Polygon):
        return region

    # MultiPolygon: return the union
    from shapely.ops import unary_union
    return unary_union(list(region.geoms))
