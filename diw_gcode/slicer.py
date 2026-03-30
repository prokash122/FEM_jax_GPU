"""Mesh slicing into 2D cross-sections."""

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union

from .config import PrintConfig


@dataclass
class LayerSlice:
    """A single 2D layer extracted from the mesh."""
    z_height: float
    polygons: list  # list of shapely.Polygon
    index: int


def slice_mesh(mesh: trimesh.Trimesh, config: PrintConfig) -> list[LayerSlice]:
    """Slice a mesh into horizontal layers.

    Returns a list of LayerSlice objects, each containing the 2D polygons
    at that Z height.
    """
    z_min = mesh.bounds[0][2]
    z_max = mesh.bounds[1][2]

    # Offset slightly from exact boundaries to avoid degenerate cuts
    eps = config.layer_height * 0.001
    z_heights = np.arange(
        z_min + config.layer_height / 2.0 + eps,
        z_max,
        config.layer_height,
    )

    layers = []
    for i, z in enumerate(z_heights):
        polygons = _slice_at_z(mesh, z)
        if polygons:
            layers.append(LayerSlice(z_height=float(z), polygons=polygons, index=i))

    print(f"Sliced into {len(layers)} layers")
    return layers


def _slice_at_z(mesh: trimesh.Trimesh, z: float) -> list[Polygon]:
    """Cut the mesh at a given Z height and return Shapely polygons."""
    try:
        section = mesh.section(
            plane_origin=[0, 0, z],
            plane_normal=[0, 0, 1],
        )
    except Exception:
        return []

    if section is None:
        return []

    try:
        planar, _transform = section.to_2D() if hasattr(section, 'to_2D') else section.to_planar()
    except Exception:
        return []

    # Extract polygons from the Path2D
    polygons = []
    try:
        # Try trimesh's built-in polygon extraction
        if hasattr(planar, 'polygons_full') and planar.polygons_full:
            for poly in planar.polygons_full:
                if poly.is_valid and poly.area > 1e-6:
                    polygons.append(poly)
        elif hasattr(planar, 'polygons_closed') and len(planar.polygons_closed) > 0:
            for poly in planar.polygons_closed:
                if poly.is_valid and poly.area > 1e-6:
                    polygons.append(poly)
    except Exception:
        pass

    # Fallback: build polygons from discrete line segments
    if not polygons:
        try:
            vertices = planar.vertices
            entities = planar.entities
            lines = []
            for entity in entities:
                pts = vertices[entity.points]
                for j in range(len(pts) - 1):
                    lines.append((tuple(pts[j]), tuple(pts[j + 1])))
            from shapely.geometry import LineString
            line_strings = [LineString(l) for l in lines]
            merged = unary_union(line_strings)
            result = list(polygonize(merged))
            for poly in result:
                if poly.is_valid and poly.area > 1e-6:
                    polygons.append(poly)
        except Exception:
            pass

    # Clean up: union overlapping polygons
    if len(polygons) > 1:
        try:
            merged = unary_union(polygons)
            if isinstance(merged, Polygon):
                polygons = [merged]
            elif isinstance(merged, MultiPolygon):
                polygons = list(merged.geoms)
        except Exception:
            pass

    return polygons
