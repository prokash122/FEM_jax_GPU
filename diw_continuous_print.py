"""
DIW Continuous Print G-code Generator - Single File Version
============================================================
Smart G-code generation for Direct Ink Writing (DIW) printers.
Generates continuous toolpaths from ANY STL geometry, minimizing
travel moves using Chinese Postman / Eulerian path optimization.

USAGE:
    pip install numpy trimesh shapely networkx scipy
    python diw_continuous_print.py your_model.stl -o output.gcode

OPTIONS:
    python diw_continuous_print.py --help
"""

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import KDTree
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point
)
from shapely.ops import polygonize, unary_union
from shapely import affinity


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class PrintConfig:
    layer_height: float = 0.4          # mm
    nozzle_diameter: float = 0.84      # mm
    extrusion_width: float = 0.9       # mm
    print_speed: float = 10.0          # mm/s
    travel_speed: float = 30.0         # mm/s
    extrusion_multiplier: float = 1.0
    filament_diameter: float = 1.75    # mm
    use_pressure_mode: bool = False
    pressure: float = 0.0             # PSI
    retraction_distance: float = 0.0
    z_hop: float = 0.0
    bed_temp: float = 0.0
    nozzle_temp: float = 0.0
    num_perimeters: int = 2
    infill_density: float = 0.5
    infill_pattern: str = "zigzag"    # "zigzag", "spiral", "hilbert"
    infill_angle: float = 45.0
    infill_angle_increment: float = 90.0
    optimize_path: bool = True
    pressure_on_code: str = "M750"
    pressure_off_code: str = "M751"


# =============================================================================
# 2. MESH LOADING
# =============================================================================

def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {path} as a single mesh")
    return mesh


def prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

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
    return mesh


# =============================================================================
# 3. SLICING
# =============================================================================

@dataclass
class LayerSlice:
    z_height: float
    polygons: list
    index: int


def slice_mesh(mesh, config):
    z_min = mesh.bounds[0][2]
    z_max = mesh.bounds[1][2]
    eps = config.layer_height * 0.001
    z_heights = np.arange(z_min + config.layer_height / 2.0 + eps, z_max, config.layer_height)

    layers = []
    for i, z in enumerate(z_heights):
        polygons = _slice_at_z(mesh, z)
        if polygons:
            layers.append(LayerSlice(z_height=float(z), polygons=polygons, index=i))

    print(f"Sliced into {len(layers)} layers")
    return layers


def _slice_at_z(mesh, z):
    try:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    except Exception:
        return []
    if section is None:
        return []

    try:
        planar, _ = section.to_2D() if hasattr(section, 'to_2D') else section.to_planar()
    except Exception:
        return []

    polygons = []
    try:
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

    if not polygons:
        try:
            vertices = planar.vertices
            entities = planar.entities
            lines = []
            for entity in entities:
                pts = vertices[entity.points]
                for j in range(len(pts) - 1):
                    lines.append((tuple(pts[j]), tuple(pts[j + 1])))
            line_strings = [LineString(l) for l in lines]
            merged = unary_union(line_strings)
            result = list(polygonize(merged))
            for poly in result:
                if poly.is_valid and poly.area > 1e-6:
                    polygons.append(poly)
        except Exception:
            pass

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


# =============================================================================
# 4. CONTOUR EXTRACTION
# =============================================================================

@dataclass
class Contour:
    points: np.ndarray
    is_outer: bool = True
    is_hole: bool = False
    depth: int = 0


def extract_contours(layer, config):
    all_contours = []
    for polygon in layer.polygons:
        all_contours.extend(_offset_contours(polygon, config.num_perimeters, config.extrusion_width))
    return all_contours


def _offset_contours(polygon, num_perimeters, width):
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
            coords = np.array(poly.exterior.coords)
            contours.append(Contour(points=coords, is_outer=(i == num_perimeters - 1),
                                     is_hole=False, depth=i))
            for interior in poly.interiors:
                contours.append(Contour(points=np.array(interior.coords),
                                         is_outer=False, is_hole=True, depth=i))

    contours.sort(key=lambda c: -c.depth)
    return contours


def get_infill_region(polygon, config):
    total_offset = config.num_perimeters * config.extrusion_width + config.extrusion_width / 2.0
    region = polygon.buffer(-total_offset)
    if region.is_empty:
        return None
    if isinstance(region, Polygon):
        return region
    return unary_union(list(region.geoms))


# =============================================================================
# 5. CONTINUOUS INFILL GENERATION
# =============================================================================

def generate_infill(polygon, config, layer_index=0):
    if polygon is None or polygon.is_empty or polygon.area < 1e-6:
        return []
    if config.infill_density <= 0:
        return []

    angle = config.infill_angle + layer_index * config.infill_angle_increment

    if config.infill_pattern == "zigzag":
        return zigzag_infill(polygon, config, angle)
    elif config.infill_pattern == "spiral":
        return spiral_infill(polygon, config)
    elif config.infill_pattern == "hilbert":
        return hilbert_infill(polygon, config)
    return zigzag_infill(polygon, config, angle)


def zigzag_infill(polygon, config, angle=45.0):
    spacing = config.extrusion_width / max(config.infill_density, 0.01)
    spacing = min(spacing, 100.0)

    if isinstance(polygon, MultiPolygon):
        results = []
        for geom in polygon.geoms:
            results.extend(zigzag_infill(geom, config, angle))
        return results

    angle_rad = math.radians(angle)
    rotated = affinity.rotate(polygon, -angle, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rotated.bounds

    y_values = np.arange(miny + spacing / 2.0, maxy, spacing)
    if len(y_values) == 0:
        return []

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

    # Connect into zigzag
    zigzag_points = []
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            seg = seg[::-1]
        if zigzag_points:
            zigzag_points.append(seg[0])
        for pt in seg:
            zigzag_points.append(pt)

    if not zigzag_points:
        return []

    zigzag = np.array(zigzag_points)

    # Rotate back
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = polygon.centroid.x, polygon.centroid.y
    rcx, rcy = rotated.centroid.x, rotated.centroid.y
    shifted = zigzag - np.array([rcx, rcy])
    rotated_back = np.column_stack([
        shifted[:, 0] * cos_a - shifted[:, 1] * sin_a,
        shifted[:, 0] * sin_a + shifted[:, 1] * cos_a,
    ])
    return [rotated_back + np.array([cx, cy])]


def spiral_infill(polygon, config):
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
                last_pt = all_points[-1]
                dists = np.linalg.norm(ring_coords - last_pt, axis=1)
                nearest_idx = np.argmin(dists)
                ring_coords = np.roll(ring_coords[:-1], -nearest_idx, axis=0)
                ring_coords = np.vstack([ring_coords, ring_coords[0]])
            all_points.extend(ring_coords.tolist())
        current = buffered
        offset = spacing

    if not all_points:
        return []
    return [np.array(all_points)]


def hilbert_infill(polygon, config):
    if isinstance(polygon, MultiPolygon):
        results = []
        for geom in polygon.geoms:
            results.extend(hilbert_infill(geom, config))
        return results

    spacing = config.extrusion_width / max(config.infill_density, 0.01)
    bounds = polygon.bounds
    size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    order = max(1, int(math.ceil(math.log2(size / spacing)))) if spacing > 0 else 3
    order = min(order, 7)

    n = 2 ** order
    points = []
    for i in range(n * n):
        x, y = _hilbert_d2xy(n, i)
        points.append([bounds[0] + (x + 0.5) / n * size, bounds[1] + (y + 0.5) / n * size])

    curve = np.array(points)
    inside_mask = np.array([polygon.contains(Point(p)) for p in curve])
    if not np.any(inside_mask):
        return []
    result = curve[inside_mask]
    return [result] if len(result) >= 2 else []


def _hilbert_d2xy(n, d):
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


# =============================================================================
# 6. GRAPH OPTIMIZER (Chinese Postman / Euler Path)
# =============================================================================

def optimize_toolpath(path_segments, config):
    if not path_segments or len(path_segments) == 1 or not config.optimize_path:
        return path_segments

    graph, segment_edges = _build_graph(path_segments)
    components = list(nx.connected_components(graph))

    if len(components) == 1:
        return _solve_single_component(graph, segment_edges, path_segments)
    else:
        return _solve_multi_component(graph, components, segment_edges, path_segments)


def _build_graph(path_segments):
    graph = nx.Graph()
    segment_edges = {}
    pos_to_node = {}
    next_id = [0]

    def get_node(pos):
        key = (round(pos[0], 4), round(pos[1], 4))
        if key in pos_to_node:
            return pos_to_node[key]
        nid = next_id[0]
        next_id[0] += 1
        pos_to_node[key] = nid
        graph.add_node(nid, pos=np.array(pos))
        return nid

    for seg_idx, seg in enumerate(path_segments):
        if len(seg) < 2:
            continue
        s = get_node(seg[0])
        e = get_node(seg[-1])
        graph.add_edge(s, e, segment_idx=seg_idx, weight=0.0)
        segment_edges[(s, e)] = seg_idx
        segment_edges[(e, s)] = seg_idx

    return graph, segment_edges


def _solve_single_component(graph, segment_edges, path_segments):
    if graph.number_of_edges() == 0:
        return path_segments

    odd_nodes = [n for n in graph.nodes() if graph.degree(n) % 2 == 1]

    if len(odd_nodes) == 0:
        return _extract_euler_circuit(graph, segment_edges, path_segments)
    if len(odd_nodes) == 2:
        return _extract_euler_path(graph, segment_edges, path_segments, odd_nodes)

    _add_matching_edges(graph, odd_nodes)
    return _extract_euler_circuit(graph, segment_edges, path_segments)


def _add_matching_edges(graph, odd_nodes):
    if len(odd_nodes) < 2:
        return
    positions = {n: graph.nodes[n].get('pos', np.zeros(2)) for n in odd_nodes}
    odd_graph = nx.Graph()
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1:]:
            dist = np.linalg.norm(positions[u] - positions[v])
            odd_graph.add_edge(u, v, weight=-dist)
    matching = nx.max_weight_matching(odd_graph, maxcardinality=True)
    for u, v in matching:
        dist = np.linalg.norm(positions.get(u, np.zeros(2)) - positions.get(v, np.zeros(2)))
        graph.add_edge(u, v, weight=dist, is_travel=True)


def _extract_euler_circuit(graph, segment_edges, path_segments):
    try:
        circuit = list(nx.eulerian_circuit(graph))
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return _greedy_order(path_segments)
    return _circuit_to_segments(circuit, segment_edges, path_segments, graph)


def _extract_euler_path(graph, segment_edges, path_segments, odd_nodes):
    try:
        path = list(nx.eulerian_path(graph, source=odd_nodes[0]))
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return _greedy_order(path_segments)
    return _circuit_to_segments(path, segment_edges, path_segments, graph)


def _circuit_to_segments(circuit, segment_edges, path_segments, graph):
    ordered = []
    used = set()
    for u, v in circuit:
        data = graph.edges[u, v]
        if data.get('is_travel', False):
            ordered.append(np.array([graph.nodes[u]['pos'], graph.nodes[v]['pos']]))
            continue
        idx = data.get('segment_idx')
        if idx is not None and idx not in used:
            seg = path_segments[idx]
            pos_u = graph.nodes[u].get('pos', np.zeros(2))
            if np.linalg.norm(seg[0] - pos_u) > np.linalg.norm(seg[-1] - pos_u):
                seg = seg[::-1]
            ordered.append(seg)
            used.add(idx)
    for i, seg in enumerate(path_segments):
        if i not in used:
            ordered.append(seg)
    return ordered


def _solve_multi_component(graph, components, segment_edges, path_segments):
    component_segments = []
    for comp_nodes in components:
        subgraph = graph.subgraph(comp_nodes)
        indices = {d.get('segment_idx') for _, _, d in subgraph.edges(data=True) if d.get('segment_idx') is not None}
        if indices:
            component_segments.append([path_segments[i] for i in sorted(indices)])
    if not component_segments:
        return path_segments
    return _tsp_order(component_segments)


def _tsp_order(component_segments):
    if len(component_segments) <= 1:
        return [seg for segs in component_segments for seg in segs]
    reps = np.array([segs[0][0] for segs in component_segments])
    n = len(reps)
    visited = [False] * n
    order = []
    current = int(np.argmin(np.linalg.norm(reps, axis=1)))
    visited[current] = True
    order.append(current)
    for _ in range(n - 1):
        dists = [np.linalg.norm(reps[current] - reps[j]) if not visited[j] else float('inf') for j in range(n)]
        nxt = int(np.argmin(dists))
        visited[nxt] = True
        order.append(nxt)
        current = nxt
    result = []
    for i in order:
        result.extend(component_segments[i])
    return result


def _greedy_order(path_segments):
    if len(path_segments) <= 1:
        return path_segments
    n = len(path_segments)
    used = [False] * n
    dists = np.array([np.linalg.norm(s[0]) for s in path_segments])
    current_idx = int(np.argmin(dists))
    used[current_idx] = True
    ordered = [path_segments[current_idx]]
    current_pos = path_segments[current_idx][-1]

    for _ in range(n - 1):
        best_d, best_j, best_rev = float('inf'), -1, False
        for j in range(n):
            if used[j]:
                continue
            d0 = np.linalg.norm(current_pos - path_segments[j][0])
            d1 = np.linalg.norm(current_pos - path_segments[j][-1])
            if d0 < best_d:
                best_d, best_j, best_rev = d0, j, False
            if d1 < best_d:
                best_d, best_j, best_rev = d1, j, True
        if best_j >= 0:
            used[best_j] = True
            seg = path_segments[best_j][::-1] if best_rev else path_segments[best_j]
            ordered.append(seg)
            current_pos = seg[-1]
    return ordered


# =============================================================================
# 7. TOOLPATH ASSEMBLY
# =============================================================================

@dataclass
class ToolpathSegment:
    points: np.ndarray
    segment_type: str  # "perimeter", "infill", "travel"


@dataclass
class LayerToolpath:
    z_height: float
    segments: list
    travel_distance: float = 0.0
    print_distance: float = 0.0


def generate_layer_toolpath(layer, config):
    contours = extract_contours(layer, config)

    infill_paths = []
    for polygon in layer.polygons:
        region = get_infill_region(polygon, config)
        if region is not None:
            infill_paths.extend(generate_infill(region, config, layer.index))

    all_segments = []
    for c in contours:
        if len(c.points) >= 2:
            all_segments.append(c.points)
    for ip in infill_paths:
        if len(ip) >= 2:
            all_segments.append(ip)

    if not all_segments:
        return LayerToolpath(z_height=layer.z_height, segments=[])

    optimized = optimize_toolpath(all_segments, config) if len(all_segments) > 1 else all_segments

    toolpath_segments = []
    total_travel = total_print = 0.0
    prev_end = None

    for seg in optimized:
        if len(seg) < 2:
            continue
        if prev_end is not None:
            gap = np.linalg.norm(seg[0] - prev_end)
            if gap > 0.01:
                toolpath_segments.append(ToolpathSegment(np.array([prev_end, seg[0]]), "travel"))
                total_travel += gap

        toolpath_segments.append(ToolpathSegment(seg, "infill"))
        total_print += float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        prev_end = seg[-1].copy()

    return LayerToolpath(layer.z_height, toolpath_segments, total_travel, total_print)


def generate_all_toolpaths(layers, config):
    toolpaths = []
    total_travel = total_print = 0.0
    for i, layer in enumerate(layers):
        tp = generate_layer_toolpath(layer, config)
        toolpaths.append(tp)
        total_travel += tp.travel_distance
        total_print += tp.print_distance
        if (i + 1) % 10 == 0 or i == len(layers) - 1:
            print(f"  Layer {i+1}/{len(layers)}: travel={tp.travel_distance:.1f}mm, print={tp.print_distance:.1f}mm")

    print(f"\nTotal print distance: {total_print:.1f} mm")
    print(f"Total travel distance: {total_travel:.1f} mm")
    if total_print > 0:
        print(f"Travel ratio: {total_travel / total_print * 100:.1f}%")
    return toolpaths


# =============================================================================
# 8. G-CODE WRITER
# =============================================================================

def write_gcode(toolpaths, config, output_path):
    lines = []

    # Header
    lines.extend([
        f"; DIW Continuous Print G-code",
        f"; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"; Layer height: {config.layer_height} mm",
        f"; Nozzle diameter: {config.nozzle_diameter} mm",
        f"; Print speed: {config.print_speed} mm/s",
        f"; Infill: {config.infill_pattern} @ {config.infill_density*100:.0f}%",
        f"; Layers: {len(toolpaths)}",
        f"; Optimized: {config.optimize_path}",
        ";",
        "; --- Machine Setup ---",
        "G21 ; Metric units",
        "G90 ; Absolute positioning",
    ])

    if config.use_pressure_mode:
        lines.append("M83 ; Relative extrusion")
    else:
        lines.append("M82 ; Absolute extrusion")

    lines.append("G28 ; Home all axes")

    if config.bed_temp > 0:
        lines.append(f"M190 S{config.bed_temp:.0f} ; Wait for bed")
    if config.nozzle_temp > 0:
        lines.append(f"M109 S{config.nozzle_temp:.0f} ; Wait for nozzle")
    if config.use_pressure_mode and config.pressure > 0:
        lines.append(f"{config.pressure_on_code} P{config.pressure:.1f} ; Set pressure")

    lines.extend(["G92 E0 ; Reset extruder", f"G0 F{config.travel_speed*60:.0f}", ""])

    # Layers
    e_total = 0.0
    for tp in toolpaths:
        lines.append(f"; --- Layer Z={tp.z_height:.2f} mm ---")
        lines.append(f"G0 Z{tp.z_height:.3f} F{config.travel_speed*60:.0f}")

        for seg in tp.segments:
            pts = seg.points
            if len(pts) < 2:
                continue

            if seg.segment_type == "travel":
                x, y = pts[-1]
                if config.z_hop > 0:
                    lines.append(f"G0 Z{tp.z_height + config.z_hop:.3f} F{config.travel_speed*60:.0f}")
                lines.append(f"G0 X{x:.3f} Y{y:.3f} F{config.travel_speed*60:.0f}")
                if config.z_hop > 0:
                    lines.append(f"G0 Z{tp.z_height:.3f} F{config.travel_speed*60:.0f}")
            else:
                x0, y0 = pts[0]
                lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{config.travel_speed*60:.0f}")
                speed_mm = config.print_speed * 60.0

                for i in range(1, len(pts)):
                    x, y = pts[i]
                    dx = pts[i][0] - pts[i-1][0]
                    dy = pts[i][1] - pts[i-1][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.001:
                        continue

                    if config.use_pressure_mode:
                        lines.append(f"G1 X{x:.3f} Y{y:.3f} F{speed_mm:.0f}")
                    else:
                        cross = config.layer_height * config.extrusion_width
                        fil_area = math.pi * (config.filament_diameter / 2.0) ** 2
                        e_total += dist * cross / fil_area * config.extrusion_multiplier
                        lines.append(f"G1 X{x:.3f} Y{y:.3f} E{e_total:.5f} F{speed_mm:.0f}")
        lines.append("")

    # Footer
    if config.use_pressure_mode and config.pressure > 0:
        lines.append(f"{config.pressure_off_code} ; Pressure off")
    lines.extend([
        "; --- End ---",
        "G91 ; Relative positioning",
        "G0 Z10 F600 ; Lift nozzle",
        "G90 ; Absolute positioning",
        "G0 X0 Y0 F3000 ; Home XY",
        "M84 ; Disable steppers",
        "; End of G-code",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    total_travel = sum(tp.travel_distance for tp in toolpaths)
    total_print = sum(tp.print_distance for tp in toolpaths)
    print(f"\nG-code written to: {output_path}")
    print(f"Lines: {len(lines)}, Layers: {len(toolpaths)}")
    print(f"Print: {total_print:.1f}mm, Travel: {total_travel:.1f}mm")


# =============================================================================
# 9. CLI
# =============================================================================

def _create_demo_mesh(shape: str = "cube") -> trimesh.Trimesh:
    """Create a built-in demo geometry for testing."""
    if shape == "cube":
        return trimesh.creation.box(extents=[20, 20, 20])
    elif shape == "cylinder":
        return trimesh.creation.cylinder(radius=10, height=20, sections=64)
    elif shape == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=10)
    elif shape == "cone":
        return trimesh.creation.cone(radius=10, height=20, sections=64)
    else:
        return trimesh.creation.box(extents=[20, 20, 20])


def main():
    parser = argparse.ArgumentParser(
        description="DIW Continuous Print G-code Generator\n"
                    "Generates continuous toolpaths from STL files for Direct Ink Writing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python diw_continuous_print.py model.stl -o output.gcode\n"
               "  python diw_continuous_print.py model.stl --pressure-mode --pressure 25\n"
               "  python diw_continuous_print.py model.stl --infill-pattern spiral --infill-density 0.6\n",
    )

    parser.add_argument("input", nargs="?", default=None,
                        help="Input STL file (omit to use built-in demo geometry)")
    parser.add_argument("-o", "--output", default="output.gcode", help="Output G-code file (default: output.gcode)")
    parser.add_argument("--demo", choices=["cube", "cylinder", "sphere", "cone"],
                        default=None,
                        help="Use a built-in demo geometry instead of an STL file")

    g = parser.add_argument_group("Geometry")
    g.add_argument("--layer-height", type=float, default=0.4, help="Layer height mm (default: 0.4)")
    g.add_argument("--nozzle-diameter", type=float, default=0.84, help="Nozzle diameter mm (default: 0.84)")
    g.add_argument("--extrusion-width", type=float, default=0.9, help="Extrusion width mm (default: 0.9)")

    g = parser.add_argument_group("Speed")
    g.add_argument("--print-speed", type=float, default=10.0, help="Print speed mm/s (default: 10)")
    g.add_argument("--travel-speed", type=float, default=30.0, help="Travel speed mm/s (default: 30)")

    g = parser.add_argument_group("Extrusion")
    g.add_argument("--extrusion-multiplier", type=float, default=1.0, help="Flow multiplier (default: 1.0)")
    g.add_argument("--filament-diameter", type=float, default=1.75, help="Filament diameter mm (default: 1.75)")
    g.add_argument("--pressure-mode", action="store_true", help="Pressure-based extrusion (no E values)")
    g.add_argument("--pressure", type=float, default=0.0, help="Pressure PSI for pneumatic systems")

    g = parser.add_argument_group("Toolpath")
    g.add_argument("--num-perimeters", type=int, default=2, help="Perimeter shells (default: 2)")
    g.add_argument("--infill-density", type=float, default=0.5, help="Infill density 0-1 (default: 0.5)")
    g.add_argument("--infill-pattern", choices=["zigzag", "spiral", "hilbert"], default="zigzag")
    g.add_argument("--infill-angle", type=float, default=45.0, help="Initial infill angle deg (default: 45)")
    g.add_argument("--no-optimize", action="store_true", help="Disable path optimization")

    args = parser.parse_args()

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

    print("=" * 50)
    print("DIW Continuous Print G-code Generator")
    print("=" * 50)

    # Load or generate mesh
    if args.input:
        print(f"\nLoading: {args.input}")
        mesh = load_mesh(args.input)
    elif args.demo:
        print(f"\nGenerating demo geometry: {args.demo}")
        mesh = _create_demo_mesh(args.demo)
    else:
        print("\nNo STL file provided. Using demo cube (20x20x20 mm).")
        print("Tip: python diw_continuous_print.py your_model.stl -o output.gcode")
        print("     python diw_continuous_print.py --demo cylinder -o output.gcode\n")
        mesh = _create_demo_mesh("cube")

    mesh = prepare_mesh(mesh)

    print(f"\nSlicing (layer height={config.layer_height}mm)...")
    layers = slice_mesh(mesh, config)
    if not layers:
        print("ERROR: No layers. Check mesh and layer height.")
        sys.exit(1)

    print(f"\nOptimizing toolpaths for continuous printing...")
    toolpaths = generate_all_toolpaths(layers, config)

    print(f"\nWriting G-code...")
    write_gcode(toolpaths, config, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
