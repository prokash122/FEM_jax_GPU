"""Chinese Postman / Eulerian path solver for continuous toolpath optimization.

This is the core algorithm that ensures the toolpath is as continuous as possible.
It models contours and infill lines as edges in a graph, then finds an Eulerian
circuit (or near-Eulerian path) that traverses all edges with minimum travel.
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .config import PrintConfig


def optimize_toolpath(
    path_segments: list[np.ndarray],
    config: PrintConfig,
) -> list[np.ndarray]:
    """Optimize a set of path segments into a continuous toolpath.

    Takes a list of polyline segments (each an Nx2 array) and returns
    an ordered list of segments that minimizes travel moves between them.

    For a single connected component, uses Chinese Postman (Euler circuit).
    For multiple disconnected components, uses nearest-neighbor TSP to
    connect them with minimal travel.
    """
    if not path_segments:
        return []

    if len(path_segments) == 1:
        return path_segments

    if not config.optimize_path:
        return path_segments

    # Build a graph from all path segments
    graph, segment_edges = _build_graph(path_segments)

    # Find connected components
    components = list(nx.connected_components(graph))

    if len(components) == 1:
        # Single connected component: solve Chinese Postman
        ordered = _solve_single_component(graph, segment_edges, path_segments)
        return ordered
    else:
        # Multiple components: optimize within each, then connect via TSP
        return _solve_multi_component(graph, components, segment_edges, path_segments)


def _build_graph(path_segments: list[np.ndarray]) -> tuple[nx.Graph, dict]:
    """Build a graph where path segments are edges.

    Nodes are unique XY positions. Edges represent segments that must be printed.
    Returns the graph and a mapping from edge keys to segment indices.
    """
    graph = nx.Graph()
    segment_edges = {}
    node_positions = {}
    pos_to_node = {}
    next_node_id = 0

    def get_node(pos):
        nonlocal next_node_id
        key = (round(pos[0], 4), round(pos[1], 4))
        if key in pos_to_node:
            return pos_to_node[key]
        node_id = next_node_id
        next_node_id += 1
        pos_to_node[key] = node_id
        node_positions[node_id] = np.array(pos)
        graph.add_node(node_id, pos=np.array(pos))
        return node_id

    for seg_idx, segment in enumerate(path_segments):
        if len(segment) < 2:
            continue

        start_node = get_node(segment[0])
        end_node = get_node(segment[-1])

        # Add edge representing this segment
        graph.add_edge(start_node, end_node, segment_idx=seg_idx, weight=0.0)
        segment_edges[(start_node, end_node)] = seg_idx
        segment_edges[(end_node, start_node)] = seg_idx

    return graph, segment_edges


def _solve_single_component(
    graph: nx.Graph,
    segment_edges: dict,
    path_segments: list[np.ndarray],
) -> list[np.ndarray]:
    """Solve the Chinese Postman Problem for a single connected component.

    1. Find nodes with odd degree
    2. Add minimum-weight duplicate edges to make all degrees even
    3. Find Eulerian circuit
    4. Convert circuit back to ordered segments
    """
    if graph.number_of_edges() == 0:
        return path_segments

    # Check for Eulerian circuit conditions
    odd_nodes = [n for n in graph.nodes() if graph.degree(n) % 2 == 1]

    if len(odd_nodes) == 0:
        # Already Eulerian - find circuit
        return _extract_euler_circuit(graph, segment_edges, path_segments)

    if len(odd_nodes) == 2:
        # Eulerian path exists between the two odd-degree nodes
        return _extract_euler_path(graph, segment_edges, path_segments, odd_nodes)

    # Need to add edges to make it Eulerian (Chinese Postman)
    _add_matching_edges(graph, odd_nodes)

    return _extract_euler_circuit(graph, segment_edges, path_segments)


def _add_matching_edges(graph: nx.Graph, odd_nodes: list[int]):
    """Add minimum-weight edges between odd-degree nodes to make the graph Eulerian.

    Uses shortest paths between all pairs of odd-degree nodes, then finds
    minimum weight perfect matching.
    """
    if len(odd_nodes) < 2:
        return

    # Build complete graph on odd-degree nodes with shortest path weights
    odd_graph = nx.Graph()
    positions = {n: graph.nodes[n].get('pos', np.zeros(2)) for n in odd_nodes}

    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1:]:
            pos_u = positions[u]
            pos_v = positions[v]
            dist = np.linalg.norm(pos_u - pos_v)
            odd_graph.add_edge(u, v, weight=-dist)  # negative for max matching

    # Minimum weight perfect matching (using max_weight_matching with negative weights)
    matching = nx.max_weight_matching(odd_graph, maxcardinality=True)

    # Add travel edges for each matched pair
    for u, v in matching:
        pos_u = positions.get(u, np.zeros(2))
        pos_v = positions.get(v, np.zeros(2))
        dist = np.linalg.norm(pos_u - pos_v)
        graph.add_edge(u, v, weight=dist, is_travel=True)


def _extract_euler_circuit(
    graph: nx.Graph,
    segment_edges: dict,
    path_segments: list[np.ndarray],
) -> list[np.ndarray]:
    """Extract Eulerian circuit and convert to ordered segments."""
    try:
        circuit = list(nx.eulerian_circuit(graph))
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        # Fallback: greedy ordering
        return _greedy_order(path_segments)

    return _circuit_to_segments(circuit, segment_edges, path_segments, graph)


def _extract_euler_path(
    graph: nx.Graph,
    segment_edges: dict,
    path_segments: list[np.ndarray],
    odd_nodes: list[int],
) -> list[np.ndarray]:
    """Extract Eulerian path between two odd-degree nodes."""
    try:
        path = list(nx.eulerian_path(graph, source=odd_nodes[0]))
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return _greedy_order(path_segments)

    return _circuit_to_segments(path, segment_edges, path_segments, graph)


def _circuit_to_segments(
    circuit: list[tuple],
    segment_edges: dict,
    path_segments: list[np.ndarray],
    graph: nx.Graph,
) -> list[np.ndarray]:
    """Convert an Euler circuit/path back to ordered path segments."""
    ordered = []
    used_segments = set()

    for u, v in circuit:
        edge_data = graph.edges[u, v]

        if edge_data.get('is_travel', False):
            # This is a travel edge added by the optimizer
            pos_u = graph.nodes[u].get('pos', np.zeros(2))
            pos_v = graph.nodes[v].get('pos', np.zeros(2))
            ordered.append(np.array([pos_u, pos_v]))
            continue

        seg_idx = edge_data.get('segment_idx')
        if seg_idx is not None and seg_idx not in used_segments:
            segment = path_segments[seg_idx]
            # Check direction: should start from u's position
            pos_u = graph.nodes[u].get('pos', np.zeros(2))
            if np.linalg.norm(segment[0] - pos_u) > np.linalg.norm(segment[-1] - pos_u):
                segment = segment[::-1]
            ordered.append(segment)
            used_segments.add(seg_idx)

    # Add any segments that weren't reached
    for i, seg in enumerate(path_segments):
        if i not in used_segments:
            ordered.append(seg)

    return ordered


def _solve_multi_component(
    graph: nx.Graph,
    components: list[set],
    segment_edges: dict,
    path_segments: list[np.ndarray],
) -> list[np.ndarray]:
    """Solve for multiple disconnected components.

    1. Optimize within each component
    2. Connect components via nearest-neighbor TSP
    """
    # Get segments per component
    component_segments = []
    for comp_nodes in components:
        comp_seg_indices = set()
        subgraph = graph.subgraph(comp_nodes)
        for u, v, data in subgraph.edges(data=True):
            idx = data.get('segment_idx')
            if idx is not None:
                comp_seg_indices.add(idx)

        if comp_seg_indices:
            segs = [path_segments[i] for i in sorted(comp_seg_indices)]
            component_segments.append(segs)

    if not component_segments:
        return path_segments

    # Find optimal order to visit components (nearest-neighbor TSP)
    ordered_components = _tsp_nearest_neighbor(component_segments)

    # Concatenate all segments
    result = []
    for comp_segs in ordered_components:
        result.extend(comp_segs)

    return result


def _tsp_nearest_neighbor(component_segments: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """Order components using nearest-neighbor heuristic.

    For each component, compute centroid. Visit components in nearest-neighbor order.
    """
    if len(component_segments) <= 1:
        return component_segments

    # Compute representative point for each component (first point of first segment)
    rep_points = []
    for segs in component_segments:
        if segs and len(segs[0]) > 0:
            rep_points.append(segs[0][0])
        else:
            rep_points.append(np.zeros(2))

    rep_points = np.array(rep_points)

    # Nearest-neighbor tour starting from component closest to origin
    n = len(rep_points)
    dists_to_origin = np.linalg.norm(rep_points, axis=1)
    visited = [False] * n
    order = []

    current = int(np.argmin(dists_to_origin))
    visited[current] = True
    order.append(current)

    for _ in range(n - 1):
        best_dist = float('inf')
        best_next = -1
        for j in range(n):
            if not visited[j]:
                d = np.linalg.norm(rep_points[current] - rep_points[j])
                if d < best_dist:
                    best_dist = d
                    best_next = j
        if best_next >= 0:
            visited[best_next] = True
            order.append(best_next)
            current = best_next

    return [component_segments[i] for i in order]


def _greedy_order(path_segments: list[np.ndarray]) -> list[np.ndarray]:
    """Fallback greedy nearest-endpoint ordering of segments.

    For each unvisited segment, pick the one whose endpoint is nearest
    to the current position.
    """
    if len(path_segments) <= 1:
        return path_segments

    # Build KDTree of all segment start/end points
    endpoints = []
    for seg in path_segments:
        endpoints.append(seg[0])
        endpoints.append(seg[-1])
    endpoints = np.array(endpoints)

    n = len(path_segments)
    used = [False] * n
    ordered = []

    # Start with segment closest to origin
    dists = np.linalg.norm(endpoints[::2], axis=1)  # start points
    current_idx = int(np.argmin(dists))
    used[current_idx] = True
    ordered.append(path_segments[current_idx])
    current_pos = path_segments[current_idx][-1]

    for _ in range(n - 1):
        best_dist = float('inf')
        best_idx = -1
        best_reversed = False

        for j in range(n):
            if used[j]:
                continue
            d_start = np.linalg.norm(current_pos - path_segments[j][0])
            d_end = np.linalg.norm(current_pos - path_segments[j][-1])

            if d_start < best_dist:
                best_dist = d_start
                best_idx = j
                best_reversed = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = j
                best_reversed = True

        if best_idx >= 0:
            used[best_idx] = True
            seg = path_segments[best_idx]
            if best_reversed:
                seg = seg[::-1]
            ordered.append(seg)
            current_pos = seg[-1]

    return ordered
