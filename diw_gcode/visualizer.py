"""Toolpath visualization for debugging and preview."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .toolpath import LayerToolpath


def plot_layer(
    toolpath: LayerToolpath,
    title: str = None,
    show: bool = True,
    save_path: str = None,
) -> None:
    """Plot a single layer's toolpath with color-coded segments."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for segment in toolpath.segments:
        pts = segment.points
        if len(pts) < 2:
            continue

        if segment.segment_type == "perimeter":
            color = 'blue'
            lw = 1.5
        elif segment.segment_type == "infill":
            color = 'green'
            lw = 1.0
        elif segment.segment_type == "travel":
            color = 'red'
            lw = 0.5
        else:
            color = 'gray'
            lw = 1.0

        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Layer Z={toolpath.z_height:.2f}mm"
    title += f" (travel={toolpath.travel_distance:.1f}mm, print={toolpath.print_distance:.1f}mm)"
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1.5, label='Perimeter'),
        Line2D([0], [0], color='green', linewidth=1.0, label='Infill'),
        Line2D([0], [0], color='red', linewidth=0.5, label='Travel'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Layer plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_layers_3d(
    toolpaths: list[LayerToolpath],
    show: bool = True,
    save_path: str = None,
) -> None:
    """Plot all layers in 3D view."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for tp in toolpaths:
        z = tp.z_height
        for segment in tp.segments:
            pts = segment.points
            if len(pts) < 2:
                continue

            zs = np.full(len(pts), z)

            if segment.segment_type == "perimeter":
                ax.plot(pts[:, 0], pts[:, 1], zs, color='blue', linewidth=0.5, alpha=0.7)
            elif segment.segment_type == "infill":
                ax.plot(pts[:, 0], pts[:, 1], zs, color='green', linewidth=0.3, alpha=0.5)
            elif segment.segment_type == "travel":
                ax.plot(pts[:, 0], pts[:, 1], zs, color='red', linewidth=0.2, alpha=0.3)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Full Print ({len(toolpaths)} layers)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"3D plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_stats(toolpaths: list[LayerToolpath]) -> None:
    """Print toolpath statistics."""
    total_travel = sum(tp.travel_distance for tp in toolpaths)
    total_print = sum(tp.print_distance for tp in toolpaths)
    num_travels = sum(
        1 for tp in toolpaths
        for seg in tp.segments
        if seg.segment_type == "travel"
    )

    print("\n=== Toolpath Statistics ===")
    print(f"Layers: {len(toolpaths)}")
    print(f"Total print distance: {total_print:.1f} mm")
    print(f"Total travel distance: {total_travel:.1f} mm")
    print(f"Travel ratio: {total_travel / max(total_print, 0.01) * 100:.1f}%")
    print(f"Number of travel moves: {num_travels}")

    if toolpaths:
        max_travel_layer = max(toolpaths, key=lambda tp: tp.travel_distance)
        print(f"Worst layer (most travel): Z={max_travel_layer.z_height:.2f}mm "
              f"({max_travel_layer.travel_distance:.1f}mm travel)")
