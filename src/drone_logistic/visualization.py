from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .engine import CircleRegion


def _world_to_grid_point(
    point: tuple[float, float],
    area_size: tuple[int, int] = (20, 27),
    world_limits: tuple[float, float] = (-12.0, 12.0),
    y_limits: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Map a world-space coordinate to the matching cell index used by the demand grid."""
    x, y = point
    if y_limits is None:
        y_limits = world_limits
    rows, cols = area_size
    min_x, max_x = world_limits
    min_y, max_y = y_limits
    grid_x = (x - min_x) / (max_x - min_x) * cols
    grid_y = (y - min_y) / (max_y - min_y) * rows
    return float(grid_x), float(grid_y)


def _save_or_show(fig: plt.Figure, file_name: str, output_dir: str | Path | None = None) -> None:
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "results"
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / file_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


def plot_demand_field(regions: List[CircleRegion], grid_size: int = 80, output_dir: str | Path | None = None) -> None:
    from .engine import build_demand_grid

    area_size = (grid_size, grid_size)
    field = build_demand_grid(regions, area_size=area_size, scale=1.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    mesh = ax.imshow(field, origin="lower", cmap="plasma", aspect="equal", interpolation="nearest", extent=(-12, 12, -13.5, 13.5))
    for region in regions:
        circle = plt.Circle(region.center, region.radius, fill=False, edgecolor="dodgerblue", linestyle="--")
        ax.add_patch(circle)
    ax.set_title("Demand Frequency Matrix")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(mesh, ax=ax, label="Demand count")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-13.5, 13.5)
    _save_or_show(fig, "demand_field.png", output_dir)


def plot_demand_grid(regions: List[CircleRegion], area_size: tuple[int, int] = (20, 27), output_dir: str | Path | None = None) -> None:
    from .engine import build_demand_grid

    grid = build_demand_grid(regions, area_size=area_size, scale=1.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    mesh = ax.imshow(grid, origin="lower", cmap="plasma", aspect="equal", interpolation="nearest", extent=(-12, 12, -13.5, 13.5))
    fig.colorbar(mesh, ax=ax, label="Demand count")
    for region in regions:
        ax.add_patch(plt.Circle(region.center, region.radius, fill=False, edgecolor="white", linewidth=1.2, linestyle="--"))
    ax.set_title("Demand Frequency Matrix")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-13.5, 13.5)
    _save_or_show(fig, "demand_grid.png", output_dir)


def plot_station_coverage(regions: List[CircleRegion], stations: list[tuple[float, float]], max_range: float = 5.0, output_dir: str | Path | None = None) -> None:
    from .engine import build_demand_grid

    fig, ax = plt.subplots(figsize=(7, 7))
    grid = build_demand_grid(regions, area_size=(20, 27), scale=1.0)
    mesh = ax.imshow(grid, origin="lower", cmap="plasma", aspect="equal", extent=(-12, 12, -13.5, 13.5))
    for region in regions:
        ax.add_patch(plt.Circle(region.center, region.radius, fill=False, edgecolor="dodgerblue", linewidth=1.5, linestyle="--"))
    for station in stations:
        x, y = station
        ax.scatter(x, y, color="lime", s=80, marker="*", zorder=3)
        ax.add_patch(plt.Circle((x, y), max_range, fill=False, edgecolor="lime", linewidth=1.2, alpha=0.7))
    ax.set_title("Station Placement and Coverage Radius")
    fig.colorbar(mesh, ax=ax, label="Demand density")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-13.5, 13.5)
    _save_or_show(fig, "station_coverage.png", output_dir)


def plot_regions(regions: List[CircleRegion], output_dir: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for region in regions:
        circle = plt.Circle(region.center, region.radius, fill=False, edgecolor="navy", linewidth=1.5)
        ax.add_patch(circle)
        ax.plot(*region.center, "o", color="black")
        for demand in region.demand_points:
            ax.plot(demand.start[0], demand.start[1], "o", color="tab:red", alpha=0.8)
            ax.plot(demand.end[0], demand.end[1], "o", color="tab:blue", alpha=0.8)
            ax.arrow(
                demand.start[0],
                demand.start[1],
                demand.end[0] - demand.start[0],
                demand.end[1] - demand.start[1],
                color="tab:blue",
                length_includes_head=True,
                head_width=0.1,
                head_length=0.12,
                alpha=0.6,
            )
    ax.set_aspect("equal")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_title("Circular Service Regions and Demand Tasks")
    _save_or_show(fig, "service_regions.png", output_dir)


def plot_delivery_trace(regions: List[CircleRegion], traces: list[dict], output_dir: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for region in regions:
        circle = plt.Circle(region.center, region.radius, fill=False, edgecolor="navy", linewidth=1.5)
        ax.add_patch(circle)
    for trace in traces:
        path = trace.get("path", [])
        if len(path) < 2:
            continue
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        ax.plot(xs, ys, alpha=0.9, linewidth=1.4, label=f"drone {trace.get('drone_id')}")
    ax.set_title("Drone Mission Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.legend(loc="best", fontsize=8)
    _save_or_show(fig, "delivery_trace.png", output_dir)


def plot_validation_suite(regions: List[CircleRegion], stations: list[tuple[float, float]], traces: list[dict], output_dir: str | Path | None = None) -> None:
    from .engine import build_demand_grid

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    grid = build_demand_grid(regions, area_size=(20, 27), scale=1.0)
    mesh = axes[0].imshow(grid, origin="lower", cmap="plasma", aspect="equal", extent=(-12, 12, -13.5, 13.5))
    axes[0].set_title("1. Demand frequency matrix")
    for region in regions:
        axes[0].add_patch(plt.Circle(region.center, region.radius, fill=False, edgecolor="dodgerblue", linestyle="--"))
    for station in stations:
        axes[0].scatter(station[0], station[1], color="lime", s=60, marker="*")
    for region in regions:
        axes[1].add_patch(plt.Circle(region.center, region.radius, fill=False, edgecolor="navy", linewidth=1.5))
    for station in stations:
        axes[1].scatter(station[0], station[1], color="darkorange", s=60, marker="*")
        axes[1].add_patch(plt.Circle(station, 5.0, fill=False, edgecolor="darkorange", linewidth=1.1, alpha=0.7))
    axes[1].set_title("2. Station coverage")
    for trace in traces:
        path = trace.get("path", [])
        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            axes[2].plot(xs, ys, alpha=0.9, linewidth=1.5)
    for region in regions:
        axes[2].add_patch(plt.Circle(region.center, region.radius, fill=False, edgecolor="navy", linewidth=1.3))
    axes[2].set_title("3. Drone trajectories")
    for ax in axes:
        ax.set_xlim(-12, 12)
        ax.set_ylim(-13.5, 13.5)
        ax.set_aspect("equal")
    fig.colorbar(mesh, ax=axes[0], label="Demand count")
    fig.tight_layout()
    _save_or_show(fig, "validation_suite.png", output_dir)
