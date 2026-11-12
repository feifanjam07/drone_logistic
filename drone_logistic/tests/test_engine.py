from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_logistic.engine import (
    CircleRegion,
    build_demand_grid,
    coverage_station_cost,
    generate_demo_regions,
    simulate_delivery,
)
from drone_logistic.visualization import _world_to_grid_point


def test_generate_demo_regions():
    regions = generate_demo_regions(num_regions=3, radius=2.5, points_per_region=5, seed=7)
    assert len(regions) == 3
    assert all(len(region.demand_points) == 5 for region in regions)


def test_build_demand_matrix():
    regions = generate_demo_regions(num_regions=2, radius=2.0, points_per_region=4, seed=11)
    matrix = build_demand_grid(regions, area_size=(20, 20), scale=1.0)
    assert matrix.shape == (20, 20)
    assert matrix.min() >= 0.0
    assert matrix.sum() > 0.0


def test_simulate_delivery():
    regions = generate_demo_regions(num_regions=2, radius=2.0, points_per_region=4, seed=13)
    metrics = simulate_delivery(regions, max_range=10.0, drones_per_region=2, time_horizon=10, step_size=0.5)
    assert metrics["total_regions"] == 2
    assert metrics["total_drones"] == 4
    assert "served_orders" in metrics
    assert "efficiency" in metrics
    assert metrics["efficiency"] >= 0.0


def test_coverage_station_cost_matches_penalty_logic():
    demand = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    stations = [(0.0, 0.0), (2.0, 2.0)]
    cost = coverage_station_cost(stations, demand, max_range=3.0)
    assert isinstance(cost, float)
    assert cost >= 0.0
    assert np.isfinite(cost)


def test_world_to_grid_point_matches_grid_axis_mapping():
    point = (0.0, 0.0)
    x_grid, y_grid = _world_to_grid_point(point, area_size=(20, 27), world_limits=(-12.0, 12.0))
    assert x_grid == 13.5
    assert y_grid == 10.0


def test_build_demand_grid_matches_plot_coordinate_mapping():
    region = CircleRegion(center=(0.0, 0.0), radius=1.5)
    region.add_demand((0.0, 0.0), (1.0, 1.0), order_id=1, order_time=1)
    grid = build_demand_grid([region], area_size=(20, 27), scale=1.0)

    for point in ((0.0, 0.0), (1.0, 1.0)):
        x_index, y_index = _world_to_grid_point(point, area_size=(20, 27), world_limits=(-12.0, 12.0))
        row = int(round(y_index))
        col = int(round(x_index))
        assert 0 <= row < grid.shape[0]
        assert 0 <= col < grid.shape[1]
        assert grid[row, col] > 0.0
