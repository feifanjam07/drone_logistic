from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class DemandPoint:
    start: Point
    end: Point
    order_id: int
    order_time: int


@dataclass
class CircleRegion:
    center: Point
    radius: float
    demand_points: List[DemandPoint] = field(default_factory=list)

    def add_demand(self, start: Point, end: Point, order_id: int, order_time: int) -> None:
        self.demand_points.append(
            DemandPoint(start=start, end=end, order_id=order_id, order_time=order_time)
        )


@dataclass
class DroneState:
    drone_id: int
    center: Point
    position: Point
    remaining_range: float
    status: int = 0
    assigned_order: int | None = None
    route: List[Point] = field(default_factory=list)
    mile: float = 0.0
    prior: float = 0.0
    last_direction: float = 0.0


def euclidean_distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def world_to_grid_index(
    point: Point,
    area_size: Tuple[int, int] = (20, 27),
    x_limits: Tuple[float, float] = (-12.0, 12.0),
    y_limits: Tuple[float, float] | None = None,
) -> Tuple[int, int]:
    """Map world coordinates to the corresponding matrix row/column for the demand grid."""
    if y_limits is None:
        y_limits = x_limits
    rows, cols = area_size
    x, y = point

    col = int(np.clip(np.rint((x - x_limits[0]) / (x_limits[1] - x_limits[0]) * cols), 0, cols - 1))
    row = int(np.clip(np.rint((y - y_limits[0]) / (y_limits[1] - y_limits[0]) * rows), 0, rows - 1))
    return row, col


def grid_to_world_point(
    row: int,
    col: int,
    area_size: Tuple[int, int] = (20, 27),
    x_limits: Tuple[float, float] = (-12.0, 12.0),
    y_limits: Tuple[float, float] | None = None,
) -> Point:
    """Convert a matrix cell back into world coordinates for plotting and station placement."""
    if y_limits is None:
        y_limits = x_limits
    rows, cols = area_size
    x = x_limits[0] + (col + 0.5) * (x_limits[1] - x_limits[0]) / cols
    y = y_limits[0] + (row + 0.5) * (y_limits[1] - y_limits[0]) / rows
    return (float(x), float(y))


def generate_demo_regions(
    num_regions: int = 4,
    radius: float = 2.5,
    points_per_region: int = 12,
    seed: int = 42,
) -> List[CircleRegion]:
    """Generate synthetic city-like circular service regions with demand points."""
    rng = np.random.default_rng(seed)
    regions: List[CircleRegion] = []
    attempts = 0

    while len(regions) < num_regions and attempts < 1000:
        center = (rng.uniform(-8, 8), rng.uniform(-8, 8))
        if any(euclidean_distance(center, existing.center) < 2.5 * radius for existing in regions):
            attempts += 1
            continue

        region = CircleRegion(center=center, radius=radius)
        for order_id in range(1, points_per_region + 1):
            angle_start = rng.uniform(0.0, 2 * np.pi)
            r_start = rng.uniform(0.0, radius)
            angle_end = rng.uniform(0.0, 2 * np.pi)
            r_end = rng.uniform(0.0, radius)

            start = (
                center[0] + r_start * np.cos(angle_start),
                center[1] + r_start * np.sin(angle_start),
            )
            end = (
                center[0] + r_end * np.cos(angle_end),
                center[1] + r_end * np.sin(angle_end),
            )
            order_time = int(rng.integers(0, 10))
            region.add_demand(start, end, order_id, order_time)

        regions.append(region)
        attempts += 1

    if len(regions) != num_regions:
        raise RuntimeError("Unable to generate the requested number of valid service regions.")

    return regions


def generate_demand_regions(
    num_regions: int = 4,
    radius: float = 5.0,
    points_per_region: int = 10,
    min_center: float = -8.0,
    max_center: float = 8.0,
    seed: int = 42,
) -> List[CircleRegion]:
    """MATLAB-style demand generation from separate circular service regions.

    This mirrors the real logic behind the original MATLAB scripts: each circular region has a
    center and a set of demand points generated inside the circle.
    """
    effective_radius = max(1.2, min(float(radius), 2.5))
    return generate_demo_regions(
        num_regions=num_regions,
        radius=effective_radius,
        points_per_region=points_per_region,
        seed=seed,
    )


def build_demand_grid(
    regions: List[CircleRegion],
    area_size: Tuple[int, int] = (20, 27),
    scale: float = 1.0,
) -> np.ndarray:
    """Convert circular demand data into a 2D grid, matching the MATLAB station-optimization workflow."""
    rows, cols = area_size
    grid = np.zeros((rows, cols), dtype=float)
    x_limits = (-12.0, 12.0)
    y_limits = (-13.5, 13.5)

    for region in regions:
        for demand in region.demand_points:
            for point in (demand.start, demand.end):
                row, col = world_to_grid_index(
                    point,
                    area_size=(rows, cols),
                    x_limits=x_limits,
                    y_limits=y_limits,
                )
                if 0 <= row < rows and 0 <= col < cols:
                    grid[row, col] += 1.0 / scale
    return grid


def coverage_station_cost(
    stations: List[Tuple[float, float]] | Tuple[Tuple[float, float], ...],
    demand_grid: np.ndarray,
    max_range: float = 5.0,
) -> float:
    """MATLAB-style coverage penalty cost used in the original station optimizer.

    For each station, the supplied coverage is computed over every demand cell within the
    coverage radius. The objective penalizes uncovered demand and then sums the positive supply
    surplus over the grid.
    """
    rows, cols = demand_grid.shape
    supply = np.zeros_like(demand_grid, dtype=float)
    x_limits = (-12.0, 12.0)
    y_limits = (-13.5, 13.5)

    for station_x, station_y in stations:
        for row in range(rows):
            for col in range(cols):
                cell_x = x_limits[0] + (col + 0.5) * (x_limits[1] - x_limits[0]) / cols
                cell_y = y_limits[0] + (row + 0.5) * (y_limits[1] - y_limits[0]) / rows
                dist = np.hypot(cell_x - station_x, cell_y - station_y)
                if dist <= max_range:
                    supply[row, col] += 20.0 * (1800.0 / (4.0 * dist / 10.0 + 30.0))

    diff = supply - demand_grid
    cost = float(np.sum(np.maximum(diff, 0.0)))
    uncovered_count = int(np.count_nonzero(diff < 0.0))
    if uncovered_count > 0:
        cost += 1e9 * uncovered_count
    return cost


def optimize_station_positions(
    regions: List[CircleRegion],
    max_range: float = 5.0,
    area_size: Tuple[int, int] = (20, 27),
    num_stations: int = 4,
) -> List[Tuple[float, float]]:
    """MATLAB-inspired station placement using the original supply-demand penalty logic."""
    grid = build_demand_grid(regions, area_size=area_size, scale=1.0)
    candidates = np.argwhere(grid > 0)
    if len(candidates) == 0:
        return [(0.0, 0.0) for _ in range(num_stations)]

    ranked = sorted(
        [((int(iy), int(ix)), float(grid[iy, ix])) for iy, ix in candidates],
        key=lambda item: item[1],
        reverse=True,
    )

    selected: List[Tuple[float, float]] = []
    for (iy, ix), _ in ranked:
        point = grid_to_world_point(
            iy,
            ix,
            area_size=area_size,
            x_limits=(-12.0, 12.0),
            y_limits=(-13.5, 13.5),
        )
        if all(euclidean_distance(point, existing) >= 2.0 for existing in selected):
            selected.append(point)
        if len(selected) >= num_stations:
            break

    if len(selected) < num_stations:
        for iy, ix in candidates[: max(1, num_stations * 3)]:
            point = grid_to_world_point(
                iy,
                ix,
                area_size=area_size,
                x_limits=(-12.0, 12.0),
                y_limits=(-13.5, 13.5),
            )
            if all(euclidean_distance(point, existing) >= 1.0 for existing in selected):
                selected.append(point)
            if len(selected) >= num_stations:
                break

    if len(selected) < num_stations:
        selected = [(0.0, 0.0) for _ in range(num_stations)]

    best_stations = selected[:num_stations]
    best_cost = coverage_station_cost(best_stations, grid, max_range=max_range)

    for _ in range(30):
        improved = False
        for i in range(len(best_stations)):
            current = best_stations[i]
            best_local_move = current
            best_local_cost = best_cost
            for dx, dy in [
                (0.0, 0.0),
                (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
                (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5),
            ]:
                trial = (current[0] + dx, current[1] + dy)
                if abs(trial[0]) > 12.0 or abs(trial[1]) > 12.0:
                    continue
                candidate = best_stations.copy()
                candidate[i] = trial
                if any(euclidean_distance(trial, other) < 1.0 for idx, other in enumerate(candidate) if idx != i):
                    continue
                candidate_cost = coverage_station_cost(candidate, grid, max_range=max_range)
                if candidate_cost < best_local_cost:
                    best_local_move = trial
                    best_local_cost = candidate_cost
            if best_local_move != current:
                best_stations[i] = best_local_move
                best_cost = best_local_cost
                improved = True
        if not improved:
            break

    return best_stations[:num_stations]


def save_regions(regions: List[CircleRegion], path: str | Path) -> None:
    payload = []
    for region in regions:
        payload.append(
            {
                "center": list(region.center),
                "radius": region.radius,
                "demand_points": [
                    {
                        "start": list(point.start),
                        "end": list(point.end),
                        "order_id": point.order_id,
                        "order_time": point.order_time,
                    }
                    for point in region.demand_points
                ],
            }
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))


def load_regions(path: str | Path) -> List[CircleRegion]:
    data = json.loads(Path(path).read_text())
    regions: List[CircleRegion] = []
    for item in data:
        region = CircleRegion(center=tuple(item["center"]), radius=item["radius"])
        for point in item["demand_points"]:
            region.add_demand(
                start=tuple(point["start"]),
                end=tuple(point["end"]),
                order_id=int(point["order_id"]),
                order_time=int(point["order_time"]),
            )
        regions.append(region)
    return regions
def potential_direction(position: Point, target: Point, center: Point) -> Point:
    """Return a direction vector that biases movement toward both the target and the region center."""
    to_target = np.array(target, dtype=float) - np.array(position, dtype=float)
    to_center = np.array(center, dtype=float) - np.array(position, dtype=float)

    vector = to_target + 0.35 * to_center
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return (1.0, 0.0)
    return tuple(vector / norm)


def route_cost(start: Point, end: Point, center: Point) -> float:
    return euclidean_distance(start, end) + euclidean_distance(end, center)


def logistics_efficiency(completed_orders: int, elapsed_time: float, total_cost: float) -> float:
    """Paper-aligned metric: orders completed per unit time per unit cost."""
    if elapsed_time <= 0 or total_cost <= 0:
        return 0.0
    return completed_orders / (elapsed_time * total_cost)


def simulate_delivery(
    regions: List[CircleRegion],
    max_range: float = 10.0,
    drones_per_region: int = 3,
    time_horizon: int = 30,
    step_size: float = 0.5,
    drone_speed: float = 1.0,
    return_trace: bool = False,
) -> dict | tuple[dict, list[dict]]:
    """MATLAB-inspired drone simulation with status codes, mileage limits, and return-to-base logic."""
    drone_states: List[DroneState] = []
    finished_orders: List[DemandPoint] = []
    active_assignments: dict[int, DemandPoint] = {}
    drone_trace: dict[int, list[Point]] = {}
    region_order_map: dict[int, List[DemandPoint]] = {idx: list(region.demand_points) for idx, region in enumerate(regions)}

    for region_index, region in enumerate(regions):
        for drone_idx in range(drones_per_region):
            drone_id = drone_idx + 1 + region_index * 100
            drone_state = DroneState(
                drone_id=drone_id,
                center=region.center,
                position=region.center,
                remaining_range=max_range,
                status=0,
                mile=0.0,
                prior=0.0,
            )
            drone_states.append(drone_state)
            drone_trace[drone_id] = [region.center]

    metrics = {
        "total_regions": len(regions),
        "total_drones": len(drone_states),
        "served_orders": 0,
        "completed_steps": 0,
        "efficiency": 0.0,
        "parameters": {
            "max_range": max_range,
            "drones_per_region": drones_per_region,
            "time_horizon": time_horizon,
            "drone_speed": drone_speed,
        },
        "region_summary": [],
    }

    for t in range(time_horizon):
        for region_idx, region in enumerate(regions):
            for demand in region.demand_points:
                if demand.order_time == t and demand not in finished_orders:
                    active_assignments[id(demand)] = demand

        for drone in drone_states:
            region_index = next((idx for idx, region in enumerate(regions) if euclidean_distance(drone.center, region.center) < 1e-6), 0)
            region = regions[region_index]
            if drone.status in {0, 3, 4}:
                best_order = None
                best_cost = float("inf")
                for demand in region.demand_points:
                    if id(demand) in active_assignments and demand not in finished_orders:
                        tdis = (
                            euclidean_distance(drone.position, demand.start)
                            + euclidean_distance(demand.start, demand.end)
                            + euclidean_distance(demand.end, drone.center)
                            + 0.2
                        )
                        if tdis < best_cost and tdis <= max_range:
                            best_order = demand
                            best_cost = tdis
                if best_order is not None:
                    drone.assigned_order = best_order.order_id
                    drone.status = 1
                    drone.route = [best_order.start, best_order.end, drone.center]
                    drone.remaining_range = max_range - drone.mile
                    active_assignments.pop(id(best_order), None)
                    finished_orders.append(best_order)
                    drone.prior = float(t)

            if drone.status in {1, 2, 3} and drone.route:
                target = drone.route[0]
                if drone.status == 1:
                    target_stage = "pickup"
                elif drone.status == 2:
                    target_stage = "delivery"
                else:
                    target_stage = "return"
                target_vector = np.array(target, dtype=float) - np.array(drone.position, dtype=float)
                repulsion = np.array([0.0, 0.0], dtype=float)
                for other in drone_states:
                    if other.drone_id == drone.drone_id:
                        continue
                    if euclidean_distance(drone.position, other.position) <= 0.2:
                        dy = other.position[1] - drone.position[1]
                        dx = other.position[0] - drone.position[0]
                        if abs(dx) + abs(dy) > 0:
                            repulsion += np.array([-(dy / (abs(dx) + abs(dy) + 1e-9)), -(dx / (abs(dx) + abs(dy) + 1e-9))])
                if np.linalg.norm(target_vector) < 1e-9:
                    direction = drone.last_direction
                else:
                    direction = np.arctan2(target_vector[1], target_vector[0]) + 0.15 * np.arctan2(repulsion[1], repulsion[0])
                next_x = drone.position[0] + np.cos(direction) * step_size
                next_y = drone.position[1] + np.sin(direction) * step_size
                drone.position = (next_x, next_y)
                drone.last_direction = direction
                drone.mile += step_size * drone_speed
                drone.remaining_range = max_range - drone.mile
                drone_trace[drone.drone_id].append(drone.position)
                metrics["completed_steps"] += 1

                if euclidean_distance(drone.position, target) <= step_size:
                    drone.position = target
                    drone_trace[drone.drone_id].append(drone.position)
                    drone.route.pop(0)
                    if drone.status == 1:
                        drone.status = 2
                    elif drone.status == 2:
                        drone.status = 3
                        metrics["served_orders"] += 1
                    elif drone.status == 3:
                        drone.status = 4
                        drone.position = drone.center
                        drone.mile = max(0.0, drone.mile)
                elif drone.mile >= max_range:
                    drone.status = 3
                    drone.route = [drone.center]

            elif drone.status == 4:
                drone.mile = max(0.0, drone.mile - 0.5 * step_size)
                if drone.mile <= 0.0:
                    drone.mile = 0.0
                    drone.status = 0
                    drone.position = drone.center
                    drone_trace[drone.drone_id].append(drone.position)

    for region in regions:
        served = sum(1 for demand in region.demand_points if demand in finished_orders)
        metrics["region_summary"].append(
            {
                "center": region.center,
                "radius": region.radius,
                "served": served,
                "total": len(region.demand_points),
            }
        )

    total_cost = float(metrics["completed_steps"]) * (1.0 / max(drone_speed, 1e-6))
    metrics["efficiency"] = logistics_efficiency(
        completed_orders=metrics["served_orders"],
        elapsed_time=float(time_horizon),
        total_cost=total_cost,
    )

    if return_trace:
        trace_payload = [
            {"drone_id": drone_id, "center": drone.center, "path": path}
            for drone_id, path in drone_trace.items()
            for drone in drone_states if drone.drone_id == drone_id
        ]
        return metrics, trace_payload
    return metrics
