from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_logistic.engine import (
    generate_demand_regions,
    optimize_station_positions,
    save_regions,
    simulate_delivery,
)
from drone_logistic.visualization import (
    plot_demand_grid,
    plot_delivery_trace,
    plot_regions,
    plot_station_coverage,
    plot_validation_suite,
)


if __name__ == "__main__":
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # MATLAB-inspired workflow: generate circular demand regions, then optimize station placement.
    regions = generate_demand_regions(num_regions=4, radius=5.0, points_per_region=10, seed=42)
    stations = optimize_station_positions(regions, max_range=5.0, area_size=(20, 27), num_stations=4)
    print("Optimized stations:")
    print(stations)

    data_path = ROOT / "data" / "generated_cases.json"
    save_regions(regions, data_path)

    metrics, traces = simulate_delivery(
        regions,
        max_range=10.0,
        drones_per_region=3,
        time_horizon=20,
        step_size=0.5,
        drone_speed=1.0,
        return_trace=True,
    )
    print("Simulation summary:")
    print(metrics)
    print(f"Logistics efficiency E = {metrics['efficiency']:.6f} orders /(time * cost)")

    plot_regions(regions, output_dir=results_dir)
    plot_demand_grid(regions, output_dir=results_dir)
    plot_station_coverage(regions, stations, max_range=5.0, output_dir=results_dir)
    plot_delivery_trace(regions, traces, output_dir=results_dir)
    plot_validation_suite(regions, stations, traces, output_dir=results_dir)

    print(f"Saved figures to: {results_dir}")
