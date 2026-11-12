# Drone Logistics Coverage Study

This repository is a compact, GitHub-ready Python version of the drone logistics study implemented in the original MATLAB workflow. The project stays focused on the actual logic used in the code: circular demand regions, a 2D demand-frequency matrix, coverage-based station placement, and a limited fleet of drones operating under range and time constraints.

## Research workflow

The project follows the actual execution flow implemented in the code:

1. Generate circular demand regions and delivery tasks.
2. Convert those demand points into a 2D demand-frequency matrix.
3. Use the matrix as the coverage map for station placement.
4. Optimize station locations with a coverage penalty objective.
5. Simulate a drone fleet operating under bounded range and task deadlines.
6. Evaluate performance with a logistics-efficiency metric.

The default workflow follows the implemented demand matrix approach. The core system is based on the circular-demand model, demand-frequency counting, and station optimization logic that match the original code path.

## Why this matches the real implementation

The original MATLAB files in the workspace show the core stages that are actually relevant to the implemented model:

- demand generation from circles and data points
- station optimization using coverage gaps and penalties
- UAV path simulation with state transitions and available range

The Python code keeps these steps and does not introduce an extra image-processing model as the main logic.

## Project structure

- `src/drone_logistic/engine.py` — demand generation, demand matrix construction, station optimization, and drone simulation
- `src/drone_logistic/visualization.py` — plots for the demand matrix and station coverage layout
- `examples/run_demo.py` — one-command experiment entry point
- `tests/test_engine.py` — regression tests for generation and simulation
- `data/` — generated demand data and experiment outputs

## Quick start

```bash
cd drone_logistic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/run_demo.py
```

The script saves the generated output in the `results/` directory.

## Core metric

The study evaluates system performance using the paper-style measure:

E = completed_orders / (time * cost)

This keeps the code aligned with the original research idea while staying easy to read and reproduce.

## Main functions

- `generate_demand_regions()` — creates circular demand areas and order pairs
- `optimize_station_positions()` — approximates the MATLAB station-selection logic
- `simulate_delivery()` — runs the drone dispatch simulation
- `logistics_efficiency()` — computes the final efficiency score

