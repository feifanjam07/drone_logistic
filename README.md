# Drone Logistics

A Python project for drone delivery simulation and station coverage optimization.

Delivery demand is generated within circular regions and mapped onto a grid for station placement. The fleet simulation models deliveries under range and time constraints, tracking completed orders, operating cost, and delivery efficiency.

## Getting started

Requires Python 3.10 or later. Run from the project root:

```bash
python -m pip install -r requirements.txt
python examples/run_demo.py
```

The demo prints a simulation summary and saves plots of demand, station coverage, and delivery routes to `results/`. Generated demand data is saved to `data/generated_cases.json`.

## Files

- `src/drone_logistic/` — simulation, optimization, and plotting code
- `examples/run_demo.py` — example experiment
- `data/` — sample demand data
- `tests/` — simulation and coordinate mapping tests

Run the tests with:

```bash
python -m pytest tests
```

