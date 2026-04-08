# EEP Simulation Skeleton

This folder is the runnable simulation workspace for the EEP scenario matrix.

## Structure

- `config/`: scenario matrix and parameter configs
- `src/eep_sim/`: simulation code package
- `tests/`: unit tests
- `outputs/`: generated CSV outputs

The raw input CSV files live at the repository root under `data/raw/`.

## First reusable module

`src/eep_sim/soc.py` provides a generic battery SOC estimator with:

- charge/discharge power limits
- SOC bounds
- charge/discharge efficiency
- fixed time-step simulation

## Core optimization module

`src/eep_sim/core_model.py` includes the requested core constraints:

- `sum_i x[i,w] <= M_total`
- `M_active[i,w] = x[i,w-delay]` (or initial active value in early windows)
- `sum_i |x[i,w]-x[i,w-1]| <= R_w`
- hourly power balance, SOC dynamics/bounds, mutually exclusive charge/discharge, charge/discharge limits, grid limits, unserved load
- objective: `C_grid + C_unserved + C_reconf + C_storage`

## Demo run

```bash
uv run python -m eep_sim.run_core_demo
```

## G2 data loader

`src/eep_sim/g2_loader.py` reads:

- `data/raw/g2_beijing_shanghai_service_areas_hourly_load_revised.csv`
- `data/raw/g2_beijing_shanghai_service_areas_hourly_price_absolute_corrected.csv`
- `data/raw/g2_beijing_shanghai_service_areas_load_grid_params_revised.csv`

and maps them to `CoreModelData`.

Timeline modes:

- `representative_12x24` (default): 12 months x 24 hourly slices = 288 steps
- `expanded_8760`: expands month-hour slices to all days in a non-leap year = 8760 steps

Quick check:

```bash
uv run python -m eep_sim.run_g2_loader_demo --base-dir .
```

By default, the loader derives `MESS` total capacity from:

- total fixed-storage capacity across sites
- `mobile_capacity_ratio_to_fixed`
- rounding to the nearest integer MWh

You can still override this by passing `--m-total-mwh`.

Additional default scenario logic:

- symbolic unserved-load penalties default to opportunity cost:
  `price(i,t) * 1.0`
- grid capacity uses `max(csv_cap, avg_load * 1.25)`
- Chinese holiday load shocks are applied in `expanded_8760`
  during `09:00-21:00` on configured holiday dates

Run with 8760 expansion:

```bash
uv run python -m eep_sim.run_g2_loader_demo --base-dir . --timeline-mode expanded_8760 --reconf-hours 168
```

Solve dedicated Spring Festival / National Day stress windows:

```bash
uv run python -m eep_sim.run_holiday_stress_demo --base-dir .
```

## Quick test

Run from repository root:

```bash
uv sync
uv run python -m unittest discover -s sim/tests -p "test_*.py" -v
```
