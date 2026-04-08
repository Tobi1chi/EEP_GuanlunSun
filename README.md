# EEP Remote

This is a public repository of my final individual project simulation codebase and the latest experiment outputs.

This repository compares fixed, mobile, and hybrid storage systems against a no-storage baseline for constrained EV charging corridors. Mobile storage is modeled as a relocatable capacity pool that activates with a delay across reconfiguration windows—rather than as an hourly routing asset—to better capture non-emergency, planning-level deployment scenarios.

Important interpretation note: the packaged `E1` to `E10` results use the repository's default sizing rule, in which total mobile capacity is derived as `0.7 * fixed_total` unless manually overridden. Those outputs are therefore not strict same-budget or same-capacity comparisons by default.

## Repository layout

- `data/raw/`: revised input CSV files used by the loader
- `sim/src/eep_sim/`: simulation package and experiment entrypoints
- `sim/tests/`: unit tests
- `sim/outputs/`: generated CSV outputs, including the master scenario table
- `figures/experiments/`: exported figure assets
- `reports/`: per-experiment memos and the overall experiment handoff
- `docs/`: workflow and modeling notes

## Quick start

This repository is configured for `uv`.

```bash
uv sync
uv run python -m unittest discover -s sim/tests -p "test_*.py" -v
uv run python -m eep_sim.run_g2_loader_demo --base-dir .
uv run python -m eep_sim.run_capacity_fair_baseline_demo --base-dir .
```

Run the full experiment suite from the repository root:

```bash
uv run python -m eep_sim.run_experiments --base-dir . --run-all --solver highs
```

## Included results

The packaged results already include:

- experiment summaries `E0` to `E10` under `reports/`
- CSV outputs under `sim/outputs/`
- figure exports under `figures/experiments/`
- consolidated handoff note at `reports/experiment_handoff.md`

