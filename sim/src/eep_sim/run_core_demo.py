"""Minimal runnable demo for the core optimization model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model


def build_demo_data() -> CoreModelData:
    sites = ["A", "B"]
    hours = list(range(6))  # 6 hourly steps
    windows = [0, 1, 2]  # each window spans 2 hours
    hour_to_window = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    load = {}
    price = {}
    grid_limit = {}
    penalty = {}
    for i in sites:
        for t in hours:
            load[(i, t)] = 1.0 if i == "A" else 0.9
            price[(i, t)] = 400.0 if t in (2, 3, 4) else 200.0
            grid_limit[(i, t)] = 0.7
            penalty[(i, t)] = 1000.0

    fixed_capacity = {"A": 0.3, "B": 0.3}
    soc_initial = {"A": 0.5, "B": 0.5}
    soc_min = {"A": 0.1, "B": 0.1}
    soc_max = {"A": 0.9, "B": 0.9}
    charge_rate = {"A": 1.0, "B": 1.0}
    discharge_rate = {"A": 1.0, "B": 1.0}

    active_init = {
        ("A", 0): 0.1,
        ("B", 0): 0.1,
        ("A", 1): 0.0,
        ("B", 1): 0.0,
        ("A", 2): 0.0,
        ("B", 2): 0.0,
    }

    reconf_limit = {0: 1e9, 1: 0.3, 2: 0.3}

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh=fixed_capacity,
        soc_initial_frac=soc_initial,
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
        charge_c_rate=charge_rate,
        discharge_c_rate=discharge_rate,
        eta_charge=0.95,
        eta_discharge=0.95,
        dt_hours=1.0,
        m_total_mwh=0.6,
        delay_windows=1,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=0.5,
        c_storage_fixed_yuan_per_mwh=0.0,
        c_storage_mobile_yuan_per_mwh=0.0,
    )


def main() -> None:
    data = build_demo_data()
    model = build_core_model(data)
    result = solve_core_model(model, solver_name="highs")
    solution = extract_solution(model)

    print("solver status:", result.solver.status)
    print("termination:", result.solver.termination_condition)
    print("costs:", solution["costs"])
    print("x decisions:", solution["x"])


if __name__ == "__main__":
    main()
