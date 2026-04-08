"""Scan MESS-vs-fixed boundaries over peak persistence and travel delay.

This script uses the current window-level MESS model and a small synthetic
corridor whose parameter magnitudes are anchored to the current G2 assumptions:

- 5 corridor sites
- fixed capacities by archetype A/B/C/B/A = 0.4/0.8/1.2/0.8/0.4 MWh
- mobile total derived from fixed total * 0.7 and rounded -> 3.0 MWh
- hourly operation with window-level reconfiguration

Interpretation note:
- "peak duration" is modeled as the number of consecutive reconfiguration
  windows for which the hotspot stays at the same site.
- "travel time" is represented by `delay_windows`, consistent with the current
  MESS abstraction in `core_model.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model


SITES = ["N1", "N2", "N3", "N4", "N5"]
FIXED_CAPACITY_MWH = {"N1": 0.4, "N2": 0.8, "N3": 1.2, "N4": 0.8, "N5": 0.4}
ROUTE = ["N1", "N2", "N3", "N4"]


@dataclass(frozen=True)
class BoundaryScanConfig:
    hours_per_window: int = 3
    windows: int = 4
    base_load_mw: float = 1.0
    peak_extra_mw: float = 1.0
    grid_margin_frac: float = 0.03
    unserved_penalty_yuan_per_mwh: float = 12.0
    c_reconf_yuan_per_mwh: float = 2.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_initial_frac: float = 0.5
    soc_min_frac: float = 0.1
    soc_max_frac: float = 0.9
    charge_c_rate: float = 1.0
    discharge_c_rate: float = 1.0

    @property
    def mobile_total_mwh(self) -> float:
        return float(round(sum(FIXED_CAPACITY_MWH.values()) * 0.7))


def _build_load_and_price_maps(
    config: BoundaryScanConfig,
    peak_residence_windows: int,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float], dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    hours = list(range(config.windows * config.hours_per_window))
    hour_to_window = {t: t // config.hours_per_window for t in hours}

    load_mw: dict[tuple[str, int], float] = {}
    price: dict[tuple[str, int], float] = {}
    grid_limit: dict[tuple[str, int], float] = {}
    penalty: dict[tuple[str, int], float] = {}

    # Flat energy price isolates shortage-relief and relocation value instead of
    # letting time-of-use arbitrage dominate the comparison.
    pulse = [0.0, config.peak_extra_mw, 0.0]
    grid_cap = config.base_load_mw * (1.0 + config.grid_margin_frac)

    for t in hours:
        w = hour_to_window[t]
        hod = t % config.hours_per_window
        hotspot_idx = min(len(ROUTE) - 1, w // peak_residence_windows)
        hotspot_site = ROUTE[hotspot_idx]

        for site in SITES:
            load_mw[(site, t)] = config.base_load_mw + (pulse[hod] if site == hotspot_site else 0.0)
            price[(site, t)] = 1.0
            grid_limit[(site, t)] = grid_cap
            penalty[(site, t)] = config.unserved_penalty_yuan_per_mwh

    return load_mw, price, grid_limit, penalty


def _build_active_init(
    mobile_total_mwh: float,
    windows: list[int],
    delay_windows: int,
) -> dict[tuple[str, int], float]:
    active_init: dict[tuple[str, int], float] = {}
    if delay_windows <= 0:
        return active_init

    share = mobile_total_mwh / len(SITES)
    for w in windows:
        if w < delay_windows:
            for site in SITES:
                active_init[(site, w)] = share
    return active_init


def _solve_scheme(
    config: BoundaryScanConfig,
    *,
    peak_residence_windows: int,
    scheme: str,
    delay_windows: int = 0,
    solver: str = "highs",
) -> dict[str, float]:
    hours = list(range(config.windows * config.hours_per_window))
    windows = list(range(config.windows))
    hour_to_window = {t: t // config.hours_per_window for t in hours}
    load_mw, price, grid_limit, penalty = _build_load_and_price_maps(config, peak_residence_windows)

    soc_initial = {site: config.soc_initial_frac for site in SITES}
    soc_min = {site: config.soc_min_frac for site in SITES}
    soc_max = {site: config.soc_max_frac for site in SITES}
    charge_rate = {site: config.charge_c_rate for site in SITES}
    discharge_rate = {site: config.discharge_c_rate for site in SITES}

    if scheme == "base":
        m_total_mwh = 0.0
        eff_delay = 0
        active_init = {}
        reconf_limit = {w: 0.0 for w in windows}
        c_reconf = 0.0
    elif scheme == "fixed":
        m_total_mwh = config.mobile_total_mwh
        eff_delay = 0
        active_init = {}
        reconf_limit = {0: 2.0 * m_total_mwh, **{w: 0.0 for w in windows if w > 0}}
        c_reconf = 0.0
    elif scheme == "mess":
        m_total_mwh = config.mobile_total_mwh
        eff_delay = delay_windows
        active_init = _build_active_init(m_total_mwh, windows, eff_delay)
        reconf_limit = {w: 2.0 * m_total_mwh for w in windows}
        c_reconf = config.c_reconf_yuan_per_mwh
    else:
        raise ValueError(f"unsupported scheme: {scheme}")

    data = CoreModelData(
        sites=SITES,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load_mw,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh=FIXED_CAPACITY_MWH,
        soc_initial_frac=soc_initial,
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
        charge_c_rate=charge_rate,
        discharge_c_rate=discharge_rate,
        eta_charge=config.eta_charge,
        eta_discharge=config.eta_discharge,
        dt_hours=1.0,
        m_total_mwh=m_total_mwh,
        delay_windows=eff_delay,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=c_reconf,
        c_storage_fixed_yuan_per_mwh=0.0,
        c_storage_mobile_yuan_per_mwh=0.0,
    )
    model = build_core_model(data)
    solve_core_model(model, solver_name=solver)
    solution = extract_solution(model)
    return solution["costs"]


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan peak-duration / delay boundary for MESS vs fixed")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--peak-residence-windows", default="1,2,3,4")
    parser.add_argument("--delay-windows", default="0,1,2,3,4")
    parser.add_argument(
        "--output-csv",
        default="sim/outputs/peak_duration_delay_boundary_scan.csv",
    )
    args = parser.parse_args()

    config = BoundaryScanConfig()
    peak_residence_windows = [int(x) for x in args.peak_residence_windows.split(",") if x.strip()]
    delay_windows = [int(x) for x in args.delay_windows.split(",") if x.strip()]

    rows: list[dict[str, float | int | str]] = []
    print("boundary scan assumptions:")
    print(f"- hours_per_window: {config.hours_per_window}")
    print(f"- windows: {config.windows}")
    print(f"- fixed_total_mwh: {sum(FIXED_CAPACITY_MWH.values())}")
    print(f"- mobile_total_mwh: {config.mobile_total_mwh}")
    print(f"- peak_extra_mw: {config.peak_extra_mw}")
    print(f"- grid_margin_frac: {config.grid_margin_frac}")
    print(f"- c_reconf_yuan_per_mwh: {config.c_reconf_yuan_per_mwh}")
    print(f"- solver: {args.solver}")
    print()

    for residence in peak_residence_windows:
        base_cost = _solve_scheme(
            config,
            peak_residence_windows=residence,
            scheme="base",
            solver=args.solver,
        )["C_total"]
        fixed_cost = _solve_scheme(
            config,
            peak_residence_windows=residence,
            scheme="fixed",
            solver=args.solver,
        )["C_total"]

        print(
            f"peak_duration={residence * config.hours_per_window}h:"
            f" base_cost={base_cost:.3f}, fixed_cost={fixed_cost:.3f}"
        )
        for delay in delay_windows:
            mess_cost = _solve_scheme(
                config,
                peak_residence_windows=residence,
                scheme="mess",
                delay_windows=delay,
                solver=args.solver,
            )["C_total"]
            row = {
                "peak_residence_windows": residence,
                "peak_duration_hours": residence * config.hours_per_window,
                "delay_windows": delay,
                "travel_time_hours": delay * config.hours_per_window,
                "base_cost": round(base_cost, 6),
                "fixed_cost": round(fixed_cost, 6),
                "mess_cost": round(mess_cost, 6),
                "fixed_value_vs_base": round(base_cost - fixed_cost, 6),
                "mess_value_vs_base": round(base_cost - mess_cost, 6),
                "mess_advantage_vs_fixed": round(fixed_cost - mess_cost, 6),
                "winner": "mess" if mess_cost < fixed_cost - 1e-6 else "fixed_or_tied",
            }
            rows.append(row)
            print(
                f"  delay={row['travel_time_hours']}h ->"
                f" mess_cost={mess_cost:.3f},"
                f" mess_value={row['mess_value_vs_base']:.3f},"
                f" mess_adv_vs_fixed={row['mess_advantage_vs_fixed']:.3f},"
                f" winner={row['winner']}"
            )
        print()

    output_path = Path(args.output_csv).resolve()
    _write_rows(output_path, rows)

    print("first fixed-dominant delay by peak duration:")
    for residence in peak_residence_windows:
        subset = [r for r in rows if r["peak_residence_windows"] == residence]
        fixed_first = next((r for r in subset if r["mess_advantage_vs_fixed"] <= 0.0), None)
        if fixed_first is None:
            print(f"- peak_duration={residence * config.hours_per_window}h -> none")
        else:
            print(
                f"- peak_duration={residence * config.hours_per_window}h ->"
                f" fixed becomes non-worse at travel_time={fixed_first['travel_time_hours']}h"
            )

    print(f"\nresults_csv: {output_path}")


if __name__ == "__main__":
    main()
