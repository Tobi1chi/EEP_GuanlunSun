"""Synthetic corridor scan to identify when mobile storage starts reconfiguring."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model


@dataclass(frozen=True)
class SyntheticScenario:
    hotspot_extra_mw: float
    grid_margin_frac: float
    delay_windows: int
    c_reconf_yuan_per_mwh: float
    unserved_penalty_multiplier: float


def _daily_shape(hour: int) -> float:
    if 0 <= hour <= 5:
        return 0.75
    if 6 <= hour <= 8:
        return 0.9
    if 9 <= hour <= 11:
        return 1.0
    if 12 <= hour <= 16:
        return 1.05
    if 17 <= hour <= 20:
        return 0.95
    return 0.8


def build_synthetic_corridor_data(scenario: SyntheticScenario) -> CoreModelData:
    sites = ["N1", "N2", "N3", "N4", "N5"]
    hours_per_window = 24
    windows = list(range(12))
    hours = list(range(len(windows) * hours_per_window))
    hour_to_window = {t: t // hours_per_window for t in hours}

    # Keep non-hotspot days close to self-sufficient so that MESS value comes
    # from chasing the rotating hotspot instead of backfilling a permanent
    # structural shortage at every site.
    base_mean_by_site = {site: 1.0 for site in sites}
    base_peak_mw = 1.05
    fixed_capacity = {site: 0.25 for site in sites}
    soc_initial = {site: 0.5 for site in sites}
    soc_min = {site: 0.1 for site in sites}
    soc_max = {site: 0.9 for site in sites}
    charge_rate = {site: 1.0 for site in sites}
    discharge_rate = {site: 1.0 for site in sites}
    m_total_mwh = 2.0
    active_init = {(site, 0): m_total_mwh / len(sites) for site in sites} if scenario.delay_windows > 0 else {}
    reconf_limit = {w: 2 * m_total_mwh for w in windows}

    load = {}
    price = {}
    grid_limit = {}
    penalty = {}

    # Hotspot rotates across sites by window.
    hotspot_order = ["N1", "N2", "N3", "N4", "N5", "N4", "N3", "N2", "N1", "N2", "N3", "N4"]
    valley_price = 0.55
    shoulder_price = 0.9
    peak_price = 1.45

    for t in hours:
        w = hour_to_window[t]
        hod = t % hours_per_window
        hotspot_site = hotspot_order[w]
        for site in sites:
            base_load = base_mean_by_site[site] * _daily_shape(hod)
            hotspot = 0.0
            if site == hotspot_site and 11 <= hod <= 18:
                # Localized holiday surge: same corridor total demand, but the
                # congested node shifts from window to window.
                if 12 <= hod <= 16:
                    hotspot = scenario.hotspot_extra_mw
                else:
                    hotspot = 0.7 * scenario.hotspot_extra_mw

            load[(site, t)] = base_load + hotspot

            if 0 <= hod <= 6:
                price[(site, t)] = valley_price
            elif 7 <= hod <= 10 or 21 <= hod <= 23:
                price[(site, t)] = shoulder_price
            else:
                price[(site, t)] = peak_price

            grid_limit[(site, t)] = base_peak_mw * (1.0 + scenario.grid_margin_frac)
            penalty[(site, t)] = price[(site, t)] * scenario.unserved_penalty_multiplier

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
        m_total_mwh=m_total_mwh,
        delay_windows=scenario.delay_windows,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=scenario.c_reconf_yuan_per_mwh,
        c_storage_fixed_yuan_per_mwh=0.0,
        c_storage_mobile_yuan_per_mwh=0.0,
    )


def solve_scenario(scenario: SyntheticScenario, solver: str) -> tuple[dict, float]:
    data = build_synthetic_corridor_data(scenario)
    model = build_core_model(data)
    result = solve_core_model(model, solver_name=solver)
    solution = extract_solution(model)
    reconf_total = sum(
        abs(solution["x"][(site, w)] - solution["x"][(site, w - 1)])
        for site in data.sites
        for w in data.windows
        if w > 0
    )
    return {
        "status": str(result.solver.status),
        "termination": str(result.solver.termination_condition),
        "costs": solution["costs"],
        "x": solution["x"],
        "unserved_total": sum(solution["unserved"].values()),
        "sites": data.sites,
        "windows": data.windows,
    }, reconf_total


def _site_ranges(solution: dict) -> list[tuple[float, str, list[float]]]:
    out = []
    for site in solution["sites"]:
        xs = [solution["x"][(site, w)] for w in solution["windows"]]
        out.append((max(xs) - min(xs), site, xs))
    out.sort(reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic scan for reconfiguration onset")
    parser.add_argument("--solver", default="highs")
    args = parser.parse_args()

    hotspot_extras = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    grid_margins = [0.20, 0.30]
    delays = [0, 1]
    reconf_costs = [0.0, 20.0]
    penalty_multipliers = [1.0, 4.0, 8.0]
    material_reconf_threshold_mwh = 10.0

    scan_rows: list[tuple[SyntheticScenario, float, float]] = []
    first_trigger: tuple[SyntheticScenario, float, dict] | None = None
    strongest: tuple[SyntheticScenario, float, dict] | None = None

    for penalty_multiplier in penalty_multipliers:
        for grid_margin in grid_margins:
            for delay in delays:
                for reconf_cost in reconf_costs:
                    for hotspot_extra in hotspot_extras:
                        scenario = SyntheticScenario(
                            hotspot_extra_mw=hotspot_extra,
                            grid_margin_frac=grid_margin,
                            delay_windows=delay,
                            c_reconf_yuan_per_mwh=reconf_cost,
                            unserved_penalty_multiplier=penalty_multiplier,
                        )
                        solution, reconf_total = solve_scenario(scenario, args.solver)
                        scan_rows.append((scenario, reconf_total, solution["unserved_total"]))

                        if reconf_total >= material_reconf_threshold_mwh and first_trigger is None:
                            first_trigger = (scenario, reconf_total, solution)
                        if reconf_total >= material_reconf_threshold_mwh:
                            if strongest is None or reconf_total > strongest[1]:
                                strongest = (scenario, reconf_total, solution)

    print("synthetic scan summary:")
    print(f"- cases_tested: {len(scan_rows)}")
    positive = [row for row in scan_rows if row[1] >= material_reconf_threshold_mwh]
    print(f"- material_reconfiguration_threshold_mwh: {material_reconf_threshold_mwh}")
    print(f"- cases_with_material_reconfiguration: {len(positive)}")

    if first_trigger is None:
        print("- first_trigger: none")
    else:
        scenario, reconf_total, solution = first_trigger
        print("- first_trigger:")
        print(
            f"  hotspot_extra_mw={scenario.hotspot_extra_mw},"
            f" grid_margin_frac={scenario.grid_margin_frac},"
            f" delay_windows={scenario.delay_windows},"
            f" c_reconf={scenario.c_reconf_yuan_per_mwh},"
            f" penalty_multiplier={scenario.unserved_penalty_multiplier}"
        )
        print(f"  reconf_total_mwh={reconf_total}")
        print(f"  total_unserved={solution['unserved_total']}")

    if strongest is not None:
        scenario, reconf_total, solution = strongest
        print("- strongest_reconfiguration_case:")
        print(
            f"  hotspot_extra_mw={scenario.hotspot_extra_mw},"
            f" grid_margin_frac={scenario.grid_margin_frac},"
            f" delay_windows={scenario.delay_windows},"
            f" c_reconf={scenario.c_reconf_yuan_per_mwh},"
            f" penalty_multiplier={scenario.unserved_penalty_multiplier}"
        )
        print(f"  reconf_total_mwh={reconf_total}")
        print(f"  total_unserved={solution['unserved_total']}")
        moved = _site_ranges(solution)
        print(
            "  top3_moved_sites="
            + str([(site, round(rng, 3), [round(v, 3) for v in xs]) for rng, site, xs in moved[:3]])
        )

    # Give a compact threshold table by (grid_margin, delay, reconf_cost, penalty_multiplier).
    print("- threshold_by_setting:")
    grouped_keys = []
    for penalty_multiplier in penalty_multipliers:
        for grid_margin in grid_margins:
            for delay in delays:
                for reconf_cost in reconf_costs:
                    grouped_keys.append((penalty_multiplier, grid_margin, delay, reconf_cost))
    for penalty_multiplier, grid_margin, delay, reconf_cost in grouped_keys:
        candidates = [
            row
            for row in scan_rows
            if math.isclose(row[0].unserved_penalty_multiplier, penalty_multiplier)
            and math.isclose(row[0].grid_margin_frac, grid_margin)
            and row[0].delay_windows == delay
            and math.isclose(row[0].c_reconf_yuan_per_mwh, reconf_cost)
            and row[1] >= material_reconf_threshold_mwh
        ]
        if candidates:
            threshold = min(candidates, key=lambda row: row[0].hotspot_extra_mw)[0].hotspot_extra_mw
            print(
                f"  penalty={penalty_multiplier}, grid_margin={grid_margin},"
                f" delay={delay}, reconf_cost={reconf_cost} -> hotspot_extra_threshold={threshold}"
            )
        else:
            print(
                f"  penalty={penalty_multiplier}, grid_margin={grid_margin},"
                f" delay={delay}, reconf_cost={reconf_cost} -> no_reconfiguration"
            )


if __name__ == "__main__":
    main()
