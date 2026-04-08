"""Solve representative 12x24 with site-specific monthly demand waves."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data, resolve_g2_input_path


ARCHETYPE_SCALE = {"A": 0.9, "B": 1.0, "C": 1.1}


def _load_corridor_order_and_archetype(base_dir: Path) -> tuple[list[str], dict[str, str]]:
    rows = list(
        csv.DictReader(
            open(
                resolve_g2_input_path(base_dir, "g2_beijing_shanghai_service_areas_load_grid_params_revised.csv"),
                encoding="utf-8-sig",
            )
        )
    )
    order: list[str] = []
    archetype_by_service: dict[str, str] = {}
    for row in rows:
        service = row["Service Area"].strip()
        if service not in order:
            order.append(service)
        archetype_by_service[service] = (row.get("archetype", "") or "").strip()
    return order, archetype_by_service


def _gaussian(month: int, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((month - center) / sigma) ** 2)


def _site_month_multiplier(month: int, position: float, archetype: str) -> float:
    north_weight = 1.0 - position
    south_weight = position
    mid_weight = 1.0 - abs(2.0 * position - 1.0)
    size_scale = ARCHETYPE_SCALE.get(archetype, 1.0)

    raw = (
        1.0
        + 0.45 * north_weight * _gaussian(month, 2.0, 1.2)
        + 0.35 * south_weight * _gaussian(month, 8.0, 1.4)
        + 0.25 * mid_weight * _gaussian(month, 5.0, 1.6)
        + 0.30 * (0.5 + 0.5 * mid_weight) * _gaussian(month, 10.0, 1.0)
    )
    centered = 0.72 + (raw - 1.0) * size_scale
    return centered


def _normalize_site_multipliers(
    corridor_order: list[str],
    archetype_by_service: dict[str, str],
) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for idx, service in enumerate(corridor_order):
        position = idx / max(1, len(corridor_order) - 1)
        archetype = archetype_by_service.get(service, "")
        raw = {month: _site_month_multiplier(month, position, archetype) for month in range(1, 13)}
        avg = sum(raw.values()) / 12.0
        for month, value in raw.items():
            normalized = value / avg
            out[(service, month)] = min(1.55, max(0.65, normalized))
    return out


def _apply_site_month_multipliers(
    data: CoreModelData,
    site_month_multiplier: dict[tuple[str, int], float],
) -> CoreModelData:
    load_mw = dict(data.load_mw)
    for site in data.sites:
        for t in data.hours:
            month = int(data.hour_to_window[t]) + 1
            load_mw[(site, t)] *= site_month_multiplier[(site, month)]

    return CoreModelData(
        sites=data.sites,
        hours=data.hours,
        windows=data.windows,
        hour_to_window=data.hour_to_window,
        load_mw=load_mw,
        price_yuan_per_mwh=data.price_yuan_per_mwh,
        grid_limit_mw=data.grid_limit_mw,
        unserved_penalty_yuan_per_mwh=data.unserved_penalty_yuan_per_mwh,
        fixed_capacity_mwh=data.fixed_capacity_mwh,
        soc_initial_frac=data.soc_initial_frac,
        soc_min_frac=data.soc_min_frac,
        soc_max_frac=data.soc_max_frac,
        charge_c_rate=data.charge_c_rate,
        discharge_c_rate=data.discharge_c_rate,
        eta_charge=data.eta_charge,
        eta_discharge=data.eta_discharge,
        dt_hours=data.dt_hours,
        m_total_mwh=data.m_total_mwh,
        delay_windows=data.delay_windows,
        active_init_mwh=data.active_init_mwh,
        reconf_limit_mwh=data.reconf_limit_mwh,
        c_reconf_yuan_per_mwh=data.c_reconf_yuan_per_mwh,
        c_storage_fixed_yuan_per_mwh=data.c_storage_fixed_yuan_per_mwh,
        c_storage_mobile_yuan_per_mwh=data.c_storage_mobile_yuan_per_mwh,
    )


def _top_moved_sites(solution: dict, sites: list[str], windows: list[int]) -> list[tuple[float, str, list[float]]]:
    out = []
    for site in sites:
        xs = [solution["x"][(site, w)] for w in windows]
        out.append((max(xs) - min(xs), site, xs))
    out.sort(reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Site-specific monthly heterogeneity demo")
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw and result folders")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--mobile-capacity-ratio-to-fixed", type=float, default=0.7)
    parser.add_argument("--grid-capacity-margin-above-avg-load", type=float, default=0.25)
    parser.add_argument("--opportunity-cost-price-multiplier", type=float, default=1.0)
    parser.add_argument("--solver", default="highs")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    corridor_order, archetype_by_service = _load_corridor_order_and_archetype(base_dir)
    site_month_multiplier = _normalize_site_multipliers(corridor_order, archetype_by_service)

    options = G2LoaderOptions(
        reconf_hours=args.reconf_hours,
        delay_windows=args.delay_windows,
        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
        timeline_mode="representative_12x24",
        grid_capacity_mode="max_of_csv_and_avg_load_margin",
        grid_capacity_margin_above_avg_load=args.grid_capacity_margin_above_avg_load,
        default_unserved_penalty_yuan_per_mwh=1000.0,
        symbolic_unserved_penalty_mode="fixed",
    )
    data, diag = load_g2_core_model_data(base_dir, options=options)
    hetero_data = _apply_site_month_multipliers(data, site_month_multiplier)
    model = build_core_model(hetero_data)
    result = solve_core_model(model, solver_name=args.solver)
    solution = extract_solution(model)

    total_unserved = sum(solution["unserved"].values())
    reconf_total = sum(
        abs(solution["x"][(site, w)] - solution["x"][(site, w - 1)])
        for site in hetero_data.sites
        for w in hetero_data.windows
        if w > 0
    )

    print("site-month heterogeneity summary:")
    print(f"- fixed_total_mwh: {diag.fixed_total_mwh}")
    print(f"- mobile_total_mwh: {diag.mobile_total_mwh}")
    print(f"- solver_status: {result.solver.status}")
    print(f"- termination: {result.solver.termination_condition}")
    print(f"- costs: {solution['costs']}")
    print(f"- total_unserved_mwh_like: {total_unserved}")
    print(f"- reconf_total_mwh: {reconf_total}")

    moved = _top_moved_sites(solution, list(hetero_data.sites), list(hetero_data.windows))
    print(
        "- top5_moved_sites:",
        [(site, round(rng, 4), [round(v, 3) for v in xs]) for rng, site, xs in moved[:5]],
    )

    print("- sample_month_multipliers:")
    for site in corridor_order[:5]:
        vals = [round(site_month_multiplier[(site, month)], 3) for month in range(1, 13)]
        print(f"  - {site}: {vals}")


if __name__ == "__main__":
    main()
