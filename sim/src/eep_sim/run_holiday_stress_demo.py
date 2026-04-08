"""Solve independent holiday stress-window cases from expanded_8760 data."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data


HOLIDAY_RANGES_2026 = {
    "spring_festival": (date(2026, 2, 15), date(2026, 2, 23)),
    "national_day": (date(2026, 10, 1), date(2026, 10, 7)),
}


def _hour_index_for_datetime(year: int, month: int, day: int, hod: int) -> int:
    base = date(year, 1, 1)
    cur = date(year, month, day)
    return ((cur - base).days * 24) + hod


def _daterange(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _active_init_for_subset(
    sites: list[str],
    windows: list[int],
    m_total_mwh: float,
    delay_windows: int,
    initial_active_mode: str,
) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    if delay_windows <= 0:
        return out
    if initial_active_mode == "equal":
        share = m_total_mwh / max(1, len(sites))
    else:
        share = 0.0
    for w in windows:
        if w < delay_windows:
            for site in sites:
                out[(site, w)] = share
    return out


def _reconf_limit_for_subset(
    windows: list[int],
    reconf_limit_mwh: float | dict[int, float],
) -> dict[int, float]:
    if isinstance(reconf_limit_mwh, dict):
        return {w: float(reconf_limit_mwh.get(w, 1e9)) for w in windows}
    return {w: float(reconf_limit_mwh) for w in windows}


def _subset_data_for_dates(
    data: CoreModelData,
    *,
    year: int,
    start: date,
    end: date,
    reconf_hours: int,
    dt_hours: float,
    delay_windows: int,
    initial_active_mode: str,
    reconf_limit_mwh: float | dict[int, float],
) -> CoreModelData:
    selected_hours: list[int] = []
    for day in _daterange(start, end):
        for hod in range(24):
            selected_hours.append(_hour_index_for_datetime(year, day.month, day.day, hod))

    hour_map = {old_t: new_t for new_t, old_t in enumerate(selected_hours)}
    hours = list(range(len(selected_hours)))
    window_size_steps = int(round(reconf_hours / dt_hours))
    windows = list(range(math.ceil(len(hours) / window_size_steps)))
    hour_to_window = {t: t // window_size_steps for t in hours}

    load_mw = {}
    price = {}
    grid_limit = {}
    unserved_penalty = {}
    for site in data.sites:
        for old_t, new_t in hour_map.items():
            load_mw[(site, new_t)] = float(data.load_mw[(site, old_t)])
            price[(site, new_t)] = float(data.price_yuan_per_mwh[(site, old_t)])
            grid_limit[(site, new_t)] = float(data.grid_limit_mw[(site, old_t)])
            unserved_penalty[(site, new_t)] = float(data.unserved_penalty_yuan_per_mwh[(site, old_t)])

    active_init = _active_init_for_subset(
        list(data.sites),
        windows,
        data.m_total_mwh,
        delay_windows,
        initial_active_mode,
    )
    reconf_limit = _reconf_limit_for_subset(windows, reconf_limit_mwh)

    return CoreModelData(
        sites=list(data.sites),
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load_mw,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=unserved_penalty,
        fixed_capacity_mwh=dict(data.fixed_capacity_mwh),
        soc_initial_frac=dict(data.soc_initial_frac),
        soc_min_frac=dict(data.soc_min_frac),
        soc_max_frac=dict(data.soc_max_frac),
        charge_c_rate=dict(data.charge_c_rate),
        discharge_c_rate=dict(data.discharge_c_rate),
        eta_charge=data.eta_charge,
        eta_discharge=data.eta_discharge,
        dt_hours=data.dt_hours,
        m_total_mwh=data.m_total_mwh,
        delay_windows=delay_windows,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=data.c_reconf_yuan_per_mwh,
        c_storage_fixed_yuan_per_mwh=data.c_storage_fixed_yuan_per_mwh,
        c_storage_mobile_yuan_per_mwh=data.c_storage_mobile_yuan_per_mwh,
    )


def _print_case_summary(case_name: str, data: CoreModelData, solution: dict) -> None:
    total_unserved = sum(solution["unserved"].values())
    reconf_total = sum(
        abs(solution["x"][(site, w)] - solution["x"][(site, w - 1)])
        for site in data.sites
        for w in data.windows
        if w > 0
    )
    print(f"{case_name}:")
    print(f"- hours: {len(data.hours)}")
    print(f"- windows: {len(data.windows)}")
    print(f"- costs: {solution['costs']}")
    print(f"- total_unserved_mwh_like: {total_unserved}")
    print(f"- reconf_total_mwh: {reconf_total}")

    site_unserved = []
    for site in data.sites:
        value = sum(solution["unserved"][(site, t)] for t in data.hours)
        site_unserved.append((value, site))
    site_unserved.sort(reverse=True)
    print(f"- top5_unserved_sites: {site_unserved[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Spring Festival / National Day stress windows")
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw and result folders")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--mobile-capacity-ratio-to-fixed", type=float, default=0.7)
    parser.add_argument("--grid-capacity-margin-above-avg-load", type=float, default=0.25)
    parser.add_argument("--opportunity-cost-price-multiplier", type=float, default=1.0)
    parser.add_argument("--holiday-year", type=int, default=2026)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--cases",
        default="spring_festival,national_day",
        help="Comma-separated holiday cases from {spring_festival,national_day}",
    )
    args = parser.parse_args()

    options = G2LoaderOptions(
        reconf_hours=args.reconf_hours,
        delay_windows=args.delay_windows,
        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
        timeline_mode="expanded_8760",
        holiday_year=args.holiday_year,
        apply_cn_holiday_shocks=True,
        grid_capacity_mode="max_of_csv_and_avg_load_margin",
        grid_capacity_margin_above_avg_load=args.grid_capacity_margin_above_avg_load,
        default_unserved_penalty_yuan_per_mwh=1000.0,
        symbolic_unserved_penalty_mode="fixed",
    )

    full_data, diag = load_g2_core_model_data(args.base_dir, options=options)
    print("base expanded_8760 summary:")
    print(f"- sites: {diag.sites}")
    print(f"- fixed_total_mwh: {diag.fixed_total_mwh}")
    print(f"- mobile_total_mwh: {diag.mobile_total_mwh}")

    case_names = [name.strip() for name in args.cases.split(",") if name.strip()]
    for case_name in case_names:
        if case_name not in HOLIDAY_RANGES_2026:
            raise ValueError(f"unsupported holiday case: {case_name}")
        start, end = HOLIDAY_RANGES_2026[case_name]
        subset_data = _subset_data_for_dates(
            full_data,
            year=args.holiday_year,
            start=start,
            end=end,
            reconf_hours=args.reconf_hours,
            dt_hours=options.dt_hours,
            delay_windows=args.delay_windows,
            initial_active_mode=options.initial_active_mode,
            reconf_limit_mwh=options.reconf_limit_mwh,
        )
        model = build_core_model(subset_data)
        result = solve_core_model(model, solver_name=args.solver)
        solution = extract_solution(model)
        print(f"- {case_name} solver_status: {result.solver.status}")
        print(f"- {case_name} termination: {result.solver.termination_condition}")
        _print_case_summary(case_name, subset_data, solution)


if __name__ == "__main__":
    main()
