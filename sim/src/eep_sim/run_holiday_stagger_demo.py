"""Solve holiday stress windows with synthetic inter-site peak staggering."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data, resolve_g2_input_path


HOLIDAY_RANGES_2026 = {
    "spring_festival": (date(2026, 2, 15), date(2026, 2, 23)),
    "national_day": (date(2026, 10, 1), date(2026, 10, 7)),
}

ARCHETYPE_SCALE = {"A": 0.95, "B": 1.05, "C": 1.15}
ARCHETYPE_WIDTH = {"A": 2.5, "B": 3.5, "C": 4.5}


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
    share = m_total_mwh / max(1, len(sites)) if initial_active_mode == "equal" else 0.0
    for w in windows:
        if w < delay_windows:
            for site in sites:
                out[(site, w)] = share
    return out


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

    if isinstance(reconf_limit_mwh, dict):
        reconf_limit = {w: float(reconf_limit_mwh.get(w, 1e9)) for w in windows}
    else:
        reconf_limit = {w: float(reconf_limit_mwh) for w in windows}

    active_init = _active_init_for_subset(
        list(data.sites), windows, data.m_total_mwh, delay_windows, initial_active_mode
    )

    def _slice_map(source: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
        out: dict[tuple[str, int], float] = {}
        for site in data.sites:
            for old_t, new_t in hour_map.items():
                out[(site, new_t)] = float(source[(site, old_t)])
        return out

    return CoreModelData(
        sites=list(data.sites),
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=_slice_map(dict(data.load_mw)),
        price_yuan_per_mwh=_slice_map(dict(data.price_yuan_per_mwh)),
        grid_limit_mw=_slice_map(dict(data.grid_limit_mw)),
        unserved_penalty_yuan_per_mwh=_slice_map(dict(data.unserved_penalty_yuan_per_mwh)),
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


def _triangle(hour: int, center: float, half_width: float) -> float:
    distance = abs(hour - center)
    if distance >= half_width:
        return 0.0
    return 1.0 - distance / half_width


def _phase_for_day(day_idx: int, total_days: int) -> str:
    ratio = (day_idx + 0.5) / total_days
    if ratio <= 0.34:
        return "outbound"
    if ratio <= 0.67:
        return "mid_holiday"
    return "return"


def _apply_staggered_holiday_profile(
    data: CoreModelData,
    *,
    holiday_name: str,
    corridor_order: list[str],
    archetype_by_service: dict[str, str],
    stagger_scale: float = 1.0,
) -> CoreModelData:
    position_by_service = {
        service: idx / max(1, len(corridor_order) - 1) for idx, service in enumerate(corridor_order)
    }
    total_days = len(data.hours) // 24
    load_mw = dict(data.load_mw)

    for day_idx in range(total_days):
        phase = _phase_for_day(day_idx, total_days)
        for service in data.sites:
            pos = position_by_service.get(service, 0.5)
            archetype = archetype_by_service.get(service, "")
            size_scale = ARCHETYPE_SCALE.get(archetype, 1.0)
            base_width = ARCHETYPE_WIDTH.get(archetype, 3.0)

            if phase == "outbound":
                amplitude = (0.45 + 0.35 * (1.0 - pos)) * size_scale
                center = 11.0 + 5.0 * pos
                half_width = base_width
                phase_offset = 0.45 * (1.0 - 2.0 * pos)
            elif phase == "mid_holiday":
                amplitude = (0.20 + 0.20 * (1.0 - abs(pos - 0.5) * 2.0)) * size_scale
                center = 14.0
                half_width = base_width + 1.5
                phase_offset = 0.30 * (1.0 - abs(pos - 0.5) * 4.0)
            else:
                amplitude = (0.45 + 0.35 * pos) * size_scale
                center = 11.0 + 5.0 * (1.0 - pos)
                half_width = base_width
                phase_offset = 0.45 * (2.0 * pos - 1.0)

            if holiday_name == "spring_festival":
                amplitude *= 1.05
            else:
                amplitude *= 1.00

            for hod in range(24):
                t = day_idx * 24 + hod
                shape = _triangle(hod, center=center, half_width=half_width)
                shoulder = 0.08 * size_scale if 9 <= hod <= 21 else 0.0
                surge_signal = max(0.0, phase_offset + shoulder + amplitude * shape) * stagger_scale
                multiplier = 0.60 + surge_signal
                load_mw[(service, t)] *= multiplier

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


def _site_x_range(solution: dict, sites: list[str], windows: list[int]) -> list[tuple[float, str, list[float]]]:
    out = []
    for site in sites:
        xs = [solution["x"][(site, w)] for w in windows]
        out.append((max(xs) - min(xs), site, xs))
    out.sort(reverse=True)
    return out


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
    x_ranges = _site_x_range(solution, list(data.sites), list(data.windows))
    print(
        "- top5_moved_sites:",
        [(site, round(rng, 4), [round(v, 3) for v in xs]) for rng, site, xs in x_ranges[:5]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve holiday windows with inter-site peak staggering")
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw and result folders")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--mobile-capacity-ratio-to-fixed", type=float, default=0.7)
    parser.add_argument("--grid-capacity-margin-above-avg-load", type=float, default=0.25)
    parser.add_argument("--opportunity-cost-price-multiplier", type=float, default=1.0)
    parser.add_argument("--holiday-year", type=int, default=2026)
    parser.add_argument("--stagger-scale", type=float, default=1.0)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--cases",
        default="spring_festival,national_day",
        help="Comma-separated holiday cases from {spring_festival,national_day}",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    corridor_order, archetype_by_service = _load_corridor_order_and_archetype(base_dir)

    options = G2LoaderOptions(
        reconf_hours=args.reconf_hours,
        delay_windows=args.delay_windows,
        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
        timeline_mode="expanded_8760",
        holiday_year=args.holiday_year,
        apply_cn_holiday_shocks=False,
        grid_capacity_mode="max_of_csv_and_avg_load_margin",
        grid_capacity_margin_above_avg_load=args.grid_capacity_margin_above_avg_load,
        default_unserved_penalty_yuan_per_mwh=1000.0,
        symbolic_unserved_penalty_mode="fixed",
    )

    full_data, diag = load_g2_core_model_data(base_dir, options=options)
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
        staggered_data = _apply_staggered_holiday_profile(
            subset_data,
            holiday_name=case_name,
            corridor_order=corridor_order,
            archetype_by_service=archetype_by_service,
            stagger_scale=args.stagger_scale,
        )
        model = build_core_model(staggered_data)
        result = solve_core_model(model, solver_name=args.solver)
        solution = extract_solution(model)
        print(f"- {case_name} solver_status: {result.solver.status}")
        print(f"- {case_name} termination: {result.solver.termination_condition}")
        _print_case_summary(case_name, staggered_data, solution)


if __name__ == "__main__":
    main()
