"""Loader that maps g2_* CSV inputs into CoreModelData."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .core_model import CoreModelData


@dataclass(frozen=True)
class G2LoaderOptions:
    """Options for building CoreModelData from g2 files."""

    reconf_hours: int = 24
    delay_windows: int = 1
    # If None, derive MESS total from fixed-storage total * ratio and round
    # to the nearest integer MWh. If set explicitly, this manual value wins.
    m_total_mwh: float | None = None
    mobile_capacity_ratio_to_fixed: float = 0.7
    dt_hours: float = 1.0
    # timeline_mode:
    # - representative_12x24: 12 * 24 representative slices
    # - expanded_8760: expand month-hour slices to all days of a non-leap year
    timeline_mode: str = "representative_12x24"
    month_day_counts: Mapping[int, int] | None = None
    month_load_multiplier_by_month: Mapping[int, float] | None = None
    base_load_scale: float = 0.85
    enable_site_month_heterogeneity: bool = True
    site_month_heterogeneity_strength: float = 0.18
    enable_holiday_site_stagger: bool = True
    holiday_site_stagger_strength: float = 0.22

    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    # Objective coefficients
    c_reconf_yuan_per_mwh: float = 0.5
    c_storage_fixed_yuan_per_mwh: float = 0.0
    c_storage_mobile_yuan_per_mwh: float = 0.0
    # Default lost-profit penalty:
    # 1 RMB/kWh gross margin = 1000 RMB/MWh
    default_unserved_penalty_yuan_per_mwh: float = 1000.0
    symbolic_unserved_penalty_mode: str = "fixed"
    symbolic_unserved_penalty_price_multiplier: float = 20.0
    opportunity_cost_price_multiplier: float = 1.0

    # Price duplicate resolution policy in (service, month, hour):
    # "max", "min", "mean", "first"
    price_duplicate_policy: str = "max"
    # Optional service-specific category override; if provided and matched,
    # rows of this category are preferred before applying duplicate policy.
    preferred_category_by_service: Mapping[str, str] | None = None

    # Fixed storage defaults
    fixed_capacity_by_archetype_mwh: Mapping[str, float] | None = None
    default_fixed_capacity_mwh: float = 0.5
    soc_initial_frac: float = 0.5
    soc_min_frac: float = 0.1
    soc_max_frac: float = 0.9
    charge_c_rate: float = 1.0
    discharge_c_rate: float = 1.0
    grid_capacity_mode: str = "max_of_csv_and_avg_load_margin"
    grid_capacity_margin_above_avg_load: float = 0.25

    # Reconfiguration hard limit
    # scalar -> apply to all windows; mapping -> window-specific override
    reconf_limit_mwh: float | Mapping[int, float] = 1e9

    # Active mobile capacity for early windows (< delay_windows)
    # "equal": equally split M_total across sites
    # "zero": all zero
    initial_active_mode: str = "equal"

    # Holiday shock settings. Applied only in expanded_8760 mode.
    holiday_year: int = 2026
    apply_cn_holiday_shocks: bool = True
    holiday_peak_hour_start: int = 9
    holiday_peak_hour_end: int = 21
    holiday_load_multiplier_by_name: Mapping[str, float] | None = None
    demand_model: str = "charger_utilization"
    regular_day_utilization: float = 0.30
    holiday_day_utilization: float = 1.00
    supercharger_power_mw: float = 0.5
    normal_charger_power_mw: float = 0.05
    charger_counts_by_archetype: Mapping[str, Tuple[int, int]] | None = None


@dataclass(frozen=True)
class G2LoadDiagnostics:
    sites: int
    months: int
    hours_of_day: int
    timeline_hours: int
    windows: int
    window_size_steps: int
    timeline_mode: str
    duplicate_price_keys: int
    duplicate_price_rows: int
    price_duplicate_policy: str
    fixed_total_mwh: float
    mobile_total_mwh: float


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_g2_input_path(base_dir: str | Path, filename: str) -> Path:
    base = Path(base_dir).resolve()
    candidates = [
        base / "data" / "raw" / filename,
        base / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _to_float(value: str, *, default: float | None = None) -> float:
    text = (value or "").strip()
    if text == "":
        if default is None:
            raise ValueError("empty numeric value")
        return default
    try:
        return float(text)
    except ValueError:
        if default is None:
            raise
        return default


def _resolve_duplicates(
    values: Sequence[float],
    policy: str,
) -> float:
    if len(values) == 0:
        raise ValueError("cannot resolve empty value list")
    if policy == "first":
        return values[0]
    if policy == "max":
        return max(values)
    if policy == "min":
        return min(values)
    if policy == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unsupported price_duplicate_policy: {policy}")


def _build_reconf_limit_map(
    windows: Sequence[int],
    reconf_limit: float | Mapping[int, float],
) -> Dict[int, float]:
    if isinstance(reconf_limit, Mapping):
        out = {int(w): float(reconf_limit.get(int(w), 1e9)) for w in windows}
    else:
        out = {int(w): float(reconf_limit) for w in windows}
    return out


def _build_fixed_capacity_map(
    service_to_archetype: Mapping[str, str],
    options: G2LoaderOptions,
) -> Dict[str, float]:
    by_arch = options.fixed_capacity_by_archetype_mwh or {"A": 0.4, "B": 0.8, "C": 1.2}
    out: Dict[str, float] = {}
    for service, archetype in service_to_archetype.items():
        out[service] = float(by_arch.get(archetype, options.default_fixed_capacity_mwh))
    return out


def _build_active_init(
    sites: Sequence[str],
    windows: Sequence[int],
    mobile_total_mwh: float,
    options: G2LoaderOptions,
) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    if options.delay_windows <= 0:
        return out
    if options.initial_active_mode not in {"equal", "zero"}:
        raise ValueError("initial_active_mode must be 'equal' or 'zero'")

    if options.initial_active_mode == "equal":
        share = mobile_total_mwh / max(1, len(sites))
    else:
        share = 0.0

    for w in windows:
        if w < options.delay_windows:
            for s in sites:
                out[(s, int(w))] = share
    return out


def _parse_penalty(text: str, default_value: float) -> float:
    # Allow direct numeric; fall back to default for symbolic notes like "c_ls >> c_grid"
    stripped = (text or "").strip()
    if stripped == "":
        return default_value
    try:
        return float(stripped)
    except ValueError:
        return default_value


def _iter_date_range(start: date, end: date) -> List[date]:
    out: List[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _build_cn_holiday_day_map(year: int) -> Dict[Tuple[int, int], str]:
    ranges_by_year: Dict[int, List[Tuple[str, date, date]]] = {
        2026: [
            ("new_year", date(2026, 1, 1), date(2026, 1, 3)),
            ("spring_festival", date(2026, 2, 15), date(2026, 2, 23)),
            ("qingming", date(2026, 4, 4), date(2026, 4, 6)),
            ("labor_day", date(2026, 5, 1), date(2026, 5, 5)),
            ("dragon_boat", date(2026, 6, 19), date(2026, 6, 21)),
            ("mid_autumn", date(2026, 9, 25), date(2026, 9, 27)),
            ("national_day", date(2026, 10, 1), date(2026, 10, 7)),
        ]
    }
    ranges = ranges_by_year.get(year)
    if ranges is None:
        raise ValueError(f"unsupported holiday_year: {year}")

    out: Dict[Tuple[int, int], str] = {}
    for holiday_name, start, end in ranges:
        for day in _iter_date_range(start, end):
            out[(day.month, day.day)] = holiday_name
    return out


def _default_holiday_load_multiplier_by_name() -> Dict[str, float]:
    return {
        "new_year": 1.03,
        "spring_festival": 1.18,
        "qingming": 1.08,
        "labor_day": 1.12,
        "dragon_boat": 1.07,
        "mid_autumn": 1.08,
        "national_day": 1.18,
    }


def _charger_counts_by_archetype(
    options: G2LoaderOptions,
) -> Dict[str, Tuple[int, int]]:
    configured = options.charger_counts_by_archetype
    if configured is not None:
        return {str(k): (int(v[0]), int(v[1])) for k, v in configured.items()}
    return {
        "A": (1, 4),
        "B": (4, 8),
        "C": (6, 8),
    }


def _nominal_site_power_from_chargers(
    service_to_archetype: Mapping[str, str],
    options: G2LoaderOptions,
) -> Dict[str, float]:
    counts = _charger_counts_by_archetype(options)
    out: Dict[str, float] = {}
    for service, archetype in service_to_archetype.items():
        super_count, normal_count = counts.get(str(archetype), (2, 8))
        out[service] = (
            float(super_count) * float(options.supercharger_power_mw)
            + float(normal_count) * float(options.normal_charger_power_mw)
        )
    return out


def _charger_daily_shape_multiplier(hour: int) -> float:
    morning_peak = _triangle(hour, center=11.5, half_width=3.5)
    evening_peak = _triangle(hour, center=18.5, half_width=4.5)
    overnight_dip = _triangle(hour, center=3.5, half_width=4.0)
    raw = 0.45 + 0.95 * morning_peak + 1.25 * evening_peak - 0.22 * overnight_dip
    normalized = raw / 1.5611111111111111
    return max(0.20, normalized)


def _gaussian(month: int, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((month - center) / sigma) ** 2)


def _triangle(hour: int, center: float, half_width: float) -> float:
    distance = abs(hour - center)
    if distance >= half_width:
        return 0.0
    return 1.0 - distance / half_width


def _phase_for_holiday_day(day_idx: int, total_days: int) -> str:
    ratio = (day_idx + 0.5) / max(1, total_days)
    if ratio <= 0.34:
        return "outbound"
    if ratio <= 0.67:
        return "mid_holiday"
    return "return"


def _parse_range_midpoint(text: str, default: float) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return default
    if "-" not in stripped:
        try:
            return float(stripped)
        except ValueError:
            return default
    left, right = stripped.split("-", 1)
    try:
        return 0.5 * (float(left) + float(right))
    except ValueError:
        return default


def _build_holiday_day_sequence_maps(
    holiday_day_map: Mapping[Tuple[int, int], str],
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int]]:
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for key, holiday_name in holiday_day_map.items():
        by_name.setdefault(holiday_name, []).append(key)

    day_index_by_date: Dict[Tuple[int, int], int] = {}
    total_days_by_name: Dict[str, int] = {}
    for holiday_name, days in by_name.items():
        ordered = sorted(days)
        total_days_by_name[holiday_name] = len(ordered)
        for idx, key in enumerate(ordered):
            day_index_by_date[key] = idx
    return day_index_by_date, total_days_by_name


def _build_site_month_multiplier_map(
    sites: Sequence[str],
    months: Sequence[int],
    service_position: Mapping[str, float],
    service_to_archetype: Mapping[str, str],
    service_peak_time: Mapping[str, float],
    service_holiday_peak_ratio: Mapping[str, float],
    service_busy_ratio: Mapping[str, float],
    strength: float,
) -> Dict[Tuple[str, int], float]:
    archetype_scale = {"A": 0.92, "B": 1.0, "C": 1.08}
    out: Dict[Tuple[str, int], float] = {}
    for service in sites:
        pos = service_position.get(service, 0.5)
        north_weight = 1.0 - pos
        south_weight = pos
        mid_weight = 1.0 - abs(2.0 * pos - 1.0)
        size_scale = archetype_scale.get(service_to_archetype.get(service, ""), 1.0)
        peak_time = service_peak_time.get(service, 14.0)
        peak_shift = (peak_time - 14.0) / 6.0
        holiday_ratio = service_holiday_peak_ratio.get(service, 1.2)
        busy_ratio = service_busy_ratio.get(service, 1.3)

        raw: Dict[int, float] = {}
        for month in months:
            seasonal = (
                1.0
                + strength
                * (
                    0.30 * north_weight * _gaussian(month, 2.0, 1.2)
                    + 0.20 * south_weight * _gaussian(month, 7.5, 1.6)
                    + 0.55 * (0.6 + 0.4 * holiday_ratio) * (0.4 + 0.6 * mid_weight) * _gaussian(month, 10.0, 1.0)
                    + 0.20 * (0.5 + 0.5 * busy_ratio) * _gaussian(month, 5.0, 1.5)
                    + 0.10 * peak_shift * math.sin((month - 1) / 12.0 * 2.0 * math.pi)
                )
                * size_scale
            )
            raw[month] = seasonal

        avg = sum(raw.values()) / max(1, len(raw))
        for month in months:
            normalized = raw[month] / avg
            out[(service, month)] = min(1.25, max(0.82, normalized))
    return out


def _holiday_site_stagger_multiplier(
    *,
    holiday_name: str,
    month: int,
    day: int,
    hod: int,
    service: str,
    service_position: Mapping[str, float],
    service_peak_time: Mapping[str, float],
    service_to_archetype: Mapping[str, str],
    service_holiday_peak_ratio: Mapping[str, float],
    holiday_day_index: Mapping[Tuple[int, int], int],
    holiday_total_days: Mapping[str, int],
    strength: float,
) -> float:
    _ = month
    pos = service_position.get(service, 0.5)
    peak_time = service_peak_time.get(service, 14.0)
    size_width = {"A": 2.8, "B": 3.6, "C": 4.4}.get(service_to_archetype.get(service, ""), 3.4)
    holiday_ratio = service_holiday_peak_ratio.get(service, 1.2)
    day_idx = holiday_day_index.get((month, day), 0)
    total_days = holiday_total_days.get(holiday_name, 1)
    phase = _phase_for_holiday_day(day_idx, total_days)

    if phase == "outbound":
        center = peak_time - 1.2 + 4.0 * pos
        phase_bias = 0.20 * (1.0 - 2.0 * pos)
    elif phase == "mid_holiday":
        center = peak_time
        phase_bias = 0.10 * (1.0 - abs(2.0 * pos - 1.0))
    else:
        center = peak_time - 1.2 + 4.0 * (1.0 - pos)
        phase_bias = 0.20 * (2.0 * pos - 1.0)

    amplitude = strength * min(0.75, max(0.12, holiday_ratio - 1.0))
    peak_shape = _triangle(hod, center=center, half_width=size_width)
    release_shape = _triangle(hod, center=min(23.0, center + 5.5), half_width=size_width + 1.4)
    signal = phase_bias + amplitude * peak_shape - 0.30 * amplitude * release_shape
    return min(1.28, max(0.86, 1.0 + signal))


def _validate_options(options: G2LoaderOptions) -> None:
    if options.reconf_hours <= 0:
        raise ValueError("reconf_hours must be > 0")
    if options.delay_windows < 0:
        raise ValueError("delay_windows must be >= 0")
    if options.m_total_mwh is not None and options.m_total_mwh < 0:
        raise ValueError("m_total_mwh must be >= 0")
    if options.base_load_scale <= 0:
        raise ValueError("base_load_scale must be > 0")
    if options.mobile_capacity_ratio_to_fixed < 0:
        raise ValueError("mobile_capacity_ratio_to_fixed must be >= 0")
    if options.dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    if options.month_load_multiplier_by_month is not None:
        for month, multiplier in options.month_load_multiplier_by_month.items():
            month_int = int(month)
            if month_int < 1 or month_int > 12:
                raise ValueError(f"month_load_multiplier_by_month has invalid month={month}")
            if float(multiplier) < 0:
                raise ValueError(
                    f"month_load_multiplier_by_month[{month}] must be >= 0, got {multiplier}"
                )
    if options.symbolic_unserved_penalty_mode not in {"fixed", "price_multiplier", "opportunity_cost"}:
        raise ValueError(
            "symbolic_unserved_penalty_mode must be one of {'fixed', 'price_multiplier', 'opportunity_cost'}"
        )
    if options.symbolic_unserved_penalty_price_multiplier < 0:
        raise ValueError("symbolic_unserved_penalty_price_multiplier must be >= 0")
    if options.opportunity_cost_price_multiplier < 0:
        raise ValueError("opportunity_cost_price_multiplier must be >= 0")
    if options.eta_charge <= 0 or options.eta_charge > 1:
        raise ValueError("eta_charge must be in (0,1]")
    if options.eta_discharge <= 0 or options.eta_discharge > 1:
        raise ValueError("eta_discharge must be in (0,1]")
    if options.grid_capacity_mode not in {"csv", "avg_load_margin", "max_of_csv_and_avg_load_margin"}:
        raise ValueError(
            "grid_capacity_mode must be one of {'csv', 'avg_load_margin', 'max_of_csv_and_avg_load_margin'}"
        )
    if options.grid_capacity_margin_above_avg_load < 0:
        raise ValueError("grid_capacity_margin_above_avg_load must be >= 0")
    if options.site_month_heterogeneity_strength < 0:
        raise ValueError("site_month_heterogeneity_strength must be >= 0")
    if options.holiday_site_stagger_strength < 0:
        raise ValueError("holiday_site_stagger_strength must be >= 0")
    if options.demand_model not in {"csv_profile", "charger_utilization"}:
        raise ValueError("demand_model must be one of {'csv_profile', 'charger_utilization'}")
    if not (0.0 <= options.regular_day_utilization <= 1.0):
        raise ValueError("regular_day_utilization must be in [0, 1]")
    if not (0.0 <= options.holiday_day_utilization <= 1.0):
        raise ValueError("holiday_day_utilization must be in [0, 1]")
    if options.supercharger_power_mw <= 0:
        raise ValueError("supercharger_power_mw must be > 0")
    if options.normal_charger_power_mw <= 0:
        raise ValueError("normal_charger_power_mw must be > 0")
    if not (0 <= options.holiday_peak_hour_start <= 23):
        raise ValueError("holiday_peak_hour_start must be in [0, 23]")
    if not (0 <= options.holiday_peak_hour_end <= 23):
        raise ValueError("holiday_peak_hour_end must be in [0, 23]")
    if options.holiday_peak_hour_end < options.holiday_peak_hour_start:
        raise ValueError("holiday_peak_hour_end must be >= holiday_peak_hour_start")
    if options.timeline_mode not in {"representative_12x24", "expanded_8760"}:
        raise ValueError(
            "timeline_mode must be one of {'representative_12x24', 'expanded_8760'}"
        )


def _default_month_day_counts() -> Dict[int, int]:
    # Non-leap Gregorian year
    return {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }


def _build_timeline(
    months: Sequence[int],
    hours_of_day: Sequence[int],
    options: G2LoaderOptions,
) -> List[Tuple[int, int, int]]:
    """Return timeline tuples as (month, day, hour_of_day)."""
    records: List[Tuple[int, int, int]] = []

    if options.timeline_mode == "representative_12x24":
        for month in months:
            for hod in hours_of_day:
                records.append((month, 1, hod))
        return records

    day_counts = dict(_default_month_day_counts())
    if options.month_day_counts is not None:
        day_counts.update({int(k): int(v) for k, v in options.month_day_counts.items()})

    for month in months:
        days = day_counts.get(month)
        if days is None or days <= 0:
            raise ValueError(f"invalid day count for month={month}: {days}")
        for day in range(1, days + 1):
            for hod in hours_of_day:
                records.append((month, day, hod))
    return records


def _calc_window_size_steps(options: G2LoaderOptions) -> int:
    ratio = options.reconf_hours / options.dt_hours
    rounded = round(ratio)
    if abs(ratio - rounded) > 1e-9:
        raise ValueError(
            f"reconf_hours / dt_hours must be an integer number of steps, got {ratio}"
        )
    if rounded <= 0:
        raise ValueError("window size in steps must be > 0")
    return int(rounded)


def _resolve_mobile_total_mwh(
    fixed_capacity: Mapping[str, float],
    options: G2LoaderOptions,
) -> float:
    if options.m_total_mwh is not None:
        return float(options.m_total_mwh)

    fixed_total_mwh = sum(float(v) for v in fixed_capacity.values())
    derived = round(fixed_total_mwh * float(options.mobile_capacity_ratio_to_fixed))
    return float(max(0, derived))


def _resolve_grid_capacity_mw(
    service: str,
    avg_load_by_service: Mapping[str, float],
    grid_cap_site: Mapping[str, float],
    load_grid_cap: Mapping[Tuple[str, int], float],
    hod: int,
    options: G2LoaderOptions,
) -> float:
    csv_cap = float(grid_cap_site.get(service, load_grid_cap.get((service, hod), 0.0)))
    avg_margin_cap = float(avg_load_by_service.get(service, 0.0)) * (
        1.0 + float(options.grid_capacity_margin_above_avg_load)
    )

    if options.grid_capacity_mode == "csv":
        return max(0.0, csv_cap)
    if options.grid_capacity_mode == "avg_load_margin":
        return max(0.0, avg_margin_cap)
    return max(0.0, csv_cap, avg_margin_cap)


def load_g2_core_model_data(
    base_dir: str | Path,
    options: G2LoaderOptions | None = None,
) -> Tuple[CoreModelData, G2LoadDiagnostics]:
    """Load g2 revised datasets and map them into CoreModelData."""
    opts = options or G2LoaderOptions()
    _validate_options(opts)

    base = Path(base_dir).resolve()
    load_path = resolve_g2_input_path(base, "g2_beijing_shanghai_service_areas_hourly_load_revised.csv")
    price_path = resolve_g2_input_path(base, "g2_beijing_shanghai_service_areas_hourly_price_absolute_corrected.csv")
    params_path = resolve_g2_input_path(base, "g2_beijing_shanghai_service_areas_load_grid_params_revised.csv")

    for p in (load_path, price_path, params_path):
        if not p.exists():
            raise FileNotFoundError(f"missing required g2 file: {p}")

    load_rows = _read_csv(load_path)
    price_rows = _read_csv(price_path)
    params_rows = _read_csv(params_path)

    services_from_load = sorted({r["Service Area"].strip() for r in load_rows})
    services_from_price = {r["Service Area"].strip() for r in price_rows}
    services_from_params = {r["Service Area"].strip() for r in params_rows}
    sites = [s for s in services_from_load if s in services_from_price and s in services_from_params]
    if not sites:
        raise ValueError("no overlapping services across load/price/params files")

    # Load profile by (service, hour_of_day)
    load_profile: Dict[Tuple[str, int], float] = {}
    load_grid_cap: Dict[Tuple[str, int], float] = {}
    hod_set = set()
    for r in load_rows:
        service = r["Service Area"].strip()
        if service not in sites:
            continue
        hod = int(r["hour"])
        hod_set.add(hod)
        load_profile[(service, hod)] = _to_float(r.get("load_mw_nominal", ""), default=0.0)
        load_grid_cap[(service, hod)] = _to_float(r.get("grid_cap_mw", ""), default=0.0)
    hours_of_day = sorted(hod_set)

    # Site-level params
    service_to_archetype: Dict[str, str] = {}
    grid_cap_site: Dict[str, float] = {}
    unserved_penalty_site_raw: Dict[str, str] = {}
    service_position: Dict[str, float] = {}
    service_peak_time: Dict[str, float] = {}
    service_peak_nominal: Dict[str, float] = {}
    service_holiday_peak_ratio: Dict[str, float] = {}
    service_busy_ratio: Dict[str, float] = {}
    corridor_indices = [
        int(_to_float(r.get("corridor_index", ""), default=0.0))
        for r in params_rows
        if (r.get("Service Area", "") or "").strip() in sites
    ]
    min_corridor = min(corridor_indices) if corridor_indices else 0
    max_corridor = max(corridor_indices) if corridor_indices else 0
    for r in params_rows:
        service = r["Service Area"].strip()
        if service not in sites:
            continue
        service_to_archetype[service] = (r.get("archetype", "") or "").strip()
        grid_cap_site[service] = _to_float(r.get("grid_cap_mw", ""), default=0.0)
        unserved_penalty_site_raw[service] = (r.get("assumption_unserved_penalty", "") or "").strip()
        corridor_index = int(_to_float(r.get("corridor_index", ""), default=0.0))
        if max_corridor > min_corridor:
            service_position[service] = (corridor_index - min_corridor) / (max_corridor - min_corridor)
        else:
            service_position[service] = 0.5
        service_peak_time[service] = _to_float(r.get("peak_time_hour", ""), default=14.0)
        peak_nominal = _to_float(r.get("peak_power_mw_nominal", ""), default=1.0)
        service_peak_nominal[service] = peak_nominal
        service_holiday_peak_ratio[service] = _parse_range_midpoint(
            r.get("holiday_peak_mw_range", ""), peak_nominal
        ) / max(1e-6, peak_nominal)
        service_busy_ratio[service] = _parse_range_midpoint(
            r.get("busy_peak_mw_range", ""), peak_nominal
        ) / max(1e-6, peak_nominal)

    # Price with duplicate handling by (service, month, hour)
    preferred_category = opts.preferred_category_by_service or {}
    price_bucket: Dict[Tuple[str, int, int], List[Tuple[float, str]]] = {}
    for r in price_rows:
        service = r["Service Area"].strip()
        if service not in sites:
            continue
        month = int(r["month"])
        hod = int(r["hour"])
        value = _to_float(r.get("hourly_price_abs", ""), default=0.0)
        category = (r.get("分类", "") or "").strip()
        key = (service, month, hod)
        price_bucket.setdefault(key, []).append((value, category))

    months = sorted({int(r["month"]) for r in price_rows})
    if not months:
        raise ValueError("no month data found in price file")

    duplicate_price_keys = 0
    duplicate_price_rows = 0

    resolved_price: Dict[Tuple[str, int, int], float] = {}
    for service in sites:
        preferred = preferred_category.get(service, "").strip()
        for month in months:
            for hod in hours_of_day:
                key = (service, month, hod)
                rows = price_bucket.get(key, [])
                if len(rows) == 0:
                    raise ValueError(f"missing price row for key={key}")
                if len(rows) > 1:
                    duplicate_price_keys += 1
                    duplicate_price_rows += len(rows) - 1

                selected_values: List[float]
                if preferred:
                    pref_vals = [v for v, cat in rows if cat == preferred]
                    if pref_vals:
                        selected_values = pref_vals
                    else:
                        selected_values = [v for v, _cat in rows]
                else:
                    selected_values = [v for v, _cat in rows]

                resolved_price[key] = _resolve_duplicates(selected_values, opts.price_duplicate_policy)

    site_default_penalty: Dict[str, float] = {}
    for service in sites:
        raw_penalty = unserved_penalty_site_raw.get(service, "")
        if raw_penalty:
            try:
                site_default_penalty[service] = float(raw_penalty)
                continue
            except ValueError:
                pass

        if opts.symbolic_unserved_penalty_mode == "fixed":
            site_default_penalty[service] = opts.default_unserved_penalty_yuan_per_mwh
        elif opts.symbolic_unserved_penalty_mode == "price_multiplier":
            site_price_max = max(
                resolved_price[(service, month, hod)] for month in months for hod in hours_of_day
            )
            site_default_penalty[service] = (
                site_price_max * opts.symbolic_unserved_penalty_price_multiplier
            )

    # Build time axis
    # timeline records = (month, day, hour_of_day)
    timeline_records = _build_timeline(months, hours_of_day, opts)
    hours = list(range(len(timeline_records)))
    window_size_steps = _calc_window_size_steps(opts)
    hour_to_window: Dict[int, int] = {t: t // window_size_steps for t in hours}
    n_windows = (len(hours) + window_size_steps - 1) // window_size_steps
    windows = list(range(n_windows))

    # Core maps
    load_mw: Dict[Tuple[str, int], float] = {}
    price_map: Dict[Tuple[str, int], float] = {}
    grid_limit: Dict[Tuple[str, int], float] = {}
    unserved_penalty: Dict[Tuple[str, int], float] = {}
    holiday_day_map = (
        _build_cn_holiday_day_map(opts.holiday_year)
        if opts.timeline_mode == "expanded_8760" and opts.apply_cn_holiday_shocks
        else {}
    )
    holiday_multiplier_by_name = dict(_default_holiday_load_multiplier_by_name())
    if opts.holiday_load_multiplier_by_name is not None:
        holiday_multiplier_by_name.update(
            {str(k): float(v) for k, v in opts.holiday_load_multiplier_by_name.items()}
        )
    holiday_day_index, holiday_total_days = _build_holiday_day_sequence_maps(holiday_day_map)
    month_load_multiplier = {
        int(month): float(multiplier)
        for month, multiplier in (opts.month_load_multiplier_by_month or {}).items()
    }
    auto_site_month_multiplier = (
        _build_site_month_multiplier_map(
            sites,
            months,
            service_position,
            service_to_archetype,
            service_peak_time,
            service_holiday_peak_ratio,
            service_busy_ratio,
            opts.site_month_heterogeneity_strength,
        )
        if opts.enable_site_month_heterogeneity
        else {(service, month): 1.0 for service in sites for month in months}
    )

    charger_nominal_power = _nominal_site_power_from_chargers(service_to_archetype, opts)
    for service in sites:
        for t, (month, day, hod) in enumerate(timeline_records):
            holiday_name = holiday_day_map.get((month, day))
            if opts.demand_model == "charger_utilization":
                utilization = (
                    opts.holiday_day_utilization if holiday_name is not None else opts.regular_day_utilization
                )
                load_value = (
                    charger_nominal_power[service]
                    * utilization
                    * _charger_daily_shape_multiplier(hod)
                )
            else:
                load_value = load_profile.get((service, hod), 0.0) * opts.base_load_scale
                load_value *= month_load_multiplier.get(month, 1.0)
                load_value *= auto_site_month_multiplier[(service, month)]
                if holiday_name is not None and opts.holiday_peak_hour_start <= hod <= opts.holiday_peak_hour_end:
                    load_value *= holiday_multiplier_by_name.get(holiday_name, 1.0)
                    if opts.enable_holiday_site_stagger:
                        load_value *= _holiday_site_stagger_multiplier(
                            holiday_name=holiday_name,
                            month=month,
                            day=day,
                            hod=hod,
                            service=service,
                            service_position=service_position,
                            service_peak_time=service_peak_time,
                            service_to_archetype=service_to_archetype,
                            service_holiday_peak_ratio=service_holiday_peak_ratio,
                            holiday_day_index=holiday_day_index,
                            holiday_total_days=holiday_total_days,
                            strength=opts.holiday_site_stagger_strength,
                        )
            load_mw[(service, t)] = load_value

    avg_load_by_service: Dict[str, float] = {
        service: (
            sum(load_mw[(service, t)] for t in hours) / len(hours)
            if len(hours) > 0
            else 0.0
        )
        for service in sites
    }

    for service in sites:
        for t, (month, _day, hod) in enumerate(timeline_records):
            price_map[(service, t)] = resolved_price[(service, month, hod)]
            if opts.demand_model == "charger_utilization":
                grid_limit[(service, t)] = charger_nominal_power[service]
            else:
                grid_limit[(service, t)] = _resolve_grid_capacity_mw(
                    service,
                    avg_load_by_service,
                    grid_cap_site,
                    load_grid_cap,
                    hod,
                    opts,
                )
            if service in site_default_penalty:
                penalty_value = site_default_penalty[service]
            elif opts.symbolic_unserved_penalty_mode == "opportunity_cost":
                penalty_value = (
                    price_map[(service, t)] * float(opts.opportunity_cost_price_multiplier)
                )
            else:
                penalty_value = opts.default_unserved_penalty_yuan_per_mwh
            unserved_penalty[(service, t)] = penalty_value

    fixed_capacity = _build_fixed_capacity_map(service_to_archetype, opts)
    mobile_total_mwh = _resolve_mobile_total_mwh(fixed_capacity, opts)
    soc_initial = {s: opts.soc_initial_frac for s in sites}
    soc_min = {s: opts.soc_min_frac for s in sites}
    soc_max = {s: opts.soc_max_frac for s in sites}
    charge_c_rate = {s: opts.charge_c_rate for s in sites}
    discharge_c_rate = {s: opts.discharge_c_rate for s in sites}

    reconf_limit = _build_reconf_limit_map(windows, opts.reconf_limit_mwh)
    active_init = _build_active_init(sites, windows, mobile_total_mwh, opts)

    data = CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load_mw,
        price_yuan_per_mwh=price_map,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=unserved_penalty,
        fixed_capacity_mwh=fixed_capacity,
        soc_initial_frac=soc_initial,
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
        charge_c_rate=charge_c_rate,
        discharge_c_rate=discharge_c_rate,
        eta_charge=opts.eta_charge,
        eta_discharge=opts.eta_discharge,
        dt_hours=opts.dt_hours,
        m_total_mwh=mobile_total_mwh,
        delay_windows=opts.delay_windows,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=opts.c_reconf_yuan_per_mwh,
        c_storage_fixed_yuan_per_mwh=opts.c_storage_fixed_yuan_per_mwh,
        c_storage_mobile_yuan_per_mwh=opts.c_storage_mobile_yuan_per_mwh,
    )

    diag = G2LoadDiagnostics(
        sites=len(sites),
        months=len(months),
        hours_of_day=len(hours_of_day),
        timeline_hours=len(hours),
        windows=len(windows),
        window_size_steps=window_size_steps,
        timeline_mode=opts.timeline_mode,
        duplicate_price_keys=duplicate_price_keys,
        duplicate_price_rows=duplicate_price_rows,
        price_duplicate_policy=opts.price_duplicate_policy,
        fixed_total_mwh=sum(fixed_capacity.values()),
        mobile_total_mwh=mobile_total_mwh,
    )
    return data, diag
