"""Unified experiment runner for the EEP study."""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .core_model import CoreModelData, CoreSolveDiagnostics, solve_core_model_with_diagnostics
from .g2_loader import G2LoaderOptions, load_g2_core_model_data, resolve_g2_input_path


EPS = 1e-9
PEAK_THRESHOLD_THETA = 0.8
SYNTHETIC_SITES = ["N1", "N2", "N3", "N4", "N5"]
SYNTHETIC_FIXED_CAPACITY_MWH = {"N1": 0.4, "N2": 0.8, "N3": 1.2, "N4": 0.8, "N5": 0.4}
CASE_ORDER = ["no_storage", "fixed_only", "mobile_only", "hybrid"]
CASE_LABELS = {
    "no_storage": "Case 0: No Storage",
    "fixed_only": "Case A: Fixed Only",
    "mobile_only": "Case B: Mobile Only",
    "mobile_only_ideal": "Case B*: Ideal Mobile Only",
    "hybrid": "Case C: Hybrid",
}
WINNER_CODE = {
    "no_storage": 0,
    "fixed_only": 1,
    "mobile_only": 2,
    "hybrid": 3,
}
WINNER_NAME = {value: key for key, value in WINNER_CODE.items()}
WINNER_COLORS = {
    "no_storage": "#9aa5b1",
    "fixed_only": "#1f77b4",
    "mobile_only": "#d95f02",
    "hybrid": "#2ca02c",
}
HOLIDAY_RANGES_2026 = {
    "spring_festival": (date(2026, 2, 15), date(2026, 2, 23)),
    "national_day": (date(2026, 10, 1), date(2026, 10, 7)),
}
_REAL_DATA_CACHE: dict[tuple[object, ...], CoreModelData] = {}
CASE_TIME_LIMIT_SECONDS = float(os.environ.get("EEP_CASE_TIME_LIMIT_SECONDS", str(8 * 60)))
LP_RELAXATION_TIME_LIMIT_SECONDS = min(
    float(os.environ.get("EEP_LP_RELAXATION_TIME_LIMIT_SECONDS", "120.0")),
    CASE_TIME_LIMIT_SECONDS,
)
WORKER_TIMEOUT_BUFFER_SECONDS = float(os.environ.get("EEP_WORKER_TIMEOUT_BUFFER_SECONDS", "120.0"))
EXPERIMENT_WORKERS = max(1, int(os.environ.get("EEP_EXPERIMENT_WORKERS", "3")))
CASE_SOLVE_WORKERS_DEFAULT = "1" if EXPERIMENT_WORKERS > 1 else str(min(4, os.cpu_count() or 1))
CASE_SOLVE_WORKERS = max(
    1,
    int(os.environ.get("EEP_CASE_SOLVE_WORKERS", CASE_SOLVE_WORKERS_DEFAULT)),
)
CONCLUSION_READY_CASE_OUTCOMES = {"optimal", "feasible_with_gap"}


@dataclass(frozen=True)
class ExperimentDefaults:
    timeline_mode: str = "representative_12x24"
    reconf_hours: int = 24
    delay_windows: int = 1
    mobile_capacity_ratio_to_fixed: float = 0.7
    fixed_cost_per_mwh: float = 1.0
    mobile_cost_premium_ratio: float = 1.5
    hybrid_fixed_budget_share: float = 0.5
    budget_multiplier_relative_to_fixed_baseline: float = 1.0
    grid_capacity_mode: str = "max_of_csv_and_avg_load_margin"
    grid_capacity_margin_above_avg_load: float = 0.25
    default_unserved_penalty_yuan_per_mwh: float = 1000.0
    symbolic_unserved_penalty_mode: str = "fixed"
    c_reconf_yuan_per_mwh: float = 0.5
    solver: str = "highs"
    objective_mode: str = "planning"


@dataclass
class ExperimentScenario:
    experiment_id: str
    variant_id: str
    question: str
    track: str
    source: str
    case: str
    timeline_mode: str
    reconf_hours: int
    delay_windows: int
    c_reconf_yuan_per_mwh: float
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class ScenarioMetrics:
    experiment_id: str
    variant_id: str
    question: str
    track: str
    source: str
    case: str
    case_label: str
    solver_status: str
    termination_condition: str
    case_outcome: str
    best_incumbent: float | None
    best_bound: float | None
    gap: float | None
    elapsed_seconds: float
    lp_relaxation_objective: float | None
    timeline_mode: str
    reconf_hours: int
    delay_windows: int
    fixed_total_mwh: float
    mobile_total_mwh: float
    c_reconf_yuan_per_mwh: float
    C_total: float
    C_grid: float
    C_unserved: float
    C_reconf: float
    C_storage: float
    C_total_with_storage: float
    total_unmet_load: float
    total_grid_purchase: float
    total_charging_energy: float
    total_discharging_energy: float
    total_reconfiguration_volume: float
    average_active_mobile_capacity_utilisation: float
    SC_1: float
    HT: float
    R_delay: float
    V_reconf_norm: float
    B_reuse: float | None = None
    L_friction: float | None = None
    V_mob_net: float | None = None
    note: str = ""
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def to_row(self) -> dict[str, str | int | float]:
        row = asdict(self)
        metadata = row.pop("metadata", {})
        for key, value in metadata.items():
            row[key] = value
        return row


@dataclass(frozen=True)
class ScenarioSolveRequest:
    data: CoreModelData
    scenario: ExperimentScenario
    solver_name: str
    include_solution: bool = False


@dataclass
class ExperimentArtifacts:
    experiment_id: str
    question: str
    raw_csv: Path
    summary_csv: Path
    memo_md: Path
    figure_paths: list[Path]
    raw_rows: list[dict[str, str | int | float]]
    summary_rows: list[dict[str, str | int | float]]
    status: str


def serialize_core_model_data(data: CoreModelData) -> dict[str, object]:
    return {
        "sites": list(data.sites),
        "hours": list(data.hours),
        "windows": list(data.windows),
        "hour_to_window": list(data.hour_to_window.items()),
        "load_mw": [(site, hour, value) for (site, hour), value in data.load_mw.items()],
        "price_yuan_per_mwh": [(site, hour, value) for (site, hour), value in data.price_yuan_per_mwh.items()],
        "grid_limit_mw": [(site, hour, value) for (site, hour), value in data.grid_limit_mw.items()],
        "unserved_penalty_yuan_per_mwh": [
            (site, hour, value) for (site, hour), value in data.unserved_penalty_yuan_per_mwh.items()
        ],
        "fixed_capacity_mwh": list(data.fixed_capacity_mwh.items()),
        "soc_initial_frac": list(data.soc_initial_frac.items()),
        "soc_min_frac": list(data.soc_min_frac.items()),
        "soc_max_frac": list(data.soc_max_frac.items()),
        "charge_c_rate": list(data.charge_c_rate.items()),
        "discharge_c_rate": list(data.discharge_c_rate.items()),
        "eta_charge": data.eta_charge,
        "eta_discharge": data.eta_discharge,
        "dt_hours": data.dt_hours,
        "m_total_mwh": data.m_total_mwh,
        "delay_windows": data.delay_windows,
        "active_init_mwh": [(site, window, value) for (site, window), value in data.active_init_mwh.items()],
        "reconf_limit_mwh": list(data.reconf_limit_mwh.items()),
        "c_reconf_yuan_per_mwh": data.c_reconf_yuan_per_mwh,
        "c_storage_fixed_yuan_per_mwh": data.c_storage_fixed_yuan_per_mwh,
        "c_storage_mobile_yuan_per_mwh": data.c_storage_mobile_yuan_per_mwh,
        "objective_mode": data.objective_mode,
    }


def deserialize_core_model_data(payload: Mapping[str, object]) -> CoreModelData:
    return CoreModelData(
        sites=list(payload["sites"]),
        hours=list(payload["hours"]),
        windows=list(payload["windows"]),
        hour_to_window={int(hour): int(window) for hour, window in payload["hour_to_window"]},
        load_mw={(site, int(hour)): float(value) for site, hour, value in payload["load_mw"]},
        price_yuan_per_mwh={(site, int(hour)): float(value) for site, hour, value in payload["price_yuan_per_mwh"]},
        grid_limit_mw={(site, int(hour)): float(value) for site, hour, value in payload["grid_limit_mw"]},
        unserved_penalty_yuan_per_mwh={
            (site, int(hour)): float(value) for site, hour, value in payload["unserved_penalty_yuan_per_mwh"]
        },
        fixed_capacity_mwh={site: float(value) for site, value in payload["fixed_capacity_mwh"]},
        soc_initial_frac={site: float(value) for site, value in payload["soc_initial_frac"]},
        soc_min_frac={site: float(value) for site, value in payload["soc_min_frac"]},
        soc_max_frac={site: float(value) for site, value in payload["soc_max_frac"]},
        charge_c_rate={site: float(value) for site, value in payload["charge_c_rate"]},
        discharge_c_rate={site: float(value) for site, value in payload["discharge_c_rate"]},
        eta_charge=float(payload["eta_charge"]),
        eta_discharge=float(payload["eta_discharge"]),
        dt_hours=float(payload["dt_hours"]),
        m_total_mwh=float(payload["m_total_mwh"]),
        delay_windows=int(payload["delay_windows"]),
        active_init_mwh={(site, int(window)): float(value) for site, window, value in payload["active_init_mwh"]},
        reconf_limit_mwh={int(window): float(value) for window, value in payload["reconf_limit_mwh"]},
        c_reconf_yuan_per_mwh=float(payload["c_reconf_yuan_per_mwh"]),
        c_storage_fixed_yuan_per_mwh=float(payload["c_storage_fixed_yuan_per_mwh"]),
        c_storage_mobile_yuan_per_mwh=float(payload["c_storage_mobile_yuan_per_mwh"]),
        objective_mode=str(payload.get("objective_mode", "operational")),
    )


def serialize_experiment_scenario(scenario: ExperimentScenario) -> dict[str, object]:
    return {
        "experiment_id": scenario.experiment_id,
        "variant_id": scenario.variant_id,
        "question": scenario.question,
        "track": scenario.track,
        "source": scenario.source,
        "case": scenario.case,
        "timeline_mode": scenario.timeline_mode,
        "reconf_hours": scenario.reconf_hours,
        "delay_windows": scenario.delay_windows,
        "c_reconf_yuan_per_mwh": scenario.c_reconf_yuan_per_mwh,
        "metadata": dict(scenario.metadata),
    }


def deserialize_experiment_scenario(payload: Mapping[str, object]) -> ExperimentScenario:
    return ExperimentScenario(
        experiment_id=str(payload["experiment_id"]),
        variant_id=str(payload["variant_id"]),
        question=str(payload["question"]),
        track=str(payload["track"]),
        source=str(payload["source"]),
        case=str(payload["case"]),
        timeline_mode=str(payload["timeline_mode"]),
        reconf_hours=int(payload["reconf_hours"]),
        delay_windows=int(payload["delay_windows"]),
        c_reconf_yuan_per_mwh=float(payload["c_reconf_yuan_per_mwh"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _ensure_dir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if len(rows) == 1 and set(rows[0].keys()) == {"empty"} and rows[0].get("empty", "") == "":
        return []
    return rows


def _normalize_existing_path(path_text: str) -> Path:
    normalized = path_text.strip()
    if normalized.startswith("/mnt/") and len(normalized) > 6 and normalized[5].isalpha() and normalized[6] == "/":
        drive = normalized[5].upper()
        remainder = normalized[7:].replace("/", "\\")
        return Path(f"{drive}:\\{remainder}")
    return Path(normalized)


def _hour_index_for_datetime(year: int, month: int, day: int, hod: int) -> int:
    base = date(year, 1, 1)
    current = date(year, month, day)
    return ((current - base).days * 24) + hod


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _load_corridor_order_and_archetype(base_dir: Path) -> tuple[list[str], dict[str, str]]:
    rows = list(
        csv.DictReader(
            resolve_g2_input_path(base_dir, "g2_beijing_shanghai_service_areas_load_grid_params_revised.csv").open(
                encoding="utf-8-sig"
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
    archetype_scale = {"A": 0.9, "B": 1.0, "C": 1.1}
    north_weight = 1.0 - position
    south_weight = position
    mid_weight = 1.0 - abs(2.0 * position - 1.0)
    size_scale = archetype_scale.get(archetype, 1.0)

    raw = (
        1.0
        + 0.45 * north_weight * _gaussian(month, 2.0, 1.2)
        + 0.35 * south_weight * _gaussian(month, 8.0, 1.4)
        + 0.25 * mid_weight * _gaussian(month, 5.0, 1.6)
        + 0.30 * (0.5 + 0.5 * mid_weight) * _gaussian(month, 10.0, 1.0)
    )
    return 0.72 + (raw - 1.0) * size_scale


def _normalize_site_multipliers(
    corridor_order: Sequence[str],
    archetype_by_service: Mapping[str, str],
) -> dict[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for idx, service in enumerate(corridor_order):
        position = idx / max(1, len(corridor_order) - 1)
        archetype = archetype_by_service.get(service, "")
        raw = {month: _site_month_multiplier(month, position, archetype) for month in range(1, 13)}
        avg = sum(raw.values()) / 12.0
        for month, value in raw.items():
            output[(service, month)] = min(1.55, max(0.65, value / avg))
    return output


def _apply_site_month_multipliers(
    data: CoreModelData,
    site_month_multiplier: Mapping[tuple[str, int], float],
) -> CoreModelData:
    load_mw = dict(data.load_mw)
    for site in data.sites:
        for t in data.hours:
            month = int(data.hour_to_window[t]) + 1
            load_mw[(site, t)] *= float(site_month_multiplier[(site, month)])
    return replace(data, load_mw=load_mw)


def _triangle(hour: int, center: float, half_width: float) -> float:
    distance = abs(hour - center)
    if distance >= half_width:
        return 0.0
    return 1.0 - distance / half_width


def _phase_for_day(day_idx: int, total_days: int) -> str:
    ratio = (day_idx + 0.5) / max(1, total_days)
    if ratio <= 0.34:
        return "outbound"
    if ratio <= 0.67:
        return "mid_holiday"
    return "return"


def _apply_staggered_holiday_profile(
    data: CoreModelData,
    *,
    holiday_name: str,
    corridor_order: Sequence[str],
    archetype_by_service: Mapping[str, str],
    stagger_scale: float = 1.0,
) -> CoreModelData:
    archetype_scale = {"A": 0.95, "B": 1.05, "C": 1.15}
    archetype_width = {"A": 2.5, "B": 3.5, "C": 4.5}
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
            size_scale = archetype_scale.get(archetype, 1.0)
            base_width = archetype_width.get(archetype, 3.0)
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

            for hod in range(24):
                t = day_idx * 24 + hod
                shape = _triangle(hod, center=center, half_width=half_width)
                shoulder = 0.08 * size_scale if 9 <= hod <= 21 else 0.0
                surge_signal = max(0.0, phase_offset + shoulder + amplitude * shape) * stagger_scale
                multiplier = 0.60 + surge_signal
                load_mw[(service, t)] *= multiplier
    return replace(data, load_mw=load_mw)


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

    active_init: dict[tuple[str, int], float] = {}
    if delay_windows > 0:
        share = data.m_total_mwh / max(1, len(data.sites)) if initial_active_mode == "equal" else 0.0
        for w in windows:
            if w < delay_windows:
                for site in data.sites:
                    active_init[(site, w)] = share

    def _slice_map(source: Mapping[tuple[str, int], float]) -> dict[tuple[str, int], float]:
        sliced: dict[tuple[str, int], float] = {}
        for site in data.sites:
            for old_t, new_t in hour_map.items():
                sliced[(site, new_t)] = float(source[(site, old_t)])
        return sliced

    return CoreModelData(
        sites=list(data.sites),
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=_slice_map(data.load_mw),
        price_yuan_per_mwh=_slice_map(data.price_yuan_per_mwh),
        grid_limit_mw=_slice_map(data.grid_limit_mw),
        unserved_penalty_yuan_per_mwh=_slice_map(data.unserved_penalty_yuan_per_mwh),
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
        objective_mode=data.objective_mode,
    )


def _winner_for_rows(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "no_storage"
    ranked = sorted(rows, key=lambda row: float(row["C_total"]))
    return str(ranked[0]["case"])


def _winner_name(case: str) -> str:
    return CASE_LABELS.get(case, case)


def _line_midpoint(values: Sequence[float], idx: int) -> float:
    if idx <= 0:
        return float(values[0])
    return (float(values[idx - 1]) + float(values[idx])) / 2.0


def _daily_shape(hour_of_day: int) -> float:
    if 0 <= hour_of_day <= 5:
        return 0.78
    if 6 <= hour_of_day <= 8:
        return 0.92
    if 9 <= hour_of_day <= 11:
        return 1.02
    if 12 <= hour_of_day <= 16:
        return 1.08
    if 17 <= hour_of_day <= 20:
        return 1.0
    return 0.84


def _price_shape(hour_of_day: int) -> float:
    if 0 <= hour_of_day <= 6:
        return 0.55
    if 7 <= hour_of_day <= 10 or 21 <= hour_of_day <= 23:
        return 0.9
    return 1.45


def _route_for_turnover(windows: int, residence_windows: int) -> list[str]:
    if residence_windows <= 0:
        raise ValueError(f"residence_windows must be > 0, got {residence_windows}")
    route = ["N1", "N2", "N3", "N4", "N5", "N4", "N3", "N2"]
    leaders: list[str] = []
    while len(leaders) < windows:
        for leader in route:
            leaders.extend([leader] * residence_windows)
            if len(leaders) >= windows:
                break
    return leaders[:windows]


def _is_case_conclusion_ready(row: Mapping[str, object]) -> bool:
    return str(row.get("case_outcome", "")).lower() in CONCLUSION_READY_CASE_OUTCOMES


def _site_distance(site_a: str, site_b: str) -> int:
    return abs(SYNTHETIC_SITES.index(site_a) - SYNTHETIC_SITES.index(site_b))


@dataclass(frozen=True)
class SyntheticCorridorConfig:
    reconf_hours: int = 24
    windows: int = 8
    delay_windows: int = 1
    c_reconf_yuan_per_mwh: float = 0.5
    concentration: float = 0.8
    residence_windows: int = 1
    hotspot_extra_mw: float = 1.6
    grid_margin_frac: float = 0.05
    base_load_mw: float = 0.95
    theta: float = PEAK_THRESHOLD_THETA

    @property
    def mobile_total_mwh(self) -> float:
        return float(round(sum(SYNTHETIC_FIXED_CAPACITY_MWH.values()) * 0.7))


def build_synthetic_corridor_data(config: SyntheticCorridorConfig) -> CoreModelData:
    sites = list(SYNTHETIC_SITES)
    hours_per_window = config.reconf_hours
    windows = list(range(config.windows))
    hours = list(range(config.windows * hours_per_window))
    hour_to_window = {t: t // hours_per_window for t in hours}
    leaders = _route_for_turnover(config.windows, config.residence_windows)

    soc_initial = {site: 0.5 for site in sites}
    soc_min = {site: 0.1 for site in sites}
    soc_max = {site: 0.9 for site in sites}
    charge_rate = {site: 1.0 for site in sites}
    discharge_rate = {site: 1.0 for site in sites}
    reconf_limit = {w: 2.0 * config.mobile_total_mwh for w in windows}
    active_init = {}
    if config.delay_windows > 0:
        share = config.mobile_total_mwh / len(sites)
        for w in windows:
            if w < config.delay_windows:
                for site in sites:
                    active_init[(site, w)] = share

    load_mw: dict[tuple[str, int], float] = {}
    price: dict[tuple[str, int], float] = {}
    grid_limit: dict[tuple[str, int], float] = {}
    penalty: dict[tuple[str, int], float] = {}

    base_grid_limit = config.base_load_mw * (1.0 + config.grid_margin_frac)
    remainder = max(0.0, 1.0 - config.concentration)

    for t in hours:
        w = hour_to_window[t]
        leader = leaders[w]
        hour_of_day = t % 24
        peak_signal = 1.0 if 11 <= hour_of_day <= 18 else 0.25 if 9 <= hour_of_day <= 20 else 0.0
        for site in sites:
            distance = _site_distance(site, leader)
            if distance == 0:
                share = config.concentration
            elif distance == 1:
                share = remainder * 0.55
            elif distance == 2:
                share = remainder * 0.3
            else:
                share = remainder * 0.15 / max(1, len(sites) - 3)

            hotspot = config.hotspot_extra_mw * share * peak_signal
            load_mw[(site, t)] = config.base_load_mw * _daily_shape(hour_of_day) + hotspot
            price[(site, t)] = _price_shape(hour_of_day)
            grid_limit[(site, t)] = base_grid_limit
            penalty[(site, t)] = 12.0

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load_mw,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh=dict(SYNTHETIC_FIXED_CAPACITY_MWH),
        soc_initial_frac=soc_initial,
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
        charge_c_rate=charge_rate,
        discharge_c_rate=discharge_rate,
        eta_charge=0.95,
        eta_discharge=0.95,
        dt_hours=1.0,
        m_total_mwh=config.mobile_total_mwh,
        delay_windows=config.delay_windows,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=config.c_reconf_yuan_per_mwh,
        c_storage_fixed_yuan_per_mwh=0.0,
        c_storage_mobile_yuan_per_mwh=0.0,
        objective_mode="operational",
    )


def load_real_data(base_dir: Path, defaults: ExperimentDefaults, *, timeline_mode: str | None = None, reconf_hours: int | None = None, delay_windows: int | None = None, c_reconf_yuan_per_mwh: float | None = None) -> CoreModelData:
    resolved_timeline_mode = timeline_mode or defaults.timeline_mode
    resolved_reconf_hours = reconf_hours if reconf_hours is not None else defaults.reconf_hours
    resolved_delay_windows = delay_windows if delay_windows is not None else defaults.delay_windows
    resolved_c_reconf = (
        c_reconf_yuan_per_mwh
        if c_reconf_yuan_per_mwh is not None
        else defaults.c_reconf_yuan_per_mwh
    )
    cache_key = (
        base_dir.resolve().as_posix(),
        resolved_timeline_mode,
        resolved_reconf_hours,
        resolved_delay_windows,
        defaults.mobile_capacity_ratio_to_fixed,
        defaults.grid_capacity_mode,
        defaults.grid_capacity_margin_above_avg_load,
        defaults.default_unserved_penalty_yuan_per_mwh,
        defaults.symbolic_unserved_penalty_mode,
        resolved_c_reconf,
    )
    cached = _REAL_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    options = G2LoaderOptions(
        reconf_hours=resolved_reconf_hours,
        delay_windows=resolved_delay_windows,
        mobile_capacity_ratio_to_fixed=defaults.mobile_capacity_ratio_to_fixed,
        timeline_mode=resolved_timeline_mode,
        grid_capacity_mode=defaults.grid_capacity_mode,
        grid_capacity_margin_above_avg_load=defaults.grid_capacity_margin_above_avg_load,
        default_unserved_penalty_yuan_per_mwh=defaults.default_unserved_penalty_yuan_per_mwh,
        symbolic_unserved_penalty_mode=defaults.symbolic_unserved_penalty_mode,
        c_reconf_yuan_per_mwh=resolved_c_reconf,
        c_storage_fixed_yuan_per_mwh=0.0,
        c_storage_mobile_yuan_per_mwh=0.0,
    )
    data, _diag = load_g2_core_model_data(base_dir, options=options)
    data = replace(data, objective_mode=defaults.objective_mode)
    _REAL_DATA_CACHE[cache_key] = data
    return data


def make_case_data(
    data: CoreModelData,
    case: str,
    *,
    delay_windows: int | None = None,
    c_reconf_yuan_per_mwh: float | None = None,
) -> CoreModelData:
    windows = list(data.windows)
    zero_fixed = {site: 0.0 for site in data.sites}
    zero_reconf = {w: 0.0 for w in windows}

    if case == "no_storage":
        return replace(
            data,
            fixed_capacity_mwh=zero_fixed,
            m_total_mwh=0.0,
            delay_windows=0,
            active_init_mwh={},
            reconf_limit_mwh=zero_reconf,
            c_reconf_yuan_per_mwh=0.0,
        )

    if case == "fixed_only":
        return replace(
            data,
            m_total_mwh=0.0,
            delay_windows=0,
            active_init_mwh={},
            reconf_limit_mwh=zero_reconf,
            c_reconf_yuan_per_mwh=0.0,
        )

    if case == "mobile_only":
        effective_delay = data.delay_windows if delay_windows is None else delay_windows
        effective_reconf = (
            data.c_reconf_yuan_per_mwh if c_reconf_yuan_per_mwh is None else c_reconf_yuan_per_mwh
        )
        return replace(
            data,
            fixed_capacity_mwh=zero_fixed,
            delay_windows=effective_delay,
            c_reconf_yuan_per_mwh=effective_reconf,
        )

    if case == "hybrid":
        return replace(
            data,
            delay_windows=data.delay_windows if delay_windows is None else delay_windows,
            c_reconf_yuan_per_mwh=(
                data.c_reconf_yuan_per_mwh
                if c_reconf_yuan_per_mwh is None
                else c_reconf_yuan_per_mwh
            ),
        )

    raise ValueError(f"unsupported case: {case}")


def prepare_capacity_fair_mobile_data(
    data: CoreModelData,
    *,
    mobile_total_mwh: float | None = None,
) -> CoreModelData:
    """Return a base dataset whose mobile budget matches the chosen fair-capacity target.

    By default the target mobile budget is the fixed-storage total already embedded in ``data``.
    Early active mobile capacity is rebuilt using the same equal-split convention as the loader,
    so a fair mobile-only comparison does not accidentally inherit an outdated initial allocation.
    """

    target_mobile_total = (
        float(sum(data.fixed_capacity_mwh.values()))
        if mobile_total_mwh is None
        else float(mobile_total_mwh)
    )
    if target_mobile_total < 0:
        raise ValueError("mobile_total_mwh must be >= 0")

    active_init: dict[tuple[str, int], float] = {}
    if data.delay_windows > 0 and len(data.sites) > 0:
        share = target_mobile_total / len(data.sites)
        active_init = {
            (site, window): share
            for site in data.sites
            for window in data.windows
            if window < data.delay_windows
        }
    reconf_scale = 0.0 if data.m_total_mwh <= 0 else target_mobile_total / float(data.m_total_mwh)
    scaled_reconf_limit = {
        window: float(limit) * reconf_scale for window, limit in data.reconf_limit_mwh.items()
    }

    return replace(
        data,
        m_total_mwh=target_mobile_total,
        active_init_mwh=active_init,
        reconf_limit_mwh=scaled_reconf_limit,
    )


def prepare_same_budget_data(
    data: CoreModelData,
    *,
    mobile_cost_premium_ratio: float,
    fixed_budget_share: float,
    budget_in_fixed_cost_units: float | None = None,
    fixed_cost_per_mwh: float = 1.0,
) -> CoreModelData:
    """Return a planning-mode dataset that satisfies a shared storage-investment budget.

    The budget is expressed in fixed-storage cost units. By default, the total budget matches the
    baseline fixed-only investment, i.e. ``fixed_cost_per_mwh * sum_i S_i^F``.
    """

    if mobile_cost_premium_ratio <= 0:
        raise ValueError("mobile_cost_premium_ratio must be > 0")
    if fixed_cost_per_mwh <= 0:
        raise ValueError("fixed_cost_per_mwh must be > 0")
    if not (0.0 <= fixed_budget_share <= 1.0):
        raise ValueError("fixed_budget_share must lie in [0, 1]")

    base_fixed_total = float(sum(data.fixed_capacity_mwh.values()))
    total_budget = (
        float(budget_in_fixed_cost_units)
        if budget_in_fixed_cost_units is not None
        else fixed_cost_per_mwh * base_fixed_total
    )
    if total_budget < 0:
        raise ValueError("budget_in_fixed_cost_units must be >= 0")

    fixed_budget = total_budget * fixed_budget_share
    mobile_budget = total_budget - fixed_budget
    target_fixed_total = fixed_budget / fixed_cost_per_mwh if fixed_cost_per_mwh > 0 else 0.0
    target_mobile_total = mobile_budget / (fixed_cost_per_mwh * mobile_cost_premium_ratio)

    if base_fixed_total > 0:
        fixed_scale = target_fixed_total / base_fixed_total
        scaled_fixed = {site: float(capacity) * fixed_scale for site, capacity in data.fixed_capacity_mwh.items()}
    else:
        scaled_fixed = {site: 0.0 for site in data.sites}

    active_init: dict[tuple[str, int], float] = {}
    if data.delay_windows > 0 and len(data.sites) > 0 and target_mobile_total > 0:
        share = target_mobile_total / len(data.sites)
        active_init = {
            (site, window): share
            for site in data.sites
            for window in data.windows
            if window < data.delay_windows
        }
    reconf_scale = 0.0 if data.m_total_mwh <= 0 else target_mobile_total / float(data.m_total_mwh)
    scaled_reconf_limit = {
        window: float(limit) * reconf_scale for window, limit in data.reconf_limit_mwh.items()
    }

    return replace(
        data,
        fixed_capacity_mwh=scaled_fixed,
        m_total_mwh=target_mobile_total,
        active_init_mwh=active_init,
        reconf_limit_mwh=scaled_reconf_limit,
        c_storage_fixed_yuan_per_mwh=fixed_cost_per_mwh,
        c_storage_mobile_yuan_per_mwh=fixed_cost_per_mwh * mobile_cost_premium_ratio,
        objective_mode="planning",
    )


def _prepare_case_data_under_shared_budget(
    data: CoreModelData,
    *,
    case: str,
    objective_mode: str,
    fixed_cost_per_mwh: float,
    mobile_cost_premium_ratio: float,
    hybrid_fixed_budget_share: float,
    budget_multiplier_relative_to_fixed_baseline: float,
    delay_windows: int | None = None,
    c_reconf_yuan_per_mwh: float | None = None,
) -> CoreModelData:
    if objective_mode != "planning":
        return make_case_data(
            data,
            "mobile_only" if case == "mobile_only_ideal" else case,
            delay_windows=0 if case == "mobile_only_ideal" else delay_windows,
            c_reconf_yuan_per_mwh=0.0 if case == "mobile_only_ideal" else c_reconf_yuan_per_mwh,
        )

    shared_budget = (
        fixed_cost_per_mwh
        * float(sum(data.fixed_capacity_mwh.values()))
        * budget_multiplier_relative_to_fixed_baseline
    )
    fixed_share_by_case = {
        "fixed_only": 1.0,
        "mobile_only": 0.0,
        "mobile_only_ideal": 0.0,
        "hybrid": hybrid_fixed_budget_share,
    }
    planning_data = prepare_same_budget_data(
        data,
        mobile_cost_premium_ratio=mobile_cost_premium_ratio,
        fixed_budget_share=fixed_share_by_case.get(case, hybrid_fixed_budget_share),
        budget_in_fixed_cost_units=shared_budget,
        fixed_cost_per_mwh=fixed_cost_per_mwh,
    )
    return make_case_data(
        planning_data,
        "mobile_only" if case == "mobile_only_ideal" else case,
        delay_windows=0 if case == "mobile_only_ideal" else delay_windows,
        c_reconf_yuan_per_mwh=0.0 if case == "mobile_only_ideal" else c_reconf_yuan_per_mwh,
    )


def _window_hours(data: CoreModelData) -> int:
    steps = max(1, len(data.hours) // max(1, len(data.windows)))
    return int(round(steps * data.dt_hours))


def _reconfiguration_volume(solution: Mapping[str, Mapping[tuple[str, int], float]], windows: Sequence[int], sites: Sequence[str]) -> float:
    if len(windows) <= 1:
        return 0.0
    return sum(
        abs(solution["x"][(site, w)] - solution["x"][(site, w - 1)])
        for site in sites
        for w in windows
        if w > 0
    )


def _stress_profiles(data: CoreModelData, solution: Mapping[str, Mapping[tuple[str, int], float]]) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float], list[str]]:
    sigma: dict[tuple[str, int], float] = {}
    sigma_window: dict[tuple[str, int], float] = {}
    leaders: list[str] = []
    hours_by_window: dict[int, list[int]] = {w: [] for w in data.windows}
    for hour in data.hours:
        hours_by_window[data.hour_to_window[hour]].append(hour)
    for site in data.sites:
        for hour in data.hours:
            sigma[(site, hour)] = (
                solution["p_grid"][(site, hour)] / (float(data.grid_limit_mw[(site, hour)]) + EPS)
                + solution["unserved"][(site, hour)] / (float(data.load_mw[(site, hour)]) + EPS)
            )

    for w in data.windows:
        scores: list[tuple[float, str]] = []
        for site in data.sites:
            total = sum(sigma[(site, t)] for t in hours_by_window[w])
            sigma_window[(site, w)] = total
            scores.append((total, site))
        scores.sort(reverse=True)
        leaders.append(scores[0][1])
    return sigma, sigma_window, leaders


def _planner_metrics(
    data: CoreModelData,
    solution: Mapping[str, Mapping[tuple[str, int], float]],
    *,
    theta: float = PEAK_THRESHOLD_THETA,
) -> tuple[float, float, float, float]:
    sigma, sigma_window, leaders = _stress_profiles(data, solution)

    sc_values: list[float] = []
    delay_ratios: list[float] = []
    hours_by_window: dict[int, list[int]] = {w: [] for w in data.windows}
    for hour in data.hours:
        hours_by_window[data.hour_to_window[hour]].append(hour)
    for w in data.windows:
        site_scores = sorted((sigma_window[(site, w)] for site in data.sites), reverse=True)
        total_score = sum(site_scores)
        sc_values.append(site_scores[0] / (total_score + EPS) if site_scores else 0.0)

        leader = leaders[w]
        peak = max(sigma[(leader, t)] for t in hours_by_window[w])
        duration_steps = sum(
            1
            for t in hours_by_window[w]
            if sigma[(leader, t)] >= theta * peak
        )
        d_peak = data.dt_hours * duration_steps
        delay_hours = _window_hours(data) * data.delay_windows
        delay_ratios.append(delay_hours / (d_peak + EPS))

    ht = (
        sum(1 for idx in range(1, len(leaders)) if leaders[idx] != leaders[idx - 1]) / max(1, len(leaders) - 1)
        if len(leaders) > 1
        else 0.0
    )
    sc_1 = mean(sc_values) if sc_values else 0.0
    r_delay = mean(delay_ratios) if delay_ratios else 0.0
    v_reconf = _reconfiguration_volume(solution, data.windows, data.sites)
    v_reconf_norm = v_reconf / (((len(data.windows) - 1) * data.m_total_mwh) + EPS) if len(data.windows) > 1 else 0.0
    return sc_1, ht, r_delay, v_reconf_norm


def _storage_cost_from_data(data: CoreModelData) -> float:
    return float(
        data.c_storage_fixed_yuan_per_mwh * sum(data.fixed_capacity_mwh.values())
        + data.c_storage_mobile_yuan_per_mwh * data.m_total_mwh
    )


def _diagnostic_note(diagnostics: CoreSolveDiagnostics) -> str:
    if diagnostics.case_outcome == "timeout_no_incumbent":
        lp_lb = (
            "n/a"
            if diagnostics.lp_relaxation_objective is None
            else f"{diagnostics.lp_relaxation_objective:.6f}"
        )
        return f"No feasible incumbent within {CASE_TIME_LIMIT_SECONDS:.0f}s; LP lower bound={lp_lb}."
    if diagnostics.case_outcome == "feasible_with_gap":
        gap = "n/a" if diagnostics.gap is None else f"{diagnostics.gap:.6%}"
        return f"Stopped before optimality proof; incumbent retained with gap={gap}."
    return ""


def _build_metrics(
    data: CoreModelData,
    scenario: ExperimentScenario,
    diagnostics: CoreSolveDiagnostics,
    solution: Mapping[str, Mapping[tuple[str, int], float]] | None,
) -> ScenarioMetrics:
    storage_cost = _storage_cost_from_data(data)
    if solution is None:
        nan = float("nan")
        return ScenarioMetrics(
            experiment_id=scenario.experiment_id,
            variant_id=scenario.variant_id,
            question=scenario.question,
            track=scenario.track,
            source=scenario.source,
            case=scenario.case,
            case_label=CASE_LABELS[scenario.case],
            solver_status=diagnostics.solver_status,
            termination_condition=diagnostics.termination_condition,
            case_outcome=diagnostics.case_outcome,
            best_incumbent=diagnostics.best_incumbent,
            best_bound=diagnostics.best_bound,
            gap=diagnostics.gap,
            elapsed_seconds=diagnostics.elapsed_seconds,
            lp_relaxation_objective=diagnostics.lp_relaxation_objective,
            timeline_mode=scenario.timeline_mode,
            reconf_hours=scenario.reconf_hours,
            delay_windows=scenario.delay_windows,
            fixed_total_mwh=float(sum(data.fixed_capacity_mwh.values())),
            mobile_total_mwh=float(data.m_total_mwh),
            c_reconf_yuan_per_mwh=float(data.c_reconf_yuan_per_mwh),
            C_total=nan,
            C_grid=nan,
            C_unserved=nan,
            C_reconf=nan,
            C_storage=storage_cost,
            C_total_with_storage=nan,
            total_unmet_load=nan,
            total_grid_purchase=nan,
            total_charging_energy=nan,
            total_discharging_energy=nan,
            total_reconfiguration_volume=nan,
            average_active_mobile_capacity_utilisation=nan,
            SC_1=nan,
            HT=nan,
            R_delay=nan,
            V_reconf_norm=nan,
            note=_diagnostic_note(diagnostics),
            metadata=dict(scenario.metadata),
        )

    sc_1, ht, r_delay, v_reconf_norm = _planner_metrics(data, solution)
    mobile_capacity_total = sum(
        solution["m_active"][(site, data.hour_to_window[t])] for site in data.sites for t in data.hours
    )
    average_util = (
        sum(solution["soc_mobile"].values()) / (mobile_capacity_total + EPS) if mobile_capacity_total > 0 else 0.0
    )
    return ScenarioMetrics(
        experiment_id=scenario.experiment_id,
        variant_id=scenario.variant_id,
        question=scenario.question,
        track=scenario.track,
        source=scenario.source,
        case=scenario.case,
        case_label=CASE_LABELS[scenario.case],
        solver_status=diagnostics.solver_status,
        termination_condition=diagnostics.termination_condition,
        case_outcome=diagnostics.case_outcome,
        best_incumbent=diagnostics.best_incumbent,
        best_bound=diagnostics.best_bound,
        gap=diagnostics.gap,
        elapsed_seconds=diagnostics.elapsed_seconds,
        lp_relaxation_objective=diagnostics.lp_relaxation_objective,
        timeline_mode=scenario.timeline_mode,
        reconf_hours=scenario.reconf_hours,
        delay_windows=scenario.delay_windows,
        fixed_total_mwh=float(sum(data.fixed_capacity_mwh.values())),
        mobile_total_mwh=float(data.m_total_mwh),
        c_reconf_yuan_per_mwh=float(data.c_reconf_yuan_per_mwh),
        C_total=float(solution["costs"]["C_total"]),
        C_grid=float(solution["costs"]["C_grid"]),
        C_unserved=float(solution["costs"]["C_unserved"]),
        C_reconf=float(solution["costs"]["C_reconf"]),
        C_storage=float(solution["costs"]["C_storage"]),
        C_total_with_storage=float(solution["costs"]["C_total_with_storage"]),
        total_unmet_load=float(sum(solution["unserved"].values()) * data.dt_hours),
        total_grid_purchase=float(sum(solution["p_grid"].values()) * data.dt_hours),
        total_charging_energy=float(sum(solution["p_ch"].values()) * data.dt_hours),
        total_discharging_energy=float(sum(solution["p_dis"].values()) * data.dt_hours),
        total_reconfiguration_volume=float(_reconfiguration_volume(solution, data.windows, data.sites)),
        average_active_mobile_capacity_utilisation=float(average_util),
        SC_1=float(sc_1),
        HT=float(ht),
        R_delay=float(r_delay),
        V_reconf_norm=float(v_reconf_norm),
        note=_diagnostic_note(diagnostics),
        metadata=dict(scenario.metadata),
    )


def _solve_scenario_detail_inprocess(
    data: CoreModelData,
    scenario: ExperimentScenario,
    *,
    solver_name: str,
    include_solution: bool,
) -> tuple[ScenarioMetrics, Mapping[str, Mapping[tuple[str, int], float]] | None]:
    diagnostics, solution = solve_core_model_with_diagnostics(
        data,
        solver_name=solver_name,
        time_limit_seconds=CASE_TIME_LIMIT_SECONDS,
        lp_time_limit_seconds=LP_RELAXATION_TIME_LIMIT_SECONDS,
    )
    metrics = _build_metrics(data, scenario, diagnostics, solution)
    return metrics, solution if include_solution and solution is not None else None


def _scenario_log_label(data: CoreModelData, scenario: ExperimentScenario) -> str:
    return (
        f"{scenario.experiment_id}/{scenario.variant_id}/{scenario.case}"
        f" sites={len(data.sites)} hours={len(data.hours)} windows={len(data.windows)}"
        f" delay={scenario.delay_windows} reconf={scenario.reconf_hours}h"
    )


def _format_case_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _solve_scenario_via_worker(
    data: CoreModelData,
    scenario: ExperimentScenario,
    *,
    solver_name: str,
    include_solution: bool,
) -> dict[str, object]:
    worker_script = Path(__file__).with_name("solve_case_worker.py")
    with tempfile.TemporaryDirectory(prefix="eep_solve_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.pkl"
        output_path = Path(tmp_dir) / "output.pkl"
        input_path.write_bytes(
            pickle.dumps(
                {
                    "data": serialize_core_model_data(data),
                    "scenario": serialize_experiment_scenario(scenario),
                    "solver_name": solver_name,
                    "include_solution": include_solution,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        completed = subprocess.run(
            [sys.executable, str(worker_script), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=CASE_TIME_LIMIT_SECONDS + WORKER_TIMEOUT_BUFFER_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"solver worker exited with code {completed.returncode}; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        if not output_path.exists():
            raise RuntimeError("solver worker exited without writing output payload")
        return pickle.loads(output_path.read_bytes())


def solve_scenario_detail(
    data: CoreModelData,
    scenario: ExperimentScenario,
    *,
    solver_name: str,
    include_solution: bool = True,
) -> tuple[ScenarioMetrics, Mapping[str, Mapping[tuple[str, int], float]] | None]:
    scenario_label = _scenario_log_label(data, scenario)
    print(f"[case] start {scenario_label} solver={solver_name}", flush=True)
    started_at = monotonic()
    try:
        result = _solve_scenario_via_worker(
            data,
            scenario,
            solver_name=solver_name,
            include_solution=include_solution,
        )
        metrics = deserialize_scenario_metrics(result["metrics"])
        solution = result.get("solution")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"solver worker exceeded hard timeout for {scenario_label} after "
            f"{CASE_TIME_LIMIT_SECONDS + WORKER_TIMEOUT_BUFFER_SECONDS:.0f}s"
        ) from exc
    except Exception as exc:
        print(
            f"[case] worker fallback {scenario_label} reason={exc.__class__.__name__}: {exc}",
            flush=True,
        )
        metrics, solution = _solve_scenario_detail_inprocess(
            data,
            scenario,
            solver_name=solver_name,
            include_solution=include_solution,
        )
    elapsed = monotonic() - started_at
    print(
        "[case] done "
        f"{scenario_label} status={metrics.solver_status}/{metrics.termination_condition}"
        f" outcome={metrics.case_outcome}"
        f" incumbent={_format_case_metric(metrics.best_incumbent)}"
        f" bound={_format_case_metric(metrics.best_bound)}"
        f" gap={_format_case_metric(metrics.gap)}"
        f" lp_lb={_format_case_metric(metrics.lp_relaxation_objective)}"
        f" elapsed={elapsed:.1f}s",
        flush=True,
    )
    return metrics, solution


def solve_scenario(data: CoreModelData, scenario: ExperimentScenario, *, solver_name: str) -> ScenarioMetrics:
    metrics, _solution = solve_scenario_detail(
        data,
        scenario,
        solver_name=solver_name,
        include_solution=False,
    )
    return metrics


def _run_scenario_requests(
    requests: Sequence[ScenarioSolveRequest],
) -> list[tuple[ScenarioMetrics, Mapping[str, Mapping[tuple[str, int], float]] | None]]:
    if len(requests) <= 1 or CASE_SOLVE_WORKERS <= 1:
        return [
            solve_scenario_detail(
                request.data,
                request.scenario,
                solver_name=request.solver_name,
                include_solution=request.include_solution,
            )
            for request in requests
        ]

    ordered_results: list[tuple[ScenarioMetrics, Mapping[str, Mapping[tuple[str, int], float]] | None] | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=min(CASE_SOLVE_WORKERS, len(requests))) as executor:
        futures = [
            executor.submit(
                solve_scenario_detail,
                request.data,
                request.scenario,
                solver_name=request.solver_name,
                include_solution=request.include_solution,
            )
            for request in requests
        ]
        for idx, future in enumerate(futures):
            ordered_results[idx] = future.result()

    return [result for result in ordered_results if result is not None]


def _attach_decomposition(rows: list[ScenarioMetrics]) -> None:
    grouped: dict[tuple[str, str], dict[str, ScenarioMetrics]] = defaultdict(dict)
    for row in rows:
        grouped[(row.experiment_id, row.variant_id)][row.case] = row

    for group in grouped.values():
        fixed_row = group.get("fixed_only")
        mobile_row = group.get("mobile_only")
        ideal_row = group.get("mobile_only_ideal")
        if fixed_row and mobile_row and ideal_row:
            b_reuse = fixed_row.C_total - ideal_row.C_total
            l_friction = mobile_row.C_total - ideal_row.C_total
            v_mob_net = fixed_row.C_total - mobile_row.C_total
            for row in (fixed_row, mobile_row, ideal_row):
                row.B_reuse = b_reuse
                row.L_friction = l_friction
                row.V_mob_net = v_mob_net


def _rows_from_metrics(metrics: Sequence[ScenarioMetrics]) -> list[dict[str, str | int | float]]:
    return [metric.to_row() for metric in metrics]


def serialize_scenario_metrics(metrics: ScenarioMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["metadata"] = dict(metrics.metadata)
    return payload


def deserialize_scenario_metrics(payload: Mapping[str, object]) -> ScenarioMetrics:
    return ScenarioMetrics(
        experiment_id=str(payload["experiment_id"]),
        variant_id=str(payload["variant_id"]),
        question=str(payload["question"]),
        track=str(payload["track"]),
        source=str(payload["source"]),
        case=str(payload["case"]),
        case_label=str(payload["case_label"]),
        solver_status=str(payload["solver_status"]),
        termination_condition=str(payload["termination_condition"]),
        case_outcome=str(payload.get("case_outcome", "")),
        best_incumbent=None if payload.get("best_incumbent") is None else float(payload["best_incumbent"]),
        best_bound=None if payload.get("best_bound") is None else float(payload["best_bound"]),
        gap=None if payload.get("gap") is None else float(payload["gap"]),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        lp_relaxation_objective=(
            None if payload.get("lp_relaxation_objective") is None else float(payload["lp_relaxation_objective"])
        ),
        timeline_mode=str(payload["timeline_mode"]),
        reconf_hours=int(payload["reconf_hours"]),
        delay_windows=int(payload["delay_windows"]),
        fixed_total_mwh=float(payload["fixed_total_mwh"]),
        mobile_total_mwh=float(payload["mobile_total_mwh"]),
        c_reconf_yuan_per_mwh=float(payload["c_reconf_yuan_per_mwh"]),
        C_total=float(payload["C_total"]),
        C_grid=float(payload["C_grid"]),
        C_unserved=float(payload["C_unserved"]),
        C_reconf=float(payload["C_reconf"]),
        C_storage=float(payload["C_storage"]),
        C_total_with_storage=float(payload["C_total_with_storage"]),
        total_unmet_load=float(payload["total_unmet_load"]),
        total_grid_purchase=float(payload["total_grid_purchase"]),
        total_charging_energy=float(payload["total_charging_energy"]),
        total_discharging_energy=float(payload["total_discharging_energy"]),
        total_reconfiguration_volume=float(payload["total_reconfiguration_volume"]),
        average_active_mobile_capacity_utilisation=float(payload["average_active_mobile_capacity_utilisation"]),
        SC_1=float(payload["SC_1"]),
        HT=float(payload["HT"]),
        R_delay=float(payload["R_delay"]),
        V_reconf_norm=float(payload["V_reconf_norm"]),
        B_reuse=None if payload.get("B_reuse") is None else float(payload["B_reuse"]),
        L_friction=None if payload.get("L_friction") is None else float(payload["L_friction"]),
        V_mob_net=None if payload.get("V_mob_net") is None else float(payload["V_mob_net"]),
        note=str(payload.get("note", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def _group_case_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment_id"]), str(row["variant_id"]))].append(dict(row))
    return grouped


def _summarize_variants(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str | int | float]]:
    summaries: list[dict[str, str | int | float]] = []
    grouped = _group_case_rows(rows)
    for (_experiment_id, variant_id), variant_rows in sorted(grouped.items()):
        if not variant_rows or any(not _is_case_conclusion_ready(row) for row in variant_rows):
            continue
        by_case = {str(row["case"]): row for row in variant_rows}
        summary: dict[str, str | int | float] = {
            "experiment_id": str(variant_rows[0]["experiment_id"]),
            "variant_id": variant_id,
            "winner_case": _winner_for_rows(variant_rows),
            "winner_label": _winner_name(_winner_for_rows(variant_rows)),
            "SC_1": float(by_case.get("mobile_only", by_case.get("no_storage", variant_rows[0]))["SC_1"]),
            "HT": float(by_case.get("mobile_only", by_case.get("no_storage", variant_rows[0]))["HT"]),
            "R_delay": float(by_case.get("mobile_only", variant_rows[0])["R_delay"]),
            "V_reconf_norm": float(by_case.get("mobile_only", variant_rows[0])["V_reconf_norm"]),
            "max_gap": max((float(row["gap"]) for row in variant_rows if row.get("gap") is not None), default=float("nan")),
            "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in variant_rows),
        }
        for row in variant_rows:
            case = str(row["case"])
            summary[f"C_total__{case}"] = float(row["C_total"])
            summary[f"unmet__{case}"] = float(row["total_unmet_load"])
        if "fixed_only" in by_case and "mobile_only" in by_case:
            summary["V_mob_net"] = float(by_case["fixed_only"]["C_total"]) - float(by_case["mobile_only"]["C_total"])
            summary["fixed_total_mwh"] = float(by_case["fixed_only"]["fixed_total_mwh"])
            summary["mobile_total_mwh"] = float(by_case["mobile_only"]["mobile_total_mwh"])
        if "hybrid" in by_case:
            summary["hybrid_fixed_total_mwh"] = float(by_case["hybrid"]["fixed_total_mwh"])
            summary["hybrid_mobile_total_mwh"] = float(by_case["hybrid"]["mobile_total_mwh"])
        if "mobile_only" in by_case:
            summary["c_reconf_yuan_per_mwh"] = float(by_case["mobile_only"]["c_reconf_yuan_per_mwh"])
        decomposition_source = next(
            (
                row
                for row in variant_rows
                if row.get("B_reuse") is not None and row.get("L_friction") is not None
            ),
            None,
        )
        if decomposition_source is not None:
            summary["B_reuse"] = float(decomposition_source["B_reuse"])
            summary["L_friction"] = float(decomposition_source["L_friction"])
            summary["V_mob_net"] = float(decomposition_source["V_mob_net"])
        for key, value in variant_rows[0].items():
            if key not in summary and key not in {"case", "case_label", "solver_status", "termination_condition", "case_outcome", "best_incumbent", "best_bound", "gap", "elapsed_seconds", "lp_relaxation_objective", "C_total", "C_grid", "C_unserved", "C_reconf", "C_storage", "C_total_with_storage", "total_unmet_load", "total_grid_purchase", "total_charging_energy", "total_discharging_energy", "total_reconfiguration_volume", "average_active_mobile_capacity_utilisation", "note"}:
                summary.setdefault(key, value)
        summaries.append(summary)
    return summaries


def _plot_cost_breakdown(rows: Sequence[Mapping[str, object]], path: Path, *, title: str) -> None:
    ordered = sorted(rows, key=lambda row: CASE_ORDER.index(str(row["case"])))
    labels = [CASE_LABELS[str(row["case"])] for row in ordered]
    grid = [float(row["C_grid"]) for row in ordered]
    unserved = [float(row["C_unserved"]) for row in ordered]
    reconf = [float(row["C_reconf"]) for row in ordered]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, grid, label="C_grid", color="#1f77b4")
    plt.bar(labels, unserved, bottom=grid, label="C_unserved", color="#d95f02")
    plt.bar(labels, reconf, bottom=[grid[idx] + unserved[idx] for idx in range(len(grid))], label="C_reconf", color="#2ca02c")
    plt.ylabel("Cost")
    plt.xticks(rotation=15, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_placeholder(path: Path, *, title: str, message: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_dispatch_from_solution(
    data: CoreModelData,
    solution: Mapping[str, Mapping[tuple[str, int], float]],
    path: Path,
) -> None:
    hours = list(data.hours)
    corridor_load = [sum(float(data.load_mw[(site, t)]) for site in data.sites) for t in hours]
    corridor_grid = [sum(solution["p_grid"][(site, t)] for site in data.sites) for t in hours]
    corridor_dis = [sum(solution["p_dis"][(site, t)] for site in data.sites) for t in hours]
    corridor_ch = [sum(solution["p_ch"][(site, t)] for site in data.sites) for t in hours]
    corridor_unserved = [sum(solution["unserved"][(site, t)] for site in data.sites) for t in hours]

    plt.figure(figsize=(12, 4.8))
    plt.plot(hours, corridor_load, label="Load", color="#111827")
    plt.plot(hours, corridor_grid, label="Grid", color="#1f77b4")
    plt.plot(hours, corridor_dis, label="Discharge", color="#2ca02c")
    plt.plot(hours, corridor_ch, label="Charge", color="#ff7f0e")
    plt.plot(hours, corridor_unserved, label="Unserved", color="#d62728")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("Baseline Corridor Dispatch")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_allocation_heatmap_from_solution(
    data: CoreModelData,
    case: str,
    solution: Mapping[str, Mapping[tuple[str, int], float]],
    path: Path,
) -> None:
    matrix = [
        [solution["x"][(site, w)] for w in data.windows]
        for site in data.sites
    ]
    plt.figure(figsize=(8, 4.5))
    plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Allocated Mobile Capacity (MWh)")
    plt.yticks(range(len(data.sites)), data.sites)
    plt.xticks(range(len(data.windows)), data.windows)
    plt.xlabel("Window")
    plt.ylabel("Site")
    plt.title(f"Allocation Heatmap: {CASE_LABELS[case]}")
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_metric_panels(summary_rows: Sequence[Mapping[str, object]], path: Path, *, x_key: str, x_label: str, title: str, metrics: Sequence[tuple[str, str]]) -> None:
    ordered = sorted(summary_rows, key=lambda row: float(row[x_key]))
    x_values = [float(row[x_key]) for row in ordered]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    winners = [str(row["winner_case"]) for row in ordered]
    for idx in range(1, len(winners)):
        if winners[idx] != winners[idx - 1]:
            for ax in axes:
                ax.axvline(_line_midpoint(x_values, idx), color="#9ca3af", linestyle="--", linewidth=1)

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for case in CASE_ORDER:
            y_values = [float(row.get(f"{metric_key}__{case}", row.get(metric_key, 0.0))) for row in ordered]
            ax.plot(x_values, y_values, marker="o", label=CASE_LABELS[case], color=WINNER_COLORS[case])
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_simple_line(summary_rows: Sequence[Mapping[str, object]], path: Path, *, x_key: str, y_key: str, x_label: str, y_label: str, title: str) -> None:
    ordered = sorted(summary_rows, key=lambda row: float(row[x_key]))
    x_values = [float(row[x_key]) for row in ordered]
    y_values = [float(row[y_key]) for row in ordered]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, marker="o", color="#d95f02")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_categorical_bar(summary_rows: Sequence[Mapping[str, object]], path: Path, *, x_key: str, y_key: str, x_label: str, y_label: str, title: str) -> None:
    labels = [str(row[x_key]) for row in summary_rows]
    values = [float(row[y_key]) for row in summary_rows]
    plt.figure(figsize=(8.5, 4.5))
    plt.bar(labels, values, color="#1f77b4")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axhline(0.0, color="#6b7280", linewidth=1)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_schedule_examples(configs: Sequence[SyntheticCorridorConfig], path: Path) -> None:
    fig, axes = plt.subplots(len(configs), 1, figsize=(9, 2.8 * len(configs)), sharex=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, config in zip(axes, configs):
        leaders = _route_for_turnover(config.windows, config.residence_windows)
        matrix = [[1.0 if site == leaders[w] else 0.0 for w in range(config.windows)] for site in SYNTHETIC_SITES]
        ax.imshow(matrix, aspect="auto", cmap="Greys")
        ax.set_yticks(range(len(SYNTHETIC_SITES)), SYNTHETIC_SITES)
        ax.set_title(f"Hotspot Schedule Example: residence={config.residence_windows}")
    axes[-1].set_xticks(range(configs[0].windows), range(configs[0].windows))
    axes[-1].set_xlabel("Window")
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_winner_heatmap(summary_rows: Sequence[Mapping[str, object]], path: Path, *, x_key: str, y_key: str, x_label: str, y_label: str, title: str) -> None:
    if not summary_rows:
        _plot_placeholder(path, title=title, message="No conclusion-ready variants were available.")
        return
    x_values = sorted({float(row[x_key]) for row in summary_rows})
    y_values = sorted({float(row[y_key]) for row in summary_rows})
    matrix = [[WINNER_CODE["no_storage"] for _ in x_values] for _ in y_values]
    for row in summary_rows:
        x_idx = x_values.index(float(row[x_key]))
        y_idx = y_values.index(float(row[y_key]))
        matrix[y_idx][x_idx] = WINNER_CODE[str(row["winner_case"])]
    cmap = ListedColormap([WINNER_COLORS[WINNER_NAME[idx]] for idx in range(len(WINNER_NAME))])
    plt.figure(figsize=(8, 5.2))
    plt.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
    plt.xticks(range(len(x_values)), [str(int(x)) if x.is_integer() else f"{x:.2f}" for x in x_values])
    plt.yticks(range(len(y_values)), [str(int(y)) if y.is_integer() else f"{y:.2f}" for y in y_values])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=WINNER_COLORS[case]) for case in CASE_ORDER]
    plt.legend(handles, [CASE_LABELS[case] for case in CASE_ORDER], fontsize=8, loc="upper right")
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_value_heatmap(summary_rows: Sequence[Mapping[str, object]], path: Path, *, x_key: str, y_key: str, value_key: str, x_label: str, y_label: str, title: str) -> None:
    if not summary_rows:
        _plot_placeholder(path, title=title, message="No conclusion-ready variants were available.")
        return
    x_values = sorted({float(row[x_key]) for row in summary_rows})
    y_values = sorted({float(row[y_key]) for row in summary_rows})
    matrix = [[0.0 for _ in x_values] for _ in y_values]
    for row in summary_rows:
        x_idx = x_values.index(float(row[x_key]))
        y_idx = y_values.index(float(row[y_key]))
        matrix[y_idx][x_idx] = float(row.get(value_key, 0.0))
    plt.figure(figsize=(8, 5.2))
    plt.imshow(matrix, aspect="auto", cmap="RdYlGn", origin="lower")
    plt.colorbar(label=value_key)
    plt.xticks(range(len(x_values)), [str(int(x)) if x.is_integer() else f"{x:.2f}" for x in x_values])
    plt.yticks(range(len(y_values)), [str(int(y)) if y.is_integer() else f"{y:.2f}" for y in y_values])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_decomposition(summary_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    labels = [str(row["variant_id"]) for row in summary_rows]
    b_values = [float(row["B_reuse"]) for row in summary_rows]
    l_values = [float(row["L_friction"]) for row in summary_rows]
    v_values = [float(row["V_mob_net"]) for row in summary_rows]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, b_values, label="B_reuse", color="#2ca02c")
    plt.bar(labels, [-value for value in l_values], label="-L_friction", color="#d62728")
    plt.plot(labels, v_values, color="#111827", marker="o", label="V_mob_net")
    plt.axhline(0.0, color="#6b7280", linewidth=1)
    plt.ylabel("Cost Difference")
    plt.title("Mechanism Decomposition")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_decomposition_scatter(summary_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for row in summary_rows:
        winner = str(row["winner_case"])
        plt.scatter(float(row["B_reuse"]), float(row["L_friction"]), color=WINNER_COLORS[winner], s=90)
        plt.text(float(row["B_reuse"]) + 0.01, float(row["L_friction"]) + 0.01, str(row["variant_id"]), fontsize=8)
    plt.xlabel("B_reuse")
    plt.ylabel("L_friction")
    plt.title("Reuse Benefit vs Mobility Friction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_scatter_grid(summary_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    metrics = [("SC_1", "SC_1"), ("HT", "HT"), ("R_delay", "R_delay"), ("V_reconf_norm", "V_reconf_norm")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (x_key, title) in zip(axes.flatten(), metrics):
        for row in summary_rows:
            winner = str(row["winner_case"])
            ax.scatter(float(row[x_key]), float(row["V_mob_net"]), color=WINNER_COLORS[winner], s=45)
        ax.set_xlabel(title)
        ax.set_ylabel("V_mob_net")
        ax.grid(alpha=0.3)
    fig.suptitle("Planner-Facing Indices vs Net Mobility Value")
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _create_memo(
    *,
    experiment_id: str,
    question: str,
    raw_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    figure_paths: Sequence[Path],
    notes: Sequence[str],
) -> str:
    status = "clean" if raw_rows or summary_rows else "failed"
    top_lines = [
        f"# {experiment_id}",
        "",
        f"- Question: {question}",
        f"- Status: {status}",
        f"- Raw rows: {len(raw_rows)}",
        f"- Summary rows: {len(summary_rows)}",
        "",
        "## Key Findings",
    ]
    if summary_rows:
        top_lines.append(f"- Main winner pattern: {', '.join(sorted({str(row['winner_label']) for row in summary_rows if 'winner_label' in row}))}")
    for note in notes:
        top_lines.append(f"- {note}")
    top_lines.extend(["", "## Figures"])
    for figure_path in figure_paths:
        top_lines.append(f"- {figure_path.as_posix()}")
    top_lines.extend(["", "## Report Notes"])
    top_lines.append("- Use the winner regions, V_mob_net sign changes, and decomposition values directly in the Results chapter discussion.")
    return "\n".join(top_lines) + "\n"


def _finalize_experiment(
    base_dir: Path,
    experiment_id: str,
    question: str,
    metrics: Sequence[ScenarioMetrics],
    summary_rows: Sequence[Mapping[str, object]],
    figure_paths: Sequence[Path],
    notes: Sequence[str],
) -> ExperimentArtifacts:
    raw_path = base_dir / "sim" / "outputs" / f"{experiment_id}_raw.csv"
    summary_path = base_dir / "sim" / "outputs" / f"{experiment_id}_summary.csv"
    memo_path = base_dir / "reports" / f"experiment_{experiment_id.lower()}.md"
    raw_rows = _rows_from_metrics(metrics)
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_text(
        memo_path,
        _create_memo(
            experiment_id=experiment_id,
            question=question,
            raw_rows=raw_rows,
            summary_rows=summary_rows,
            figure_paths=figure_paths,
            notes=notes,
        ),
    )
    failed = [row for row in raw_rows if not _is_case_conclusion_ready(row)]
    status = "clean" if not failed else "ambiguous"
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        question=question,
        raw_csv=raw_path,
        summary_csv=summary_path,
        memo_md=memo_path,
        figure_paths=list(figure_paths),
        raw_rows=raw_rows,
        summary_rows=[dict(row) for row in summary_rows],
        status=status,
    )


def _load_existing_experiment_artifact(base_dir: Path, experiment_id: str) -> ExperimentArtifacts | None:
    raw_path = base_dir / "sim" / "outputs" / f"{experiment_id}_raw.csv"
    summary_path = base_dir / "sim" / "outputs" / f"{experiment_id}_summary.csv"
    memo_path = base_dir / "reports" / f"experiment_{experiment_id.lower()}.md"
    if not raw_path.exists() or not summary_path.exists() or not memo_path.exists():
        return None

    memo_lines = memo_path.read_text(encoding="utf-8").splitlines()
    question = ""
    memo_status = "clean"
    figure_paths: list[Path] = []
    in_figures = False
    for line in memo_lines:
        if line.startswith("- Question: "):
            question = line[len("- Question: ") :].strip()
        elif line.startswith("- Status: "):
            memo_status = line[len("- Status: ") :].strip()
        elif line.strip() == "## Figures":
            in_figures = True
        elif in_figures and line.startswith("## "):
            in_figures = False
        elif in_figures and line.startswith("- "):
            figure_paths.append(_normalize_existing_path(line[2:]))

    raw_rows = _read_csv_rows(raw_path)
    summary_rows = _read_csv_rows(summary_path)
    failed = [row for row in raw_rows if not _is_case_conclusion_ready(row)]
    status = memo_status or ("clean" if not failed else "ambiguous")
    if failed and status == "clean":
        status = "ambiguous"

    return ExperimentArtifacts(
        experiment_id=experiment_id,
        question=question,
        raw_csv=raw_path,
        summary_csv=summary_path,
        memo_md=memo_path,
        figure_paths=figure_paths,
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        status=status,
    )


def run_e0(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Do the real and synthetic sanity cases solve cleanly and produce sensible movement?"
    requests: list[ScenarioSolveRequest] = []

    real_data = load_real_data(base_dir, defaults)
    for case in ["hybrid"]:
        case_data = _prepare_case_data_under_shared_budget(
            real_data,
            case=case,
            objective_mode=defaults.objective_mode,
            fixed_cost_per_mwh=defaults.fixed_cost_per_mwh,
            mobile_cost_premium_ratio=defaults.mobile_cost_premium_ratio,
            hybrid_fixed_budget_share=defaults.hybrid_fixed_budget_share,
            budget_multiplier_relative_to_fixed_baseline=defaults.budget_multiplier_relative_to_fixed_baseline,
        )
        scenario = ExperimentScenario(
            experiment_id="E0",
            variant_id="real_baseline",
            question=question,
            track="real",
            source="g2_representative_12x24",
            case=case,
            timeline_mode=defaults.timeline_mode,
            reconf_hours=defaults.reconf_hours,
            delay_windows=defaults.delay_windows,
            c_reconf_yuan_per_mwh=defaults.c_reconf_yuan_per_mwh,
        )
        requests.append(
            ScenarioSolveRequest(
                data=case_data,
                scenario=scenario,
                solver_name=defaults.solver,
            )
        )

    easy_cfg = SyntheticCorridorConfig(reconf_hours=24, windows=6, delay_windows=1, concentration=0.55, residence_windows=3, hotspot_extra_mw=1.2, grid_margin_frac=0.12)
    hard_cfg = SyntheticCorridorConfig(reconf_hours=24, windows=6, delay_windows=1, concentration=0.95, residence_windows=1, hotspot_extra_mw=1.9, grid_margin_frac=0.02)
    for variant_id, cfg in [("synthetic_easy", easy_cfg), ("synthetic_extreme", hard_cfg)]:
        case_data = _prepare_case_data_under_shared_budget(
            build_synthetic_corridor_data(cfg),
            case="mobile_only",
            objective_mode=defaults.objective_mode,
            fixed_cost_per_mwh=defaults.fixed_cost_per_mwh,
            mobile_cost_premium_ratio=defaults.mobile_cost_premium_ratio,
            hybrid_fixed_budget_share=defaults.hybrid_fixed_budget_share,
            budget_multiplier_relative_to_fixed_baseline=defaults.budget_multiplier_relative_to_fixed_baseline,
            delay_windows=cfg.delay_windows,
            c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
        )
        scenario = ExperimentScenario(
            experiment_id="E0",
            variant_id=variant_id,
            question=question,
            track="synthetic",
            source="synthetic_corridor",
            case="mobile_only",
            timeline_mode="synthetic_hourly",
            reconf_hours=cfg.reconf_hours,
            delay_windows=cfg.delay_windows,
            c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
            metadata={"concentration": cfg.concentration, "residence_windows": cfg.residence_windows},
        )
        requests.append(
            ScenarioSolveRequest(
                data=case_data,
                scenario=scenario,
                solver_name=defaults.solver,
            )
        )

    metrics = [metric for metric, _solution in _run_scenario_requests(requests)]

    raw_rows = _rows_from_metrics(metrics)
    fig_path = base_dir / "figures" / "experiments" / "e0_sanity_costs.png"
    plt.figure(figsize=(9, 4.5))
    plt.bar([str(row["variant_id"]) for row in raw_rows], [float(row["C_total"]) for row in raw_rows], color="#1f77b4")
    plt.ylabel("C_total")
    plt.title("E0 Sanity Cases")
    plt.tight_layout()
    _ensure_dir(fig_path.parent)
    plt.savefig(fig_path, dpi=180)
    plt.close()

    notes = [
        "Hybrid real baseline, easy synthetic mobile case, and extreme synthetic mobile case were all solved as the sanity triad.",
        "Use this memo to verify finite costs, bounded SOC, and visible reconfiguration volume before trusting scans.",
    ]
    return _finalize_experiment(base_dir, "E0", question, metrics, raw_rows, [fig_path], notes)


def run_case_bundle(
    *,
    experiment_id: str,
    question: str,
    track: str,
    source: str,
    variant_id: str,
    base_data: CoreModelData,
    timeline_mode: str,
    reconf_hours: int,
    delay_windows: int,
    c_reconf_yuan_per_mwh: float,
    solver_name: str,
    objective_mode: str = "planning",
    fixed_cost_per_mwh: float = 1.0,
    mobile_cost_premium_ratio: float = 1.5,
    hybrid_fixed_budget_share: float = 0.5,
    budget_multiplier_relative_to_fixed_baseline: float = 1.0,
    metadata: Mapping[str, str | int | float] | None = None,
    include_ideal_mobile: bool = False,
) -> list[ScenarioMetrics]:
    requests: list[ScenarioSolveRequest] = []
    for case in CASE_ORDER:
        scenario = ExperimentScenario(
            experiment_id=experiment_id,
            variant_id=variant_id,
            question=question,
            track=track,
            source=source,
            case=case,
            timeline_mode=timeline_mode,
            reconf_hours=reconf_hours,
            delay_windows=delay_windows,
            c_reconf_yuan_per_mwh=c_reconf_yuan_per_mwh,
            metadata=dict(metadata or {}),
        )
        case_data = _prepare_case_data_under_shared_budget(
            base_data,
            case=case,
            objective_mode=objective_mode,
            fixed_cost_per_mwh=fixed_cost_per_mwh,
            mobile_cost_premium_ratio=mobile_cost_premium_ratio,
            hybrid_fixed_budget_share=hybrid_fixed_budget_share,
            budget_multiplier_relative_to_fixed_baseline=budget_multiplier_relative_to_fixed_baseline,
            delay_windows=delay_windows,
            c_reconf_yuan_per_mwh=c_reconf_yuan_per_mwh,
        )
        requests.append(
            ScenarioSolveRequest(
                data=case_data,
                scenario=scenario,
                solver_name=solver_name,
            )
        )
    if include_ideal_mobile:
        scenario = ExperimentScenario(
            experiment_id=experiment_id,
            variant_id=variant_id,
            question=question,
            track=track,
            source=source,
            case="mobile_only_ideal",
            timeline_mode=timeline_mode,
            reconf_hours=reconf_hours,
            delay_windows=0,
            c_reconf_yuan_per_mwh=0.0,
            metadata=dict(metadata or {}),
        )
        ideal_data = _prepare_case_data_under_shared_budget(
            base_data,
            case="mobile_only_ideal",
            objective_mode=objective_mode,
            fixed_cost_per_mwh=fixed_cost_per_mwh,
            mobile_cost_premium_ratio=mobile_cost_premium_ratio,
            hybrid_fixed_budget_share=hybrid_fixed_budget_share,
            budget_multiplier_relative_to_fixed_baseline=budget_multiplier_relative_to_fixed_baseline,
            delay_windows=0,
            c_reconf_yuan_per_mwh=0.0,
        )
        requests.append(
            ScenarioSolveRequest(
                data=ideal_data,
                scenario=scenario,
                solver_name=solver_name,
            )
        )
    return [metric for metric, _solution in _run_scenario_requests(requests)]


def run_e1(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Under the shared-budget planning setting, how do no-storage, fixed-only, mobile-only, and hybrid compare?"
    data = load_real_data(base_dir, defaults)
    metrics: list[ScenarioMetrics] = []
    solved_cases: dict[str, tuple[CoreModelData, Mapping[str, Mapping[tuple[str, int], float]]]] = {}
    requests: list[ScenarioSolveRequest] = []
    case_data_by_case: dict[str, CoreModelData] = {}
    for case in CASE_ORDER:
        scenario = ExperimentScenario(
            experiment_id="E1",
            variant_id="baseline",
            question=question,
            track="real",
            source="g2_representative_12x24",
            case=case,
            timeline_mode=defaults.timeline_mode,
            reconf_hours=defaults.reconf_hours,
            delay_windows=defaults.delay_windows,
            c_reconf_yuan_per_mwh=defaults.c_reconf_yuan_per_mwh,
        )
        case_data = _prepare_case_data_under_shared_budget(
            data,
            case=case,
            objective_mode=defaults.objective_mode,
            fixed_cost_per_mwh=defaults.fixed_cost_per_mwh,
            mobile_cost_premium_ratio=defaults.mobile_cost_premium_ratio,
            hybrid_fixed_budget_share=defaults.hybrid_fixed_budget_share,
            budget_multiplier_relative_to_fixed_baseline=defaults.budget_multiplier_relative_to_fixed_baseline,
        )
        case_data_by_case[case] = case_data
        requests.append(
            ScenarioSolveRequest(
                data=case_data,
                scenario=scenario,
                solver_name=defaults.solver,
                include_solution=True,
            )
        )
    for request, (metric, solution) in zip(requests, _run_scenario_requests(requests)):
        metrics.append(metric)
        if solution is not None:
            solved_cases[request.scenario.case] = (case_data_by_case[request.scenario.case], solution)
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_paths = [
        base_dir / "figures" / "experiments" / "e1_cost_breakdown.png",
        base_dir / "figures" / "experiments" / "e1_dispatch.png",
        base_dir / "figures" / "experiments" / "e1_mobile_heatmap.png",
        base_dir / "figures" / "experiments" / "e1_hybrid_heatmap.png",
    ]
    _plot_cost_breakdown(raw_rows, figure_paths[0], title="E1 Baseline Cost Breakdown")
    if "hybrid" in solved_cases:
        _plot_dispatch_from_solution(solved_cases["hybrid"][0], solved_cases["hybrid"][1], figure_paths[1])
        _plot_allocation_heatmap_from_solution(solved_cases["hybrid"][0], "hybrid", solved_cases["hybrid"][1], figure_paths[3])
    else:
        _plot_placeholder(
            figure_paths[1],
            title="Baseline Corridor Dispatch",
            message="Hybrid case did not return a feasible incumbent within the hard time limit.",
        )
        _plot_placeholder(
            figure_paths[3],
            title="Allocation Heatmap: Case C: Hybrid",
            message="Hybrid case did not return a feasible incumbent within the hard time limit.",
        )
    if "mobile_only" in solved_cases:
        _plot_allocation_heatmap_from_solution(
            solved_cases["mobile_only"][0],
            "mobile_only",
            solved_cases["mobile_only"][1],
            figure_paths[2],
        )
    else:
        _plot_placeholder(
            figure_paths[2],
            title="Allocation Heatmap: Case B: Mobile Only",
            message="Mobile-only case did not return a feasible incumbent within the hard time limit.",
        )
    winner_label = str(summary_rows[0]["winner_label"]) if summary_rows else "inconclusive"
    notes = [
        f"Winner on baseline: {winner_label}",
        "All storage cases use the same storage-investment budget; hybrid is a budget split, not an add-on capacity case.",
    ]
    return _finalize_experiment(base_dir, "E1", question, metrics, summary_rows, figure_paths, notes)


def run_e2(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "How sensitive is mobile value to the reconfiguration interval under a shared storage budget?"
    metrics: list[ScenarioMetrics] = []
    for reconf_hours in [6, 12, 24, 48, 168]:
        data = load_real_data(base_dir, defaults, reconf_hours=reconf_hours)
        metrics.extend(
            run_case_bundle(
                experiment_id="E2",
                question=question,
                track="real",
                source="g2_representative_12x24",
                variant_id=f"reconf_{reconf_hours}",
                base_data=data,
                timeline_mode=defaults.timeline_mode,
                reconf_hours=reconf_hours,
                delay_windows=defaults.delay_windows,
                c_reconf_yuan_per_mwh=defaults.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"reconf_hours_scan": reconf_hours},
            )
        )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_path = base_dir / "figures" / "experiments" / "e2_interval_scan.png"
    _plot_metric_panels(
        summary_rows,
        figure_path,
        x_key="reconf_hours_scan",
        x_label="Reconfiguration Interval (hours)",
        title="E2 Reconfiguration Interval Scan",
        metrics=[
            ("C_total", "C_total"),
            ("unmet", "Unmet Load"),
        ],
    )
    notes = [
        "Dashed vertical markers indicate where the winning case changes across the reconfiguration interval scan.",
        "Use the sign of V_mob_net and the winner transitions to discuss mobility sensitivity.",
    ]
    return _finalize_experiment(base_dir, "E2", question, metrics, summary_rows, [figure_path], notes)


def run_e3(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "How much deployment delay kills the mobile advantage under a shared storage budget?"
    metrics: list[ScenarioMetrics] = []
    for delay_windows in [0, 1, 2, 3]:
        data = load_real_data(base_dir, defaults, delay_windows=delay_windows)
        metrics.extend(
            run_case_bundle(
                experiment_id="E3",
                question=question,
                track="real",
                source="g2_representative_12x24",
                variant_id=f"delay_{delay_windows}",
                base_data=data,
                timeline_mode=defaults.timeline_mode,
                reconf_hours=defaults.reconf_hours,
                delay_windows=delay_windows,
                c_reconf_yuan_per_mwh=defaults.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"delay_windows_scan": delay_windows, "delay_hours": delay_windows * defaults.reconf_hours},
            )
        )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_path = base_dir / "figures" / "experiments" / "e3_delay_scan.png"
    _plot_metric_panels(
        summary_rows,
        figure_path,
        x_key="delay_windows_scan",
        x_label="Deployment Delay (windows)",
        title="E3 Deployment Delay Scan",
        metrics=[
            ("C_total", "C_total"),
            ("unmet", "Unmet Load"),
        ],
    )
    notes = [
        "Delay is scanned in windows and also recorded in hours in the raw output.",
        "Use the first delay setting where mobile loses to fixed as the main threshold statement.",
    ]
    return _finalize_experiment(base_dir, "E3", question, metrics, summary_rows, [figure_path], notes)


def run_e4(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Under a shared storage budget, does mobile help more when stress is localized or distributed?"
    metrics: list[ScenarioMetrics] = []
    concentration_levels = [0.35, 0.5, 0.65, 0.8, 0.95]
    for concentration in concentration_levels:
        cfg = SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=1, concentration=concentration, residence_windows=1, hotspot_extra_mw=1.6, grid_margin_frac=0.05)
        metrics.extend(
            run_case_bundle(
                experiment_id="E4",
                question=question,
                track="synthetic",
                source="synthetic_concentration_scan",
                variant_id=f"conc_{concentration:.2f}",
                base_data=build_synthetic_corridor_data(cfg),
                timeline_mode="synthetic_hourly",
                reconf_hours=cfg.reconf_hours,
                delay_windows=cfg.delay_windows,
                c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"concentration": concentration, "concentration_level": concentration},
            )
        )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_path = base_dir / "figures" / "experiments" / "e4_concentration_scan.png"
    _plot_simple_line(
        summary_rows,
        figure_path,
        x_key="SC_1",
        y_key="V_mob_net",
        x_label="SC_1",
        y_label="V_mob_net",
        title="E4 Mobile Advantage vs Stress Concentration",
    )
    notes = [
        "Positive V_mob_net indicates mobile-only beats fixed-only.",
        "Use the SC_1 sweep to explain the spatial concentration driver.",
    ]
    return _finalize_experiment(base_dir, "E4", question, metrics, summary_rows, [figure_path], notes)


def run_e5(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Under a shared storage budget, does mobility win when the dominant stress site changes over time?"
    metrics: list[ScenarioMetrics] = []
    residence_levels = [4, 3, 2, 1]
    configs: list[SyntheticCorridorConfig] = []
    for residence in residence_levels:
        cfg = SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=1, concentration=0.85, residence_windows=residence, hotspot_extra_mw=1.6, grid_margin_frac=0.05)
        configs.append(cfg)
        metrics.extend(
            run_case_bundle(
                experiment_id="E5",
                question=question,
                track="synthetic",
                source="synthetic_turnover_scan",
                variant_id=f"turnover_{residence}",
                base_data=build_synthetic_corridor_data(cfg),
                timeline_mode="synthetic_hourly",
                reconf_hours=cfg.reconf_hours,
                delay_windows=cfg.delay_windows,
                c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"residence_windows": residence, "turnover_level": residence},
            )
        )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_paths = [
        base_dir / "figures" / "experiments" / "e5_turnover_scan.png",
        base_dir / "figures" / "experiments" / "e5_turnover_schedule_examples.png",
    ]
    _plot_simple_line(
        summary_rows,
        figure_paths[0],
        x_key="HT",
        y_key="V_mob_net",
        x_label="HT",
        y_label="V_mob_net",
        title="E5 Mobile Advantage vs Hotspot Turnover",
    )
    _plot_schedule_examples([configs[0], configs[-1]], figure_paths[1])
    notes = [
        "Lower residence_windows means higher hotspot turnover.",
        "Use the schedule examples to illustrate low-turnover and high-turnover regimes.",
    ]
    return _finalize_experiment(base_dir, "E5", question, metrics, summary_rows, figure_paths, notes)


def run_e6(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Under a shared storage budget, for a mobile resource with delay d and reconfiguration interval L, which intervention wins?"
    metrics: list[ScenarioMetrics] = []
    for delay_windows in [0, 1, 2, 3, 4]:
        for reconf_hours in [3, 6, 12, 24, 48]:
            cfg = SyntheticCorridorConfig(reconf_hours=reconf_hours, windows=6, delay_windows=delay_windows, concentration=0.9, residence_windows=1, hotspot_extra_mw=1.7, grid_margin_frac=0.04)
            metrics.extend(
                run_case_bundle(
                    experiment_id="E6",
                    question=question,
                    track="synthetic",
                    source="synthetic_delay_interval_map",
                    variant_id=f"d{delay_windows}_L{reconf_hours}",
                    base_data=build_synthetic_corridor_data(cfg),
                    timeline_mode="synthetic_hourly",
                    reconf_hours=reconf_hours,
                    delay_windows=delay_windows,
                    c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
                    solver_name=defaults.solver,
                    metadata={"delay_windows_map": delay_windows, "reconf_hours_map": reconf_hours},
                )
            )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_paths = [
        base_dir / "figures" / "experiments" / "e6_winner_heatmap.png",
        base_dir / "figures" / "experiments" / "e6_v_mob_net_heatmap.png",
    ]
    _plot_winner_heatmap(summary_rows, figure_paths[0], x_key="reconf_hours_map", y_key="delay_windows_map", x_label="Reconfiguration Interval (hours)", y_label="Delay (windows)", title="E6 Winner Map")
    _plot_value_heatmap(summary_rows, figure_paths[1], x_key="reconf_hours_map", y_key="delay_windows_map", value_key="V_mob_net", x_label="Reconfiguration Interval (hours)", y_label="Delay (windows)", title="E6 Net Mobility Value")
    notes = [
        "This is the primary regime-identification hero figure.",
        "Use the value heatmap to explain not just who wins, but by how much mobile beats fixed.",
    ]
    return _finalize_experiment(base_dir, "E6", question, metrics, summary_rows, figure_paths, notes)


def run_e7(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Under a shared storage budget, when does mobile storage become preferable across spatial concentration and hotspot movement?"
    metrics: list[ScenarioMetrics] = []
    concentration_levels = [0.35, 0.5, 0.65, 0.8, 0.95]
    residence_levels = [5, 4, 3, 2, 1]
    for concentration in concentration_levels:
        for residence in residence_levels:
            cfg = SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=1, concentration=concentration, residence_windows=residence, hotspot_extra_mw=1.6, grid_margin_frac=0.05)
            metrics.extend(
                run_case_bundle(
                    experiment_id="E7",
                    question=question,
                    track="synthetic",
                    source="synthetic_concentration_turnover_map",
                    variant_id=f"conc{concentration:.2f}_res{residence}",
                    base_data=build_synthetic_corridor_data(cfg),
                    timeline_mode="synthetic_hourly",
                    reconf_hours=cfg.reconf_hours,
                    delay_windows=cfg.delay_windows,
                    c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
                    solver_name=defaults.solver,
                    metadata={"concentration_map": concentration, "turnover_map": residence},
                )
            )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_paths = [
        base_dir / "figures" / "experiments" / "e7_winner_heatmap.png",
        base_dir / "figures" / "experiments" / "e7_v_mob_net_heatmap.png",
    ]
    _plot_winner_heatmap(summary_rows, figure_paths[0], x_key="concentration_map", y_key="turnover_map", x_label="Concentration", y_label="Residence Windows", title="E7 Winner Map")
    _plot_value_heatmap(summary_rows, figure_paths[1], x_key="concentration_map", y_key="turnover_map", value_key="V_mob_net", x_label="Concentration", y_label="Residence Windows", title="E7 Net Mobility Value")
    notes = [
        "Residence windows are the inverse turnover control: smaller values imply faster hotspot changes.",
        "This map supports the spatiotemporal mismatch narrative directly.",
    ]
    return _finalize_experiment(base_dir, "E7", question, metrics, summary_rows, figure_paths, notes)


def run_e8(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Why does mobile win or lose in representative scenarios under a shared storage budget?"
    metrics: list[ScenarioMetrics] = []
    scenario_defs = {
        "mobile_wins": SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=0, concentration=0.9, residence_windows=1, hotspot_extra_mw=1.8, grid_margin_frac=0.04),
        "near_boundary": SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=1, concentration=0.7, residence_windows=2, hotspot_extra_mw=1.5, grid_margin_frac=0.05),
        "fixed_wins": SyntheticCorridorConfig(reconf_hours=24, windows=8, delay_windows=3, concentration=0.45, residence_windows=4, hotspot_extra_mw=1.2, grid_margin_frac=0.08),
    }
    for variant_id, cfg in scenario_defs.items():
        metrics.extend(
            run_case_bundle(
                experiment_id="E8",
                question=question,
                track="synthetic",
                source="synthetic_mechanism_decomposition",
                variant_id=variant_id,
                base_data=build_synthetic_corridor_data(cfg),
                timeline_mode="synthetic_hourly",
                reconf_hours=cfg.reconf_hours,
                delay_windows=cfg.delay_windows,
                c_reconf_yuan_per_mwh=cfg.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"concentration": cfg.concentration, "residence_windows": cfg.residence_windows},
                include_ideal_mobile=True,
            )
        )
    _attach_decomposition(metrics)
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants([row for row in raw_rows if row["case"] != "mobile_only_ideal"])
    figure_paths = [
        base_dir / "figures" / "experiments" / "e8_decomposition.png",
        base_dir / "figures" / "experiments" / "e8_reuse_vs_friction.png",
    ]
    _plot_decomposition(summary_rows, figure_paths[0])
    _plot_decomposition_scatter(summary_rows, figure_paths[1])
    for row in summary_rows:
        identity_gap = abs(float(row["V_mob_net"]) - (float(row["B_reuse"]) - float(row["L_friction"])))
        if identity_gap > 1e-6:
            raise RuntimeError(f"E8 decomposition identity mismatch for {row['variant_id']}: {identity_gap}")
    notes = [
        "B_reuse, L_friction, and V_mob_net are reported for three representative regimes.",
        "This experiment supplies the mechanism-decomposition figure for the thesis.",
    ]
    return _finalize_experiment(base_dir, "E8", question, metrics, summary_rows, figure_paths, notes)


def _load_holiday_case(base_dir: Path, defaults: ExperimentDefaults, holiday_name: str) -> CoreModelData:
    full_data = load_real_data(base_dir, defaults, timeline_mode="expanded_8760")
    start, end = HOLIDAY_RANGES_2026[holiday_name]
    return _subset_data_for_dates(
        full_data,
        year=2026,
        start=start,
        end=end,
        reconf_hours=defaults.reconf_hours,
        dt_hours=1.0,
        delay_windows=defaults.delay_windows,
        initial_active_mode="equal",
        reconf_limit_mwh=2.0 * full_data.m_total_mwh,
    )


def run_e9(base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    question = "Do the shared-budget planning conclusions survive realistic perturbations?"
    metrics: list[ScenarioMetrics] = []

    holiday_stress = _load_holiday_case(base_dir, defaults, "spring_festival")
    corridor_order, archetype_by_service = _load_corridor_order_and_archetype(base_dir)
    holiday_stagger = _apply_staggered_holiday_profile(
        _load_holiday_case(base_dir, defaults, "national_day"),
        holiday_name="national_day",
        corridor_order=corridor_order,
        archetype_by_service=archetype_by_service,
        stagger_scale=1.0,
    )
    hetero_base = load_real_data(base_dir, defaults)
    site_month_multiplier = _normalize_site_multipliers(corridor_order, archetype_by_service)
    heterogeneity = _apply_site_month_multipliers(hetero_base, site_month_multiplier)

    variants = [
        ("holiday_stress", holiday_stress, "holiday_stress"),
        ("holiday_stagger", holiday_stagger, "holiday_stagger"),
        ("site_month_heterogeneity", heterogeneity, "site_month_heterogeneity"),
    ]
    for variant_id, data, source in variants:
        metrics.extend(
            run_case_bundle(
                experiment_id="E9",
                question=question,
                track="real",
                source=source,
                variant_id=variant_id,
                base_data=data,
                timeline_mode="mixed_real",
                reconf_hours=defaults.reconf_hours,
                delay_windows=defaults.delay_windows,
                c_reconf_yuan_per_mwh=defaults.c_reconf_yuan_per_mwh,
                solver_name=defaults.solver,
                metadata={"robustness_case": variant_id},
            )
        )
    raw_rows = _rows_from_metrics(metrics)
    summary_rows = _summarize_variants(raw_rows)
    figure_path = base_dir / "figures" / "experiments" / "e9_robustness_v_mob_net.png"
    _plot_categorical_bar(
        summary_rows,
        figure_path,
        x_key="variant_id",
        y_key="V_mob_net",
        x_label="Robustness Case",
        y_label="V_mob_net",
        title="E9 Robustness Summary",
    )
    notes = [
        "Use this table to check whether the baseline ranking is consistent under holiday and heterogeneity stress.",
        "Robustness alignment should be discussed relative to the regime maps from E6 and E7.",
    ]
    return _finalize_experiment(base_dir, "E9", question, metrics, summary_rows, [figure_path], notes)


def _find_threshold_rule(summary_rows: Sequence[Mapping[str, object]]) -> str:
    if not summary_rows:
        return "No stable threshold rule was evaluated because no summary rows were produced."
    sc_values = [float(row["SC_1"]) for row in summary_rows]
    delay_values = [float(row["R_delay"]) for row in summary_rows]
    sc_cut = mean(sc_values)
    delay_cut = mean(delay_values)
    favorable = [
        row
        for row in summary_rows
        if float(row.get("V_mob_net", 0.0)) > 0 and float(row["SC_1"]) >= sc_cut and float(row["R_delay"]) <= delay_cut
    ]
    if favorable:
        share = len(favorable) / max(1, sum(1 for row in summary_rows if float(row["SC_1"]) >= sc_cut and float(row["R_delay"]) <= delay_cut))
        return f"If SC_1 >= {sc_cut:.3f} and R_delay <= {delay_cut:.3f}, mobile is favorable in {share:.0%} of matching scenarios."
    return "No stable threshold rule emerged from the current E10 sample; report the scatter structure as explanatory but not predictive."


def run_e10(base_dir: Path, prior_artifacts: Sequence[ExperimentArtifacts]) -> ExperimentArtifacts:
    question = "Do SC_k, HT, R_delay, and V_reconf_norm help explain who wins?"
    summary_rows: list[dict[str, str | int | float]] = []
    for artifact in prior_artifacts:
        if artifact.experiment_id in {"E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"}:
            summary_rows.extend([dict(row) for row in artifact.summary_rows])
    figure_path = base_dir / "figures" / "experiments" / "e10_indices_scatter.png"
    _plot_scatter_grid(summary_rows, figure_path)
    threshold_rule = _find_threshold_rule(summary_rows)
    rows = [
        {
            "experiment_id": "E10",
            "variant_id": row.get("variant_id", ""),
            "question": question,
            "winner_case": row.get("winner_case", ""),
            "winner_label": row.get("winner_label", ""),
            "SC_1": row.get("SC_1", 0.0),
            "HT": row.get("HT", 0.0),
            "R_delay": row.get("R_delay", 0.0),
            "V_reconf_norm": row.get("V_reconf_norm", 0.0),
            "V_mob_net": row.get("V_mob_net", 0.0),
        }
        for row in summary_rows
    ]
    return _finalize_experiment(
        base_dir,
        "E10",
        question,
        metrics=[],
        summary_rows=rows,
        figure_paths=[figure_path],
        notes=[threshold_rule],
    )


def _check_stop_conditions(artifacts: Sequence[ExperimentArtifacts]) -> None:
    raw_rows = [row for artifact in artifacts for row in artifact.raw_rows]
    if raw_rows:
        failed = [row for row in raw_rows if not _is_case_conclusion_ready(row)]
        if len(failed) / len(raw_rows) > 0.2:
            raise RuntimeError("More than 20% of solves failed or were infeasible.")

    e6 = next((artifact for artifact in artifacts if artifact.experiment_id == "E6"), None)
    if e6:
        winner_count = len({str(row["winner_case"]) for row in e6.summary_rows})
        if winner_count < 2:
            print(
                "[experiments] warning: E6 hero map has a single winner across all scanned regimes; "
                "outputs were still materialized",
                flush=True,
            )


def _build_manifest(base_dir: Path, artifacts: Sequence[ExperimentArtifacts]) -> Path:
    manifest_rows = []
    for artifact in artifacts:
        manifest_rows.append(
            {
                "experiment_id": artifact.experiment_id,
                "question": artifact.question,
                "raw_csv": artifact.raw_csv.as_posix(),
                "summary_csv": artifact.summary_csv.as_posix(),
                "memo_md": artifact.memo_md.as_posix(),
                "figures": ";".join(path.as_posix() for path in artifact.figure_paths),
                "status": artifact.status,
            }
        )
    manifest_path = base_dir / "sim" / "outputs" / "experiment_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    return manifest_path


def _build_master_csv(base_dir: Path, artifacts: Sequence[ExperimentArtifacts]) -> Path:
    rows = [row for artifact in artifacts for row in artifact.raw_rows]
    path = base_dir / "sim" / "outputs" / "all_scenarios_master.csv"
    _write_csv(path, rows)
    return path


def _build_handoff(base_dir: Path, artifacts: Sequence[ExperimentArtifacts], commands: Sequence[str]) -> Path:
    lines = [
        "# Experiment Handoff",
        "",
        "## Commands Run",
    ]
    for command in commands:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Outputs"])
    for artifact in artifacts:
        lines.append(f"- {artifact.experiment_id}: `{artifact.raw_csv.as_posix()}`, `{artifact.summary_csv.as_posix()}`, `{artifact.memo_md.as_posix()}`")
        for figure in artifact.figure_paths:
            lines.append(f"- figure: `{figure.as_posix()}`")
    thesis_ready = [artifact.experiment_id for artifact in artifacts if artifact.experiment_id in {"E1", "E6", "E8"}]
    inconclusive = [artifact.experiment_id for artifact in artifacts if artifact.status != "clean"]
    lines.extend(
        [
            "",
            "## Conclusion",
            f"- Thesis-ready figures: {', '.join(thesis_ready)}",
            f"- Inconclusive experiments: {', '.join(inconclusive) if inconclusive else 'none'}",
            "- Next Results chapter step: write baseline comparison first, then the E6 regime map, then the E8 mechanism decomposition, and finally use E9/E10 as robustness and interpretation support.",
        ]
    )
    handoff_path = base_dir / "reports" / "experiment_handoff.md"
    _write_text(handoff_path, "\n".join(lines) + "\n")
    return handoff_path


def _experiment_runner_map() -> dict[str, callable]:
    return {
        "E0": run_e0,
        "E1": run_e1,
        "E2": run_e2,
        "E3": run_e3,
        "E4": run_e4,
        "E5": run_e5,
        "E6": run_e6,
        "E7": run_e7,
        "E8": run_e8,
        "E9": run_e9,
    }


def _run_experiment_worker(exp_id: str, base_dir: Path, defaults: ExperimentDefaults) -> ExperimentArtifacts:
    runner = _experiment_runner_map()[exp_id]
    return runner(base_dir, defaults)


def run_all_experiments(
    base_dir: Path,
    *,
    solver_name: str = "highs",
    experiment_ids: Sequence[str] | None = None,
    reuse_existing: bool = False,
) -> dict[str, object]:
    defaults = replace(ExperimentDefaults(), solver=solver_name)
    commands = [f"uv run python sim/src/eep_sim/run_experiments.py --base-dir {base_dir.as_posix()} --run-all --solver {solver_name}"]
    runner_map = _experiment_runner_map()
    selected = [exp.upper() for exp in experiment_ids] if experiment_ids else list(runner_map.keys())
    independent_experiments = [exp_id for exp_id in selected if exp_id != "E10"]
    artifacts_by_id: dict[str, ExperimentArtifacts] = {}

    if reuse_existing:
        commands[0] += " --reuse-existing"
        for exp_id in runner_map:
            if exp_id in independent_experiments:
                continue
            artifact = _load_existing_experiment_artifact(base_dir, exp_id)
            if artifact is not None:
                artifacts_by_id[exp_id] = artifact
                print(f"[experiments] reusing {exp_id} from disk ({artifact.status})", flush=True)

    if len(independent_experiments) <= 1 or EXPERIMENT_WORKERS <= 1:
        for exp_id in independent_experiments:
            runner = runner_map[exp_id]
            print(f"[experiments] starting {runner.__name__}", flush=True)
            artifact = runner(base_dir, defaults)
            artifacts_by_id[exp_id] = artifact
            print(f"[experiments] finished {artifact.experiment_id} ({artifact.status})", flush=True)
            _check_stop_conditions([artifacts_by_id[item] for item in runner_map if item in artifacts_by_id])
    else:
        max_workers = min(EXPERIMENT_WORKERS, len(independent_experiments))
        print(f"[experiments] running {len(independent_experiments)} experiments with {max_workers} workers", flush=True)
        ordered_artifacts: dict[str, ExperimentArtifacts] = {}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            future_by_exp_id = {
                executor.submit(_run_experiment_worker, exp_id, base_dir, defaults): exp_id
                for exp_id in independent_experiments
            }
            for future in as_completed(future_by_exp_id):
                exp_id = future_by_exp_id[future]
                artifact = future.result()
                ordered_artifacts[exp_id] = artifact
                print(f"[experiments] finished {artifact.experiment_id} ({artifact.status})", flush=True)
        for exp_id in independent_experiments:
            artifacts_by_id[exp_id] = ordered_artifacts[exp_id]
        _check_stop_conditions([artifacts_by_id[item] for item in runner_map if item in artifacts_by_id])

    artifacts = [artifacts_by_id[exp_id] for exp_id in runner_map if exp_id in artifacts_by_id]

    if experiment_ids is None or "E10" in selected:
        print("[experiments] starting run_e10", flush=True)
        e10 = run_e10(base_dir, artifacts)
        artifacts.append(e10)
        print(f"[experiments] finished {e10.experiment_id} ({e10.status})", flush=True)

    manifest_path = _build_manifest(base_dir, artifacts)
    master_path = _build_master_csv(base_dir, artifacts)
    handoff_path = _build_handoff(base_dir, artifacts, commands)
    return {
        "manifest_path": manifest_path,
        "master_path": master_path,
        "handoff_path": handoff_path,
        "artifacts": artifacts,
    }
