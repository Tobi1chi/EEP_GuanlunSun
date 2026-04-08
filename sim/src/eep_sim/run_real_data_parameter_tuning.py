"""Tune uncertain real-data scenario parameters and compare base/fixed/MESS.

This script keeps the service-area positions and electricity prices from the
current G2 dataset, while scanning only the parameters that are weakly
evidenced in the current workflow:

- demand scale (`base_load_scale`)
- grid-capacity margin above average load
- holiday inter-site staggering strength
- MESS reconfiguration penalty

The scan is run on a Spring Festival holiday window first, then the selected
parameter set is validated on National Day.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import CoreModelData, build_core_model, extract_solution, solve_core_model
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data
from eep_sim.run_holiday_stagger_demo import (
    HOLIDAY_RANGES_2026,
    _apply_staggered_holiday_profile,
    _load_corridor_order_and_archetype,
)
from eep_sim.run_holiday_stress_demo import _subset_data_for_dates


@dataclass(frozen=True)
class ScenarioParams:
    base_load_scale: float
    grid_capacity_margin_above_avg_load: float
    stagger_scale: float
    c_reconf_yuan_per_mwh: float


@dataclass(frozen=True)
class SolveMetrics:
    total_cost: float
    grid_cost: float
    unserved_cost: float
    reconf_cost: float
    unserved_total: float
    reconf_total: float


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _leader_metrics(data: CoreModelData) -> dict[str, float]:
    leaders: list[str] = []
    total_excess_values: list[float] = []
    positive_sites_values: list[int] = []
    top1_share_values: list[float] = []

    for w in data.windows:
        excess_by_site: list[tuple[float, str]] = []
        total_excess = 0.0
        positive_sites = 0
        for site in data.sites:
            excess = sum(
                max(0.0, data.load_mw[(site, t)] - data.grid_limit_mw[(site, t)])
                for t in data.hours
                if data.hour_to_window[t] == w
            )
            excess_by_site.append((excess, site))
            total_excess += excess
            if excess > 1e-9:
                positive_sites += 1
        excess_by_site.sort(reverse=True)
        leaders.append(excess_by_site[0][1])
        total_excess_values.append(total_excess)
        positive_sites_values.append(positive_sites)
        top1_share_values.append(excess_by_site[0][0] / total_excess if total_excess > 0 else 0.0)

    leader_changes = sum(1 for idx in range(1, len(leaders)) if leaders[idx] != leaders[idx - 1])
    return {
        "leader_changes": float(leader_changes),
        "mean_positive_sites": sum(positive_sites_values) / len(positive_sites_values),
        "mean_top1_share": sum(top1_share_values) / len(top1_share_values),
        "mean_total_excess": sum(total_excess_values) / len(total_excess_values),
    }


def _solve_metrics(data: CoreModelData, solver: str) -> SolveMetrics:
    model = build_core_model(data)
    solve_core_model(model, solver_name=solver)
    solution = extract_solution(model)
    reconf_total = sum(
        abs(solution["x"][(site, w)] - solution["x"][(site, w - 1)])
        for site in data.sites
        for w in data.windows
        if w > 0
    )
    return SolveMetrics(
        total_cost=float(solution["costs"]["C_total"]),
        grid_cost=float(solution["costs"]["C_grid"]),
        unserved_cost=float(solution["costs"]["C_unserved"]),
        reconf_cost=float(solution["costs"]["C_reconf"]),
        unserved_total=float(sum(solution["unserved"].values())),
        reconf_total=float(reconf_total),
    )


def _make_scheme_data(
    data: CoreModelData,
    *,
    scheme: str,
    c_reconf_yuan_per_mwh: float,
) -> CoreModelData:
    windows = list(data.windows)
    if scheme == "base":
        return replace(
            data,
            m_total_mwh=0.0,
            delay_windows=0,
            active_init_mwh={},
            reconf_limit_mwh={w: 0.0 for w in windows},
            c_reconf_yuan_per_mwh=0.0,
        )

    if scheme == "fixed":
        m_total = float(data.m_total_mwh)
        return replace(
            data,
            delay_windows=0,
            active_init_mwh={},
            reconf_limit_mwh={0: 2.0 * m_total, **{w: 0.0 for w in windows if w > 0}},
            c_reconf_yuan_per_mwh=0.0,
        )

    if scheme == "mess":
        m_total = float(data.m_total_mwh)
        return replace(
            data,
            reconf_limit_mwh={w: 2.0 * m_total for w in windows},
            c_reconf_yuan_per_mwh=c_reconf_yuan_per_mwh,
        )

    raise ValueError(f"unsupported scheme: {scheme}")


def _load_holiday_case(
    base_dir: Path,
    *,
    holiday_name: str,
    params: ScenarioParams,
    delay_windows: int,
    mobile_capacity_ratio_to_fixed: float,
) -> tuple[CoreModelData, dict[str, float]]:
    corridor_order, archetype_by_service = _load_corridor_order_and_archetype(base_dir)
    options = G2LoaderOptions(
        reconf_hours=24,
        delay_windows=delay_windows,
        mobile_capacity_ratio_to_fixed=mobile_capacity_ratio_to_fixed,
        timeline_mode="expanded_8760",
        holiday_year=2026,
        apply_cn_holiday_shocks=False,
        base_load_scale=params.base_load_scale,
        enable_site_month_heterogeneity=True,
        site_month_heterogeneity_strength=0.18,
        enable_holiday_site_stagger=False,
        grid_capacity_mode="max_of_csv_and_avg_load_margin",
        grid_capacity_margin_above_avg_load=params.grid_capacity_margin_above_avg_load,
        c_reconf_yuan_per_mwh=params.c_reconf_yuan_per_mwh,
        default_unserved_penalty_yuan_per_mwh=1000.0,
        symbolic_unserved_penalty_mode="fixed",
    )
    full_data, _diag = load_g2_core_model_data(base_dir, options=options)
    start, end = HOLIDAY_RANGES_2026[holiday_name]
    subset_data = _subset_data_for_dates(
        full_data,
        year=2026,
        start=start,
        end=end,
        reconf_hours=24,
        dt_hours=options.dt_hours,
        delay_windows=delay_windows,
        initial_active_mode=options.initial_active_mode,
        reconf_limit_mwh=options.reconf_limit_mwh,
    )
    staggered_data = _apply_staggered_holiday_profile(
        subset_data,
        holiday_name=holiday_name,
        corridor_order=corridor_order,
        archetype_by_service=archetype_by_service,
        stagger_scale=params.stagger_scale,
    )
    metrics = _leader_metrics(staggered_data)
    return staggered_data, metrics


def _norm(value: float, lower: float, upper: float) -> float:
    if math.isclose(upper, lower):
        return 0.5
    return (value - lower) / (upper - lower)


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune uncertain real-data scenario parameters")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--mobile-capacity-ratio-to-fixed", type=float, default=0.7)
    parser.add_argument("--base-load-scales", default="0.85,0.95")
    parser.add_argument("--grid-margins", default="0.10,0.20")
    parser.add_argument("--stagger-scales", default="1.0,1.2")
    parser.add_argument("--reconf-costs", default="0.1,0.5,1.0")
    parser.add_argument(
        "--output-csv",
        default="sim/outputs/real_data_parameter_tuning.csv",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    base_load_scales = _parse_float_list(args.base_load_scales)
    grid_margins = _parse_float_list(args.grid_margins)
    stagger_scales = _parse_float_list(args.stagger_scales)
    reconf_costs = _parse_float_list(args.reconf_costs)

    scenario_rows: list[dict[str, float | str]] = []
    print("scan grid:")
    print(f"- base_load_scales: {base_load_scales}")
    print(f"- grid_margins: {grid_margins}")
    print(f"- stagger_scales: {stagger_scales}")
    print(f"- reconf_costs: {reconf_costs}")
    print(f"- delay_windows: {args.delay_windows}")
    print()

    for base_load_scale in base_load_scales:
        for grid_margin in grid_margins:
            for stagger_scale in stagger_scales:
                for reconf_cost in reconf_costs:
                    params = ScenarioParams(
                        base_load_scale=base_load_scale,
                        grid_capacity_margin_above_avg_load=grid_margin,
                        stagger_scale=stagger_scale,
                        c_reconf_yuan_per_mwh=reconf_cost,
                    )
                    spring_data, data_metrics = _load_holiday_case(
                        base_dir,
                        holiday_name="spring_festival",
                        params=params,
                        delay_windows=args.delay_windows,
                        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
                    )
                    total_load = sum(spring_data.load_mw.values())
                    base_metrics = _solve_metrics(
                        _make_scheme_data(
                            spring_data,
                            scheme="base",
                            c_reconf_yuan_per_mwh=reconf_cost,
                        ),
                        args.solver,
                    )
                    fixed_metrics = _solve_metrics(
                        _make_scheme_data(
                            spring_data,
                            scheme="fixed",
                            c_reconf_yuan_per_mwh=reconf_cost,
                        ),
                        args.solver,
                    )
                    mess_metrics = _solve_metrics(
                        _make_scheme_data(
                            spring_data,
                            scheme="mess",
                            c_reconf_yuan_per_mwh=reconf_cost,
                        ),
                        args.solver,
                    )
                    scenario_rows.append(
                        {
                            "holiday": "spring_festival",
                            "base_load_scale": base_load_scale,
                            "grid_margin": grid_margin,
                            "stagger_scale": stagger_scale,
                            "c_reconf": reconf_cost,
                            "leader_changes": data_metrics["leader_changes"],
                            "mean_positive_sites": round(data_metrics["mean_positive_sites"], 6),
                            "mean_top1_share": round(data_metrics["mean_top1_share"], 6),
                            "mean_total_excess": round(data_metrics["mean_total_excess"], 6),
                            "base_total_cost": round(base_metrics.total_cost, 6),
                            "fixed_total_cost": round(fixed_metrics.total_cost, 6),
                            "mess_total_cost": round(mess_metrics.total_cost, 6),
                            "base_unserved_total": round(base_metrics.unserved_total, 6),
                            "fixed_unserved_total": round(fixed_metrics.unserved_total, 6),
                            "mess_unserved_total": round(mess_metrics.unserved_total, 6),
                            "mess_reconf_total": round(mess_metrics.reconf_total, 6),
                            "fixed_value_vs_base": round(base_metrics.total_cost - fixed_metrics.total_cost, 6),
                            "mess_value_vs_base": round(base_metrics.total_cost - mess_metrics.total_cost, 6),
                            "mess_advantage_vs_fixed": round(
                                fixed_metrics.total_cost - mess_metrics.total_cost, 6
                            ),
                            "fixed_unserved_share": round(fixed_metrics.unserved_total / total_load, 6),
                            "mess_unserved_share": round(mess_metrics.unserved_total / total_load, 6),
                            "unserved_reduction_vs_fixed": round(
                                fixed_metrics.unserved_total - mess_metrics.unserved_total, 6
                            ),
                        }
                    )
                    print(
                        "tested"
                        f" base_load_scale={base_load_scale},"
                        f" grid_margin={grid_margin},"
                        f" stagger_scale={stagger_scale},"
                        f" c_reconf={reconf_cost}"
                        f" -> mess_adv_vs_fixed={scenario_rows[-1]['mess_advantage_vs_fixed']:.1f},"
                        f" reconf={scenario_rows[-1]['mess_reconf_total']:.2f},"
                        f" leader_changes={int(data_metrics['leader_changes'])}"
                    )

    benefit_values = [float(r["mess_advantage_vs_fixed"]) for r in scenario_rows]
    reconf_values = [float(r["mess_reconf_total"]) for r in scenario_rows]
    unserved_values = [float(r["unserved_reduction_vs_fixed"]) for r in scenario_rows]
    baseline = ScenarioParams(
        base_load_scale=0.85,
        grid_capacity_margin_above_avg_load=0.25,
        stagger_scale=1.0,
        c_reconf_yuan_per_mwh=0.5,
    )

    for row in scenario_rows:
        distance = (
            abs(float(row["base_load_scale"]) - baseline.base_load_scale) / 0.10
            + abs(float(row["grid_margin"]) - baseline.grid_capacity_margin_above_avg_load) / 0.15
            + abs(float(row["stagger_scale"]) - baseline.stagger_scale) / 0.20
            + abs(float(row["c_reconf"]) - baseline.c_reconf_yuan_per_mwh) / 0.90
        )
        score = (
            0.55 * _norm(float(row["mess_advantage_vs_fixed"]), min(benefit_values), max(benefit_values))
            + 0.25 * _norm(float(row["unserved_reduction_vs_fixed"]), min(unserved_values), max(unserved_values))
            + 0.20 * _norm(float(row["mess_reconf_total"]), min(reconf_values), max(reconf_values))
            - 0.10 * distance
        )
        row["recommendation_score"] = round(score, 6)
        row["distance_from_baseline"] = round(distance, 6)

    scenario_rows.sort(key=lambda r: float(r["recommendation_score"]), reverse=True)
    best = scenario_rows[0]

    best_params = ScenarioParams(
        base_load_scale=float(best["base_load_scale"]),
        grid_capacity_margin_above_avg_load=float(best["grid_margin"]),
        stagger_scale=float(best["stagger_scale"]),
        c_reconf_yuan_per_mwh=float(best["c_reconf"]),
    )
    national_data, national_metrics = _load_holiday_case(
        base_dir,
        holiday_name="national_day",
        params=best_params,
        delay_windows=args.delay_windows,
        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
    )
    national_total_load = sum(national_data.load_mw.values())
    national_base = _solve_metrics(
        _make_scheme_data(
            national_data,
            scheme="base",
            c_reconf_yuan_per_mwh=best_params.c_reconf_yuan_per_mwh,
        ),
        args.solver,
    )
    national_fixed = _solve_metrics(
        _make_scheme_data(
            national_data,
            scheme="fixed",
            c_reconf_yuan_per_mwh=best_params.c_reconf_yuan_per_mwh,
        ),
        args.solver,
    )
    national_mess = _solve_metrics(
        _make_scheme_data(
            national_data,
            scheme="mess",
            c_reconf_yuan_per_mwh=best_params.c_reconf_yuan_per_mwh,
        ),
        args.solver,
    )

    output_path = Path(args.output_csv).resolve()
    _write_csv(output_path, scenario_rows)

    print("\nrecommended spring_festival scenario:")
    print(
        f"- base_load_scale={best['base_load_scale']},"
        f" grid_margin={best['grid_margin']},"
        f" stagger_scale={best['stagger_scale']},"
        f" c_reconf={best['c_reconf']}"
    )
    print(f"- leader_changes={int(float(best['leader_changes']))}")
    print(f"- mean_positive_sites={best['mean_positive_sites']}")
    print(f"- mean_top1_share={best['mean_top1_share']}")
    print(f"- fixed_value_vs_base={best['fixed_value_vs_base']}")
    print(f"- mess_value_vs_base={best['mess_value_vs_base']}")
    print(f"- mess_advantage_vs_fixed={best['mess_advantage_vs_fixed']}")
    print(f"- mess_reconf_total={best['mess_reconf_total']}")
    print(f"- fixed_unserved_share={best['fixed_unserved_share']}")
    print(f"- mess_unserved_share={best['mess_unserved_share']}")

    print("\nnational_day validation with recommended parameters:")
    print(f"- leader_changes={int(national_metrics['leader_changes'])}")
    print(f"- mean_positive_sites={round(national_metrics['mean_positive_sites'], 6)}")
    print(f"- mean_top1_share={round(national_metrics['mean_top1_share'], 6)}")
    print(f"- fixed_value_vs_base={round(national_base.total_cost - national_fixed.total_cost, 6)}")
    print(f"- mess_value_vs_base={round(national_base.total_cost - national_mess.total_cost, 6)}")
    print(f"- mess_advantage_vs_fixed={round(national_fixed.total_cost - national_mess.total_cost, 6)}")
    print(f"- mess_reconf_total={round(national_mess.reconf_total, 6)}")
    print(f"- fixed_unserved_share={round(national_fixed.unserved_total / national_total_load, 6)}")
    print(f"- mess_unserved_share={round(national_mess.unserved_total / national_total_load, 6)}")
    print(f"\nresults_csv: {output_path}")


if __name__ == "__main__":
    main()
