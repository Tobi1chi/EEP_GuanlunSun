"""Solve a same-budget planning baseline on the real corridor data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import solve_core_model_with_diagnostics  # noqa: E402
from eep_sim.experiments import make_case_data, prepare_same_budget_data  # noqa: E402
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data  # noqa: E402


def _parse_hybrid_fixed_shares(text: str) -> list[float]:
    shares = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not (0.0 < value < 1.0):
            raise ValueError("hybrid fixed shares must lie strictly between 0 and 1")
        shares.append(value)
    if not shares:
        raise ValueError("at least one hybrid fixed share is required")
    return sorted(set(shares))


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed, mobile, and hybrid storage under a shared investment budget"
    )
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--lp-time-limit-seconds", type=float, default=60.0)
    parser.add_argument(
        "--mobile-cost-premium-ratio",
        type=float,
        default=1.5,
        help="alpha = c_mobile / c_fixed used in the same-budget comparison",
    )
    parser.add_argument(
        "--fixed-cost-per-mwh",
        type=float,
        default=1.0,
        help="normalized fixed-storage investment cost per MWh",
    )
    parser.add_argument(
        "--hybrid-fixed-shares",
        default="0.25,0.5,0.75",
        help="comma-separated fixed-budget shares for hybrid scan, e.g. 0.25,0.5,0.75",
    )
    args = parser.parse_args()

    hybrid_fixed_shares = _parse_hybrid_fixed_shares(args.hybrid_fixed_shares)

    options = G2LoaderOptions(
        reconf_hours=args.reconf_hours,
        delay_windows=args.delay_windows,
        timeline_mode="representative_12x24",
        grid_capacity_mode="max_of_csv_and_avg_load_margin",
        grid_capacity_margin_above_avg_load=0.25,
        default_unserved_penalty_yuan_per_mwh=1000.0,
        symbolic_unserved_penalty_mode="fixed",
        c_reconf_yuan_per_mwh=0.5,
    )
    data, diag = load_g2_core_model_data(args.base_dir, options=options)
    baseline_fixed_total = sum(data.fixed_capacity_mwh.values())
    shared_budget = args.fixed_cost_per_mwh * baseline_fixed_total

    case_specs: list[tuple[str, str, float]] = [
        ("fixed_only", "fixed_only", 1.0),
        ("mobile_only", "mobile_only", 0.0),
    ]
    case_specs.extend(
        ("hybrid", f"hybrid_fixed_share_{_slug(share)}", share) for share in hybrid_fixed_shares
    )

    result_rows: list[dict[str, float | str]] = []
    for case, variant_label, fixed_share in case_specs:
        budget_data = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=args.mobile_cost_premium_ratio,
            fixed_budget_share=fixed_share,
            budget_in_fixed_cost_units=shared_budget,
            fixed_cost_per_mwh=args.fixed_cost_per_mwh,
        )
        case_data = make_case_data(budget_data, case)
        diagnostics, solution = solve_core_model_with_diagnostics(
            case_data,
            solver_name=args.solver,
            time_limit_seconds=args.time_limit_seconds,
            lp_time_limit_seconds=args.lp_time_limit_seconds,
        )
        if solution is None:
            raise RuntimeError(
                f"Same-budget baseline requires a feasible incumbent for {variant_label}, got {diagnostics.case_outcome}."
            )

        fixed_total = float(sum(case_data.fixed_capacity_mwh.values()))
        mobile_total = float(case_data.m_total_mwh)
        result_rows.append(
            {
                "variant_label": variant_label,
                "case": case,
                "fixed_budget_share": fixed_share,
                "fixed_total_mwh": fixed_total,
                "mobile_total_mwh": mobile_total,
                "budget_check": fixed_total * args.fixed_cost_per_mwh
                + mobile_total * args.fixed_cost_per_mwh * args.mobile_cost_premium_ratio,
                "case_outcome": diagnostics.case_outcome,
                "C_total_primary": float(solution["costs"]["C_total_primary"]),
                "C_storage": float(solution["costs"]["C_storage"]),
                "C_total_planning": float(solution["costs"]["C_total_planning"]),
                "total_unmet_load": float(sum(solution["unserved"].values())),
            }
        )

    result_rows.sort(key=lambda row: float(row["C_total_planning"]))
    winner = result_rows[0]

    reports_dir = Path(args.base_dir).resolve() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    alpha_slug = _slug(args.mobile_cost_premium_ratio)
    csv_path = reports_dir / f"same_budget_baseline_alpha_{alpha_slug}.csv"
    md_path = reports_dir / f"same_budget_baseline_alpha_{alpha_slug}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    lines = [
        "# Same-budget planning baseline",
        "",
        f"- sites: {diag.sites}",
        f"- timeline_mode: {diag.timeline_mode}",
        f"- shared_budget_in_fixed_cost_units: {shared_budget:.6f}",
        f"- fixed_baseline_total_mwh: {baseline_fixed_total:.6f}",
        f"- mobile_cost_premium_ratio_alpha: {args.mobile_cost_premium_ratio:.6f}",
        f"- winner_by_planning_objective: {winner['variant_label']}",
        "",
        "## Results",
        "",
        "| variant | case | fixed share | fixed MWh | mobile MWh | C_total_primary | C_storage | C_total_planning | unmet load |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            "| "
            f"{row['variant_label']} | {row['case']} | {float(row['fixed_budget_share']):.2f} | "
            f"{float(row['fixed_total_mwh']):.3f} | {float(row['mobile_total_mwh']):.3f} | "
            f"{float(row['C_total_primary']):.3f} | {float(row['C_storage']):.3f} | "
            f"{float(row['C_total_planning']):.3f} | {float(row['total_unmet_load']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The shared budget is defined relative to the original fixed-only baseline investment.",
            "Hybrid candidates are generated by scanning fixed-budget shares while enforcing the same total budget.",
            "The planning ranking is based on C_total_planning = C_grid + C_unserved + C_reconf + C_storage.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("same-budget planning baseline summary:")
    print(f"- sites: {diag.sites}")
    print(f"- timeline_mode: {diag.timeline_mode}")
    print(f"- fixed_baseline_total_mwh: {baseline_fixed_total:.6f}")
    print(f"- shared_budget_in_fixed_cost_units: {shared_budget:.6f}")
    print(f"- mobile_cost_premium_ratio_alpha: {args.mobile_cost_premium_ratio:.6f}")
    for row in result_rows:
        print(
            f"- {row['variant_label']}: case={row['case']}, fixed_share={float(row['fixed_budget_share']):.2f}, "
            f"fixed_mwh={float(row['fixed_total_mwh']):.3f}, mobile_mwh={float(row['mobile_total_mwh']):.3f}, "
            f"C_total_primary={float(row['C_total_primary']):.6f}, C_storage={float(row['C_storage']):.6f}, "
            f"C_total_planning={float(row['C_total_planning']):.6f}, unmet_load={float(row['total_unmet_load']):.6f}"
        )
    print(f"- winner_by_planning_objective: {winner['variant_label']}")
    print(f"- csv: {csv_path}")
    print(f"- memo: {md_path}")


if __name__ == "__main__":
    main()
