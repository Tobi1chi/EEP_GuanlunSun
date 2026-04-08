"""Scan planning winners over shared-budget multipliers and hybrid fixed-budget shares."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import solve_core_model_with_diagnostics  # noqa: E402
from eep_sim.experiments import WINNER_CODE, WINNER_NAME, WINNER_COLORS, make_case_data, prepare_same_budget_data  # noqa: E402
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data  # noqa: E402


def _parse_sorted_floats(text: str, *, lower: float | None = None, upper: float | None = None) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if lower is not None and value < lower:
            raise ValueError(f"value {value} must be >= {lower}")
        if upper is not None and value > upper:
            raise ValueError(f"value {value} must be <= {upper}")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return sorted(set(values))


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _winner_from_rows(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    ranked = sorted(rows, key=lambda row: float(row["C_total_planning"]))
    return ranked[0]


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_winner_heatmap(summary_rows: list[dict[str, float | str]], path: Path, *, alpha: float) -> None:
    budget_values = sorted({float(row["budget_multiplier"]) for row in summary_rows})
    share_values = sorted({float(row["fixed_budget_share"]) for row in summary_rows})
    matrix = [[WINNER_CODE["fixed_only"] for _ in budget_values] for _ in share_values]
    for row in summary_rows:
        x_idx = budget_values.index(float(row["budget_multiplier"]))
        y_idx = share_values.index(float(row["fixed_budget_share"]))
        matrix[y_idx][x_idx] = WINNER_CODE[str(row["case"])]

    cmap = ListedColormap([WINNER_COLORS[WINNER_NAME[idx]] for idx in range(len(WINNER_NAME)) if WINNER_NAME[idx] != "no_storage"])
    remap = {"fixed_only": 0, "mobile_only": 1, "hybrid": 2}
    matrix = [[remap[WINNER_NAME[value]] for value in row] for row in matrix]

    plt.figure(figsize=(8.5, 5.4))
    plt.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
    plt.xticks(range(len(budget_values)), [f"{value:.2f}x" for value in budget_values])
    plt.yticks(range(len(share_values)), [f"{value:.2f}" for value in share_values])
    plt.xlabel("Budget Multiplier vs Current Fixed Baseline")
    plt.ylabel("Hybrid Fixed-Budget Share")
    plt.title(f"Planning Winner Map (alpha={alpha:.2f})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WINNER_COLORS[case])
        for case in ("fixed_only", "mobile_only", "hybrid")
    ]
    plt.legend(handles, ["Fixed only", "Mobile only", "Hybrid"], fontsize=8, loc="upper right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_total_capacity_lines(detail_rows: list[dict[str, float | str]], path: Path, *, alpha: float) -> None:
    grouped: dict[float, list[dict[str, float | str]]] = {}
    for row in detail_rows:
        grouped.setdefault(float(row["budget_multiplier"]), []).append(row)

    plt.figure(figsize=(8.5, 5.2))
    for budget_multiplier in sorted(grouped):
        rows = sorted(grouped[budget_multiplier], key=lambda row: float(row["fixed_budget_share"]))
        x = [float(row["fixed_budget_share"]) for row in rows]
        y = [float(row["total_storage_mwh"]) for row in rows]
        plt.plot(x, y, marker="o", label=f"{budget_multiplier:.2f}x budget")
    plt.xlabel("Fixed-Budget Share")
    plt.ylabel("Total Storage Capacity (MWh)")
    plt.title(f"Planning Capacity Envelope (alpha={alpha:.2f})")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan planning winners from the current storage baseline upward under a shared budget"
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
        help="alpha = c_mobile / c_fixed used in the planning comparison",
    )
    parser.add_argument(
        "--fixed-cost-per-mwh",
        type=float,
        default=1.0,
        help="normalized fixed-storage investment cost per MWh",
    )
    parser.add_argument(
        "--budget-multipliers",
        default="1.0,1.25,1.5,1.75,2.0",
        help="comma-separated multipliers relative to the current fixed-only baseline budget",
    )
    parser.add_argument(
        "--hybrid-fixed-shares",
        default="0.0,0.25,0.5,0.75,1.0",
        help="comma-separated fixed-budget shares; 0 and 1 correspond to mobile-only and fixed-only endpoints",
    )
    args = parser.parse_args()

    budget_multipliers = _parse_sorted_floats(args.budget_multipliers, lower=1.0)
    fixed_shares = _parse_sorted_floats(args.hybrid_fixed_shares, lower=0.0, upper=1.0)

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
    baseline_fixed_total = float(sum(data.fixed_capacity_mwh.values()))
    base_budget = args.fixed_cost_per_mwh * baseline_fixed_total

    detail_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for budget_multiplier in budget_multipliers:
        shared_budget = base_budget * budget_multiplier
        rows_for_budget: list[dict[str, float | str]] = []
        for fixed_share in fixed_shares:
            planning_data = prepare_same_budget_data(
                data,
                mobile_cost_premium_ratio=args.mobile_cost_premium_ratio,
                fixed_budget_share=fixed_share,
                budget_in_fixed_cost_units=shared_budget,
                fixed_cost_per_mwh=args.fixed_cost_per_mwh,
            )
            case = "hybrid"
            if fixed_share == 0.0:
                case = "mobile_only"
            elif fixed_share == 1.0:
                case = "fixed_only"

            case_data = make_case_data(planning_data, case)
            diagnostics, solution = solve_core_model_with_diagnostics(
                case_data,
                solver_name=args.solver,
                time_limit_seconds=args.time_limit_seconds,
                lp_time_limit_seconds=args.lp_time_limit_seconds,
            )
            if solution is None:
                raise RuntimeError(
                    f"Planning capacity scan requires a feasible incumbent for budget={budget_multiplier:.2f}x, "
                    f"fixed_share={fixed_share:.2f}; got {diagnostics.case_outcome}."
                )

            fixed_total = float(sum(case_data.fixed_capacity_mwh.values()))
            mobile_total = float(case_data.m_total_mwh)
            row = {
                "budget_multiplier": budget_multiplier,
                "shared_budget_in_fixed_cost_units": shared_budget,
                "fixed_budget_share": fixed_share,
                "case": case,
                "fixed_total_mwh": fixed_total,
                "mobile_total_mwh": mobile_total,
                "total_storage_mwh": fixed_total + mobile_total,
                "C_total_primary": float(solution["costs"]["C_total_primary"]),
                "C_storage": float(solution["costs"]["C_storage"]),
                "C_total_planning": float(solution["costs"]["C_total_planning"]),
                "total_unmet_load": float(sum(solution["unserved"].values())),
                "case_outcome": diagnostics.case_outcome,
            }
            detail_rows.append(row)
            rows_for_budget.append(row)

        winner = _winner_from_rows(rows_for_budget)
        summary_rows.append(
            {
                "budget_multiplier": budget_multiplier,
                "shared_budget_in_fixed_cost_units": shared_budget,
                "winner_case": winner["case"],
                "winner_fixed_budget_share": winner["fixed_budget_share"],
                "winner_total_storage_mwh": winner["total_storage_mwh"],
                "winner_C_total_planning": winner["C_total_planning"],
                "winner_unmet_load": winner["total_unmet_load"],
            }
        )

    reports_dir = Path(args.base_dir).resolve() / "reports"
    figures_dir = Path(args.base_dir).resolve() / "figures" / "experiments"
    alpha_slug = _slug(args.mobile_cost_premium_ratio)
    detail_csv = reports_dir / f"planning_capacity_scan_alpha_{alpha_slug}.csv"
    summary_csv = reports_dir / f"planning_capacity_scan_summary_alpha_{alpha_slug}.csv"
    summary_md = reports_dir / f"planning_capacity_scan_alpha_{alpha_slug}.md"
    winner_map_path = figures_dir / f"planning_winner_map_alpha_{alpha_slug}.png"
    capacity_line_path = figures_dir / f"planning_capacity_envelope_alpha_{alpha_slug}.png"

    _write_csv(detail_csv, detail_rows)
    _write_csv(summary_csv, summary_rows)
    _plot_winner_heatmap(detail_rows, winner_map_path, alpha=args.mobile_cost_premium_ratio)
    _plot_total_capacity_lines(detail_rows, capacity_line_path, alpha=args.mobile_cost_premium_ratio)

    lines = [
        "# Planning capacity scan",
        "",
        f"- sites: {diag.sites}",
        f"- timeline_mode: {diag.timeline_mode}",
        f"- baseline_fixed_total_mwh: {baseline_fixed_total:.6f}",
        f"- baseline_budget_in_fixed_cost_units: {base_budget:.6f}",
        f"- mobile_cost_premium_ratio_alpha: {args.mobile_cost_premium_ratio:.6f}",
        f"- budget_multipliers: {', '.join(f'{value:.2f}' for value in budget_multipliers)}",
        f"- hybrid_fixed_shares: {', '.join(f'{value:.2f}' for value in fixed_shares)}",
        "",
        "Interpretation:",
        "- All variants are solved in planning mode with a shared budget cap.",
        "- Because alpha > 1, total storage capacity is monotone in fixed-budget share: fixed >= hybrid >= mobile.",
        "- Budget multipliers start from the current fixed-only baseline budget and scan upward.",
        "",
        "## Winner by budget",
        "",
        "| budget x | winner | fixed share | total storage [MWh] | planning objective | unmet load |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{float(row['budget_multiplier']):.2f} | {row['winner_case']} | "
            f"{float(row['winner_fixed_budget_share']):.2f} | {float(row['winner_total_storage_mwh']):.3f} | "
            f"{float(row['winner_C_total_planning']):.3f} | {float(row['winner_unmet_load']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- detail csv: {detail_csv}",
            f"- summary csv: {summary_csv}",
            f"- winner map: {winner_map_path}",
            f"- capacity envelope: {capacity_line_path}",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("planning capacity scan completed")
    print(f"- detail csv: {detail_csv}")
    print(f"- summary csv: {summary_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- winner map: {winner_map_path}")
    print(f"- capacity envelope: {capacity_line_path}")


if __name__ == "__main__":
    main()
