"""Build CoreModelData from g2 files and print a concise summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import build_core_model  # noqa: E402
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load revised corridor input CSVs into CoreModelData")
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw and result folders")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--m-total-mwh", type=float, default=None)
    parser.add_argument("--mobile-capacity-ratio-to-fixed", type=float, default=0.7)
    parser.add_argument("--base-load-scale", type=float, default=0.85)
    parser.add_argument("--disable-site-month-heterogeneity", action="store_true")
    parser.add_argument("--site-month-heterogeneity-strength", type=float, default=0.18)
    parser.add_argument("--disable-holiday-site-stagger", action="store_true")
    parser.add_argument("--holiday-site-stagger-strength", type=float, default=0.22)
    parser.add_argument(
        "--grid-capacity-mode",
        choices=["csv", "avg_load_margin", "max_of_csv_and_avg_load_margin"],
        default="max_of_csv_and_avg_load_margin",
    )
    parser.add_argument("--grid-capacity-margin-above-avg-load", type=float, default=0.25)
    parser.add_argument(
        "--symbolic-unserved-penalty-mode",
        choices=["fixed", "price_multiplier", "opportunity_cost"],
        default="fixed",
    )
    parser.add_argument("--default-unserved-penalty-yuan-per-mwh", type=float, default=1000.0)
    parser.add_argument("--symbolic-unserved-penalty-price-multiplier", type=float, default=20.0)
    parser.add_argument("--opportunity-cost-price-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--timeline-mode",
        choices=["representative_12x24", "expanded_8760"],
        default="representative_12x24",
    )
    parser.add_argument("--holiday-year", type=int, default=2026)
    parser.add_argument("--disable-cn-holiday-shocks", action="store_true")
    parser.add_argument(
        "--month-load-multipliers-json",
        default=None,
        help='JSON object like \'{"1":0.9,"2":0.95,"7":1.1,"10":1.2}\'',
    )
    parser.add_argument(
        "--price-dup-policy",
        choices=["max", "min", "mean", "first"],
        default="max",
    )
    args = parser.parse_args()
    month_load_multiplier_by_month = (
        json.loads(args.month_load_multipliers_json) if args.month_load_multipliers_json else None
    )

    options = G2LoaderOptions(
        reconf_hours=args.reconf_hours,
        delay_windows=args.delay_windows,
        m_total_mwh=args.m_total_mwh,
        mobile_capacity_ratio_to_fixed=args.mobile_capacity_ratio_to_fixed,
        base_load_scale=args.base_load_scale,
        enable_site_month_heterogeneity=not args.disable_site_month_heterogeneity,
        site_month_heterogeneity_strength=args.site_month_heterogeneity_strength,
        enable_holiday_site_stagger=not args.disable_holiday_site_stagger,
        holiday_site_stagger_strength=args.holiday_site_stagger_strength,
        grid_capacity_mode=args.grid_capacity_mode,
        grid_capacity_margin_above_avg_load=args.grid_capacity_margin_above_avg_load,
        default_unserved_penalty_yuan_per_mwh=args.default_unserved_penalty_yuan_per_mwh,
        symbolic_unserved_penalty_mode=args.symbolic_unserved_penalty_mode,
        symbolic_unserved_penalty_price_multiplier=args.symbolic_unserved_penalty_price_multiplier,
        opportunity_cost_price_multiplier=args.opportunity_cost_price_multiplier,
        timeline_mode=args.timeline_mode,
        month_load_multiplier_by_month=month_load_multiplier_by_month,
        holiday_year=args.holiday_year,
        apply_cn_holiday_shocks=not args.disable_cn_holiday_shocks,
        price_duplicate_policy=args.price_dup_policy,
    )

    data, diag = load_g2_core_model_data(args.base_dir, options=options)
    model = build_core_model(data)

    print("g2 loader summary:")
    print(f"- sites: {diag.sites}")
    print(f"- months: {diag.months}")
    print(f"- hours_of_day: {diag.hours_of_day}")
    print(f"- timeline_hours: {diag.timeline_hours}")
    print(f"- windows: {diag.windows}")
    print(f"- window_size_steps: {diag.window_size_steps}")
    print(f"- timeline_mode: {diag.timeline_mode}")
    print(f"- duplicate_price_keys: {diag.duplicate_price_keys}")
    print(f"- duplicate_price_rows: {diag.duplicate_price_rows}")
    print(f"- price_duplicate_policy: {diag.price_duplicate_policy}")
    print(f"- fixed_total_mwh: {diag.fixed_total_mwh}")
    print(f"- mobile_total_mwh: {diag.mobile_total_mwh}")
    print("model built successfully:")
    print(f"- |I|={len(list(model.I))}, |T|={len(list(model.T))}, |W|={len(list(model.W))}")


if __name__ == "__main__":
    main()
