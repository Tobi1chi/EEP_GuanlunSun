"""Solve a capacity-fair fixed-versus-mobile baseline on the real corridor data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.core_model import solve_core_model_with_diagnostics  # noqa: E402
from eep_sim.experiments import make_case_data, prepare_capacity_fair_mobile_data  # noqa: E402
from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-only and mobile-only under equal total storage capacity"
    )
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--reconf-hours", type=int, default=24)
    parser.add_argument("--delay-windows", type=int, default=1)
    parser.add_argument("--time-limit-seconds", type=float, default=240.0)
    parser.add_argument("--lp-time-limit-seconds", type=float, default=30.0)
    args = parser.parse_args()

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
    fair_data = prepare_capacity_fair_mobile_data(data)

    fixed_case = make_case_data(fair_data, "fixed_only")
    mobile_case = make_case_data(fair_data, "mobile_only")

    fixed_diag, fixed_sol = solve_core_model_with_diagnostics(
        fixed_case,
        solver_name=args.solver,
        time_limit_seconds=args.time_limit_seconds,
        lp_time_limit_seconds=args.lp_time_limit_seconds,
    )
    mobile_diag, mobile_sol = solve_core_model_with_diagnostics(
        mobile_case,
        solver_name=args.solver,
        time_limit_seconds=args.time_limit_seconds,
        lp_time_limit_seconds=args.lp_time_limit_seconds,
    )

    if fixed_sol is None or mobile_sol is None:
        raise RuntimeError(
            "Capacity-fair baseline demo requires feasible incumbents for both fixed-only and mobile-only."
        )

    fixed_total = sum(data.fixed_capacity_mwh.values())
    fixed_cost = float(fixed_sol["costs"]["C_total"])
    mobile_cost = float(mobile_sol["costs"]["C_total"])
    fixed_unserved = float(sum(fixed_sol["unserved"].values()))
    mobile_unserved = float(sum(mobile_sol["unserved"].values()))

    print("capacity-fair baseline summary:")
    print(f"- sites: {diag.sites}")
    print(f"- timeline_mode: {diag.timeline_mode}")
    print(f"- fair_total_storage_mwh: {fixed_total:.1f}")
    print(
        f"- fixed_only: C_total={fixed_cost:.6f}, unmet_load={fixed_unserved:.6f}, "
        f"case_outcome={fixed_diag.case_outcome}"
    )
    print(
        f"- mobile_only: C_total={mobile_cost:.6f}, unmet_load={mobile_unserved:.6f}, "
        f"case_outcome={mobile_diag.case_outcome}"
    )
    print(f"- delta_C_total_fixed_minus_mobile: {fixed_cost - mobile_cost:.6f}")
    print(f"- delta_unmet_load_fixed_minus_mobile: {fixed_unserved - mobile_unserved:.6f}")
    print("")
    print(
        "Interpretation: this check isolates deployment logic from the default 0.7 mobile-to-fixed "
        "sizing rule used in the packaged baseline experiments."
    )


if __name__ == "__main__":
    main()
