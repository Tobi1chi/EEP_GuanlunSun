"""CLI entrypoint for the unified experiment suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from eep_sim.experiments import run_all_experiments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EEP experiment suite")
    parser.add_argument("--base-dir", default=".", help="Repository root containing data/raw and result folders")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--run-all", action="store_true", help="Run E0-E10 in sequence")
    parser.add_argument(
        "--experiments",
        default=None,
        help="Optional comma-separated subset such as E2,E3,E6",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse previously materialized outputs for experiments not selected in --experiments",
    )
    args = parser.parse_args()

    if not args.run_all:
        parser.error("--run-all is required")

    base_dir = Path(args.base_dir).resolve()
    experiment_ids = (
        [part.strip() for part in args.experiments.split(",") if part.strip()]
        if args.experiments
        else None
    )
    result = run_all_experiments(
        base_dir,
        solver_name=args.solver,
        experiment_ids=experiment_ids,
        reuse_existing=args.reuse_existing,
    )
    print("experiment suite completed")
    print(f"- manifest: {result['manifest_path']}")
    print(f"- master: {result['master_path']}")
    print(f"- handoff: {result['handoff_path']}")


if __name__ == "__main__":
    main()
