"""Worker process for solving a single experiment scenario."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    SRC = Path(__file__).resolve().parents[1]
    ROOT = Path(__file__).resolve().parents[3]
    for candidate in (ROOT, SRC):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from eep_sim.experiments import (  # noqa: E402
    _solve_scenario_detail_inprocess,
    deserialize_core_model_data,
    deserialize_experiment_scenario,
    serialize_scenario_metrics,
)


def _debug(message: str) -> None:
    if os.environ.get("EEP_WORKER_DEBUG") == "1":
        print(f"[worker] {message}", file=sys.stderr, flush=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: solve_case_worker.py <input.pkl> <output.pkl>")
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    _debug(f"loading {input_path}")
    payload = pickle.loads(input_path.read_bytes())
    _debug("deserializing request")
    data = deserialize_core_model_data(payload["data"])
    scenario = deserialize_experiment_scenario(payload["scenario"])
    _debug(f"solving {scenario.experiment_id}:{scenario.variant_id}:{scenario.case}")
    metrics, solution = _solve_scenario_detail_inprocess(
        data,
        scenario,
        solver_name=payload["solver_name"],
        include_solution=bool(payload.get("include_solution", True)),
    )
    _debug("writing output")
    result_payload = {"metrics": serialize_scenario_metrics(metrics)}
    if solution is not None:
        result_payload["solution"] = solution
    output_path.write_bytes(pickle.dumps(result_payload, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
