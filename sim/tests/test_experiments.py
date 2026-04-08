import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sim" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eep_sim.experiments import (  # noqa: E402
    ExperimentScenario,
    SyntheticCorridorConfig,
    _attach_decomposition,
    _finalize_experiment,
    _summarize_variants,
    build_synthetic_corridor_data,
    make_case_data,
    prepare_capacity_fair_mobile_data,
    prepare_same_budget_data,
    run_case_bundle,
)


class TestExperiments(unittest.TestCase):
    def test_make_case_data_switches_storage_modes(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))

        no_storage = make_case_data(data, "no_storage")
        fixed_only = make_case_data(data, "fixed_only")
        mobile_only = make_case_data(data, "mobile_only")
        hybrid = make_case_data(data, "hybrid")

        self.assertEqual(sum(no_storage.fixed_capacity_mwh.values()), 0.0)
        self.assertEqual(no_storage.m_total_mwh, 0.0)

        self.assertGreater(sum(fixed_only.fixed_capacity_mwh.values()), 0.0)
        self.assertEqual(fixed_only.m_total_mwh, 0.0)

        self.assertEqual(sum(mobile_only.fixed_capacity_mwh.values()), 0.0)
        self.assertGreater(mobile_only.m_total_mwh, 0.0)

        self.assertGreater(sum(hybrid.fixed_capacity_mwh.values()), 0.0)
        self.assertGreater(hybrid.m_total_mwh, 0.0)

    def test_run_case_bundle_returns_all_cases_with_metrics(self) -> None:
        data = build_synthetic_corridor_data(
            SyntheticCorridorConfig(
                windows=2,
                reconf_hours=3,
                delay_windows=1,
                concentration=0.8,
                residence_windows=1,
                hotspot_extra_mw=1.2,
            )
        )
        metrics = run_case_bundle(
            experiment_id="T1",
            question="test",
            track="synthetic",
            source="synthetic",
            variant_id="bundle",
            base_data=data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
            include_ideal_mobile=True,
        )
        cases = {metric.case for metric in metrics}
        self.assertEqual(cases, {"no_storage", "fixed_only", "mobile_only", "hybrid", "mobile_only_ideal"})
        mobile_row = next(metric for metric in metrics if metric.case == "mobile_only")
        self.assertIn(mobile_row.case_outcome, {"optimal", "feasible_with_gap"})
        self.assertIsNotNone(mobile_row.best_bound)
        self.assertIsNotNone(mobile_row.elapsed_seconds)
        self.assertIsNotNone(mobile_row.lp_relaxation_objective)
        self.assertGreaterEqual(mobile_row.SC_1, 0.0)
        self.assertGreaterEqual(mobile_row.HT, 0.0)
        self.assertGreaterEqual(mobile_row.R_delay, 0.0)
        self.assertGreaterEqual(mobile_row.V_reconf_norm, 0.0)

    def test_prepare_capacity_fair_mobile_data_matches_fixed_total(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=3, reconf_hours=3, delay_windows=1))
        fair_data = prepare_capacity_fair_mobile_data(data)

        self.assertAlmostEqual(
            fair_data.m_total_mwh,
            sum(data.fixed_capacity_mwh.values()),
            places=6,
        )
        self.assertTrue(fair_data.active_init_mwh)
        expected_share = fair_data.m_total_mwh / len(fair_data.sites)
        for value in fair_data.active_init_mwh.values():
            self.assertAlmostEqual(value, expected_share, places=6)

    def test_prepare_same_budget_data_respects_budget_identity(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=3, reconf_hours=3, delay_windows=1))
        planning_data = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=0.4,
        )

        fixed_total = sum(planning_data.fixed_capacity_mwh.values())
        mobile_total = planning_data.m_total_mwh
        baseline_budget = sum(data.fixed_capacity_mwh.values())

        self.assertAlmostEqual(fixed_total + 1.5 * mobile_total, baseline_budget, places=6)
        self.assertEqual(planning_data.objective_mode, "planning")
        self.assertAlmostEqual(planning_data.c_storage_fixed_yuan_per_mwh, 1.0, places=6)
        self.assertAlmostEqual(planning_data.c_storage_mobile_yuan_per_mwh, 1.5, places=6)

    def test_same_budget_total_capacity_is_monotone_in_fixed_share(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=3, reconf_hours=3, delay_windows=1))
        mobile_only = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=0.0,
        )
        hybrid = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=0.5,
        )
        fixed_only = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=1.0,
        )

        mobile_total_capacity = sum(mobile_only.fixed_capacity_mwh.values()) + mobile_only.m_total_mwh
        hybrid_total_capacity = sum(hybrid.fixed_capacity_mwh.values()) + hybrid.m_total_mwh
        fixed_total_capacity = sum(fixed_only.fixed_capacity_mwh.values()) + fixed_only.m_total_mwh

        self.assertLessEqual(mobile_total_capacity, hybrid_total_capacity)
        self.assertLessEqual(hybrid_total_capacity, fixed_total_capacity)

    def test_same_budget_data_scales_with_budget_multiplier(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=3, reconf_hours=3, delay_windows=1))
        base_budget = sum(data.fixed_capacity_mwh.values())
        lower = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=0.5,
            budget_in_fixed_cost_units=base_budget,
        )
        higher = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=0.5,
            budget_in_fixed_cost_units=1.5 * base_budget,
        )

        lower_total = sum(lower.fixed_capacity_mwh.values()) + lower.m_total_mwh
        higher_total = sum(higher.fixed_capacity_mwh.values()) + higher.m_total_mwh
        self.assertGreater(higher_total, lower_total)

    def test_planning_mode_metrics_use_total_with_storage_objective(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))
        planning_data = prepare_same_budget_data(
            data,
            mobile_cost_premium_ratio=1.5,
            fixed_budget_share=1.0,
        )
        metrics = run_case_bundle(
            experiment_id="TP",
            question="planning objective",
            track="synthetic",
            source="synthetic",
            variant_id="planning_bundle",
            base_data=planning_data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
        )
        fixed_row = next(metric for metric in metrics if metric.case == "fixed_only")
        self.assertAlmostEqual(fixed_row.C_total, fixed_row.C_total_with_storage, places=6)
        self.assertGreaterEqual(fixed_row.C_total, fixed_row.C_grid)

    def test_decomposition_attachment_and_summary(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))
        metrics = run_case_bundle(
            experiment_id="E8",
            question="decomposition",
            track="synthetic",
            source="synthetic",
            variant_id="decomp",
            base_data=data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
            include_ideal_mobile=True,
        )
        _attach_decomposition(metrics)
        rows = [metric.to_row() for metric in metrics if metric.case != "mobile_only_ideal"]
        summary_rows = _summarize_variants(rows)
        self.assertEqual(len(summary_rows), 1)
        summary = summary_rows[0]
        self.assertIn("V_mob_net", summary)
        self.assertIn("B_reuse", summary)
        self.assertIn("L_friction", summary)
        self.assertAlmostEqual(
            float(summary["V_mob_net"]),
            float(summary["B_reuse"]) - float(summary["L_friction"]),
            places=6,
        )
        self.assertIn("max_gap", summary)
        self.assertIn("max_elapsed_seconds", summary)

    def test_summary_uses_case_specific_capacity_fields(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))
        metrics = run_case_bundle(
            experiment_id="TS",
            question="summary capacities",
            track="synthetic",
            source="synthetic",
            variant_id="summary_variant",
            base_data=data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
        )
        summary_rows = _summarize_variants([metric.to_row() for metric in metrics])
        self.assertEqual(len(summary_rows), 1)
        summary = summary_rows[0]
        self.assertGreater(float(summary["fixed_total_mwh"]), 0.0)
        self.assertGreater(float(summary["mobile_total_mwh"]), 0.0)
        self.assertGreater(float(summary["hybrid_fixed_total_mwh"]), 0.0)
        self.assertGreater(float(summary["hybrid_mobile_total_mwh"]), 0.0)

    def test_timeout_without_incumbent_is_excluded_from_summary(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))
        metrics = run_case_bundle(
            experiment_id="E2",
            question="screening",
            track="synthetic",
            source="synthetic",
            variant_id="variant",
            base_data=data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
        )
        rows = [metric.to_row() for metric in metrics]
        rows[0]["case_outcome"] = "timeout_no_incumbent"
        summary_rows = _summarize_variants(rows)
        self.assertEqual(summary_rows, [])

    def test_finalize_experiment_writes_output_contract(self) -> None:
        data = build_synthetic_corridor_data(SyntheticCorridorConfig(windows=2, reconf_hours=3))
        metrics = run_case_bundle(
            experiment_id="TX",
            question="contract",
            track="synthetic",
            source="synthetic",
            variant_id="contract_variant",
            base_data=data,
            timeline_mode="synthetic_hourly",
            reconf_hours=3,
            delay_windows=1,
            c_reconf_yuan_per_mwh=0.5,
            solver_name="highs",
        )
        summary_rows = _summarize_variants([metric.to_row() for metric in metrics])
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            figure_path = base_dir / "figures" / "experiments" / "tx.png"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.write_bytes(b"fake")
            artifact = _finalize_experiment(
                base_dir=base_dir,
                experiment_id="TX",
                question="contract",
                metrics=metrics,
                summary_rows=summary_rows,
                figure_paths=[figure_path],
                notes=["test note"],
            )
            self.assertTrue(artifact.raw_csv.exists())
            self.assertTrue(artifact.summary_csv.exists())
            self.assertTrue(artifact.memo_md.exists())
            self.assertEqual(len(artifact.figure_paths), 1)


if __name__ == "__main__":
    unittest.main()
