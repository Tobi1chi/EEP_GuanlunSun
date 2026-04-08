import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sim" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eep_sim.g2_loader import G2LoaderOptions, load_g2_core_model_data  # noqa: E402


def _expanded_hour_index(month: int, day: int, hod: int) -> int:
    month_day_counts = {
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
    days_before = sum(month_day_counts[m] for m in range(1, month))
    return (days_before + (day - 1)) * 24 + hod


def _required_input_paths() -> list[Path]:
    return [
        ROOT / "data" / "raw" / "g2_beijing_shanghai_service_areas_hourly_load_revised.csv",
        ROOT / "data" / "raw" / "g2_beijing_shanghai_service_areas_hourly_price_absolute_corrected.csv",
        ROOT / "data" / "raw" / "g2_beijing_shanghai_service_areas_load_grid_params_revised.csv",
    ]


class TestG2Loader(unittest.TestCase):
    def test_load_g2_core_model_data_representative(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            price_duplicate_policy="max",
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, diag = load_g2_core_model_data(ROOT, options=options)

        self.assertEqual(len(data.sites), 23)
        self.assertEqual(len(data.hours), 288)
        self.assertEqual(len(data.windows), 12)
        self.assertEqual(diag.timeline_hours, 288)
        self.assertEqual(diag.timeline_mode, "representative_12x24")
        self.assertEqual(diag.window_size_steps, 24)
        self.assertGreater(diag.duplicate_price_keys, 0)
        self.assertEqual(diag.fixed_total_mwh, 18.0)
        self.assertEqual(diag.mobile_total_mwh, 13.0)
        self.assertEqual(data.m_total_mwh, 13.0)
        self.assertAlmostEqual(data.grid_limit_mw[("Dezhou Service Area", 0)], 3.0)
        self.assertAlmostEqual(
            data.unserved_penalty_yuan_per_mwh[("Qingxian Service Area", 0)],
            1000.0,
        )
        self.assertAlmostEqual(
            data.unserved_penalty_yuan_per_mwh[("Qingxian Service Area", 100)],
            1000.0,
        )

        # Core maps should be dense over (site, t)
        expected_n = len(data.sites) * len(data.hours)
        self.assertEqual(len(data.load_mw), expected_n)
        self.assertEqual(len(data.price_yuan_per_mwh), expected_n)
        self.assertEqual(len(data.grid_limit_mw), expected_n)
        self.assertEqual(len(data.unserved_penalty_yuan_per_mwh), expected_n)

        # Ensure window mapping is valid
        for t in data.hours:
            self.assertIn(data.hour_to_window[t], data.windows)

    def test_load_g2_core_model_data_expanded_8760(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            timeline_mode="expanded_8760",
            reconf_hours=168,  # weekly window on 1h timeline
            delay_windows=1,
            price_duplicate_policy="max",
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, diag = load_g2_core_model_data(ROOT, options=options)

        self.assertEqual(len(data.hours), 8760)
        self.assertEqual(diag.timeline_hours, 8760)
        self.assertEqual(diag.timeline_mode, "expanded_8760")
        self.assertEqual(diag.window_size_steps, 168)
        self.assertEqual(len(data.windows), 53)  # ceil(8760 / 168)
        self.assertEqual(diag.mobile_total_mwh, 13.0)

        holiday_t = _expanded_hour_index(10, 1, 10)
        non_holiday_t = _expanded_hour_index(10, 8, 10)
        self.assertGreater(
            data.load_mw[("Dezhou Service Area", holiday_t)],
            data.load_mw[("Dezhou Service Area", non_holiday_t)],
        )

    def test_explicit_mobile_total_overrides_ratio(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            m_total_mwh=11.0,
            mobile_capacity_ratio_to_fixed=0.1,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, diag = load_g2_core_model_data(ROOT, options=options)

        self.assertEqual(diag.fixed_total_mwh, 18.0)
        self.assertEqual(diag.mobile_total_mwh, 11.0)
        self.assertEqual(data.m_total_mwh, 11.0)

    def test_disable_holiday_shocks_keeps_baseline_load(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            timeline_mode="expanded_8760",
            reconf_hours=168,
            apply_cn_holiday_shocks=False,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, _diag = load_g2_core_model_data(ROOT, options=options)

        holiday_t = _expanded_hour_index(10, 1, 10)
        non_holiday_t = _expanded_hour_index(10, 8, 10)
        self.assertAlmostEqual(
            data.load_mw[("Dezhou Service Area", holiday_t)],
            data.load_mw[("Dezhou Service Area", non_holiday_t)],
        )

    def test_price_multiplier_penalty_mode(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            symbolic_unserved_penalty_mode="price_multiplier",
            symbolic_unserved_penalty_price_multiplier=20.0,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, _diag = load_g2_core_model_data(ROOT, options=options)

        qingxian_prices = [
            data.price_yuan_per_mwh[("Qingxian Service Area", t)] for t in data.hours
        ]
        self.assertAlmostEqual(
            data.unserved_penalty_yuan_per_mwh[("Qingxian Service Area", 0)],
            max(qingxian_prices) * 20.0,
        )

    def test_monthly_load_multiplier_applies_in_representative_timeline(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            month_load_multiplier_by_month={1: 0.8, 2: 1.2},
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, _diag = load_g2_core_model_data(ROOT, options=options)

        # representative_12x24 timeline is month-major with 24 hours per month
        self.assertAlmostEqual(data.load_mw[("Dezhou Service Area", 0)], 2.25 * 0.8)
        self.assertAlmostEqual(data.load_mw[("Dezhou Service Area", 24)], 2.25 * 1.2)

    def test_monthly_load_multiplier_applies_in_expanded_timeline(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            timeline_mode="expanded_8760",
            reconf_hours=168,
            apply_cn_holiday_shocks=False,
            month_load_multiplier_by_month={1: 0.9, 10: 1.3},
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, _diag = load_g2_core_model_data(ROOT, options=options)

        jan_t = _expanded_hour_index(1, 8, 10)
        oct_t = _expanded_hour_index(10, 8, 10)
        base_load = 1.8  # Dezhou hour=10 from current revised profile
        self.assertAlmostEqual(data.load_mw[("Dezhou Service Area", jan_t)], base_load * 0.9)
        self.assertAlmostEqual(data.load_mw[("Dezhou Service Area", oct_t)], base_load * 1.3)

    def test_grid_limit_uses_transformed_average_load(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            grid_capacity_mode="avg_load_margin",
            grid_capacity_margin_above_avg_load=0.25,
            base_load_scale=0.85,
            month_load_multiplier_by_month={1: 0.8, 2: 1.2},
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        data, _diag = load_g2_core_model_data(ROOT, options=options)

        avg_load = sum(data.load_mw[("Dezhou Service Area", t)] for t in data.hours) / len(data.hours)
        expected_limit = avg_load * 1.25
        self.assertAlmostEqual(data.grid_limit_mw[("Dezhou Service Area", 0)], expected_limit, places=6)
        self.assertAlmostEqual(data.grid_limit_mw[("Dezhou Service Area", 100)], expected_limit, places=6)

    def test_base_load_scale_reduces_nominal_load(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        base_options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        scaled_options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            base_load_scale=0.85,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=False,
        )
        base_data, _ = load_g2_core_model_data(ROOT, options=base_options)
        scaled_data, _ = load_g2_core_model_data(ROOT, options=scaled_options)

        self.assertAlmostEqual(
            scaled_data.load_mw[("Dezhou Service Area", 0)],
            base_data.load_mw[("Dezhou Service Area", 0)] * 0.85,
        )

    def test_site_month_heterogeneity_creates_cross_site_month_difference(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            reconf_hours=24,
            delay_windows=1,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=True,
            site_month_heterogeneity_strength=0.18,
            enable_holiday_site_stagger=False,
        )
        data, _ = load_g2_core_model_data(ROOT, options=options)

        # representative timeline is month-major; compare Jan vs Oct for two sites
        majuqiao_jan = data.load_mw[("Majuqiao Service Area", 0)]
        majuqiao_oct = data.load_mw[("Majuqiao Service Area", 9 * 24)]
        dezhou_jan = data.load_mw[("Dezhou Service Area", 0)]
        dezhou_oct = data.load_mw[("Dezhou Service Area", 9 * 24)]

        self.assertNotAlmostEqual(majuqiao_jan / majuqiao_oct, dezhou_jan / dezhou_oct)

    def test_holiday_site_stagger_creates_cross_site_holiday_difference(self) -> None:
        required = _required_input_paths()
        if not all(p.exists() for p in required):
            self.skipTest("g2 input files not found in workspace")

        options = G2LoaderOptions(
            timeline_mode="expanded_8760",
            reconf_hours=168,
            base_load_scale=1.0,
            enable_site_month_heterogeneity=False,
            enable_holiday_site_stagger=True,
            holiday_site_stagger_strength=0.22,
        )
        data, _ = load_g2_core_model_data(ROOT, options=options)

        holiday_t = _expanded_hour_index(10, 1, 10)
        non_holiday_t = _expanded_hour_index(10, 8, 10)
        dezhou_ratio = data.load_mw[("Dezhou Service Area", holiday_t)] / data.load_mw[("Dezhou Service Area", non_holiday_t)]
        majuqiao_ratio = data.load_mw[("Majuqiao Service Area", holiday_t)] / data.load_mw[("Majuqiao Service Area", non_holiday_t)]

        self.assertNotAlmostEqual(dezhou_ratio, majuqiao_ratio)


if __name__ == "__main__":
    unittest.main()
