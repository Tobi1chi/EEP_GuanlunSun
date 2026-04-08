import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sim" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eep_sim import BatterySpec, simulate_soc_series, soc_step  # noqa: E402


class TestSOC(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = BatterySpec(
            energy_capacity_mwh=1.0,
            charge_power_max_mw=0.5,
            discharge_power_max_mw=0.5,
            eta_charge=0.95,
            eta_discharge=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )

    def test_charge_step(self) -> None:
        result = soc_step(
            soc_prev=0.5,
            requested_power_mw=0.3,
            dt_hours=1.0,
            spec=self.spec,
        )
        self.assertAlmostEqual(result.soc_next, 0.785, places=6)

    def test_charge_soc_ceiling(self) -> None:
        result = soc_step(
            soc_prev=0.85,
            requested_power_mw=0.5,
            dt_hours=1.0,
            spec=self.spec,
        )
        self.assertAlmostEqual(result.soc_next, 0.9, places=6)
        self.assertGreater(result.curtailed_power_mw, 0.0)

    def test_discharge_step(self) -> None:
        result = soc_step(
            soc_prev=0.8,
            requested_power_mw=-0.5,
            dt_hours=1.0,
            spec=self.spec,
        )
        expected_soc = 0.8 - (0.5 / 0.95) / 1.0
        self.assertAlmostEqual(result.soc_next, expected_soc, places=6)

    def test_discharge_soc_floor(self) -> None:
        result = soc_step(
            soc_prev=0.12,
            requested_power_mw=-0.5,
            dt_hours=1.0,
            spec=self.spec,
        )
        self.assertAlmostEqual(result.soc_next, 0.1, places=6)
        self.assertLess(result.curtailed_power_mw, 0.0)

    def test_series(self) -> None:
        series = [0.5, 0.5, -0.4, -0.4, 0.2]
        results = simulate_soc_series(
            soc_initial=0.5,
            requested_power_series_mw=series,
            dt_hours=0.5,
            spec=self.spec,
        )
        self.assertEqual(len(results), len(series))
        self.assertTrue(all(self.spec.soc_min <= r.soc_next <= self.spec.soc_max for r in results))


if __name__ == "__main__":
    unittest.main()
