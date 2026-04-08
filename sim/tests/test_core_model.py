import sys
import unittest
from pathlib import Path

import pyomo.environ as pyo


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sim" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eep_sim.core_model import (  # noqa: E402
    CoreModelData,
    build_core_model,
    extract_solution,
    solve_core_model,
)


def _demo_data() -> CoreModelData:
    sites = ["A", "B"]
    hours = list(range(6))
    windows = [0, 1, 2]
    hour_to_window = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    load = {(i, t): (1.0 if i == "A" else 0.8) for i in sites for t in hours}
    price = {(i, t): (300.0 if t in (2, 3, 4) else 150.0) for i in sites for t in hours}
    grid_limit = {(i, t): 0.7 for i in sites for t in hours}
    penalty = {(i, t): 2000.0 for i in sites for t in hours}

    fixed_capacity = {"A": 0.4, "B": 0.4}
    soc_initial = {"A": 0.5, "B": 0.5}
    soc_min = {"A": 0.1, "B": 0.1}
    soc_max = {"A": 0.9, "B": 0.9}
    charge_rate = {"A": 1.0, "B": 1.0}
    discharge_rate = {"A": 1.0, "B": 1.0}

    active_init = {("A", 0): 0.2, ("B", 0): 0.0}
    reconf_limit = {0: 1e9, 1: 0.4, 2: 0.4}

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh=fixed_capacity,
        soc_initial_frac=soc_initial,
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
        charge_c_rate=charge_rate,
        discharge_c_rate=discharge_rate,
        eta_charge=0.95,
        eta_discharge=0.95,
        dt_hours=1.0,
        m_total_mwh=0.6,
        delay_windows=1,
        active_init_mwh=active_init,
        reconf_limit_mwh=reconf_limit,
        c_reconf_yuan_per_mwh=20.0,
    )


def _efficiency_data() -> CoreModelData:
    sites = ["A"]
    hours = [0, 1, 2]
    windows = [0]
    hour_to_window = {0: 0, 1: 0, 2: 0}

    load = {("A", 0): 0.0, ("A", 1): 1.0, ("A", 2): 0.0}
    price = {("A", 0): 10.0, ("A", 1): 1000.0, ("A", 2): 10.0}
    grid_limit = {("A", 0): 1.0, ("A", 1): 0.28, ("A", 2): 1.0}
    penalty = {("A", 0): 10000.0, ("A", 1): 10000.0, ("A", 2): 10000.0}

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh={"A": 1.0},
        soc_initial_frac={"A": 0.1},
        soc_min_frac={"A": 0.1},
        soc_max_frac={"A": 0.9},
        charge_c_rate={"A": 1.0},
        discharge_c_rate={"A": 1.0},
        eta_charge=0.8,
        eta_discharge=0.9,
        dt_hours=1.0,
        m_total_mwh=0.0,
        delay_windows=0,
        active_init_mwh={},
        reconf_limit_mwh={0: 1e9},
        c_reconf_yuan_per_mwh=0.0,
    )


def _mobile_transfer_data() -> CoreModelData:
    sites = ["A", "B"]
    hours = [0, 1, 2, 3]
    windows = [0, 1]
    hour_to_window = {0: 0, 1: 0, 2: 1, 3: 1}

    load = {(i, t): 0.0 for i in sites for t in hours}
    price = {(i, t): 10.0 for i in sites for t in hours}
    grid_limit = {(i, t): 10.0 for i in sites for t in hours}
    penalty = {(i, t): 1000.0 for i in sites for t in hours}

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh={"A": 0.0, "B": 0.0},
        soc_initial_frac={"A": 0.5, "B": 0.5},
        soc_min_frac={"A": 0.0, "B": 0.0},
        soc_max_frac={"A": 1.0, "B": 1.0},
        charge_c_rate={"A": 1.0, "B": 1.0},
        discharge_c_rate={"A": 1.0, "B": 1.0},
        eta_charge=1.0,
        eta_discharge=1.0,
        dt_hours=1.0,
        m_total_mwh=1.0,
        delay_windows=0,
        active_init_mwh={},
        reconf_limit_mwh={0: 1e9, 1: 1e9},
        c_reconf_yuan_per_mwh=0.0,
    )


def _static_mobile_capacity_data() -> CoreModelData:
    sites = ["A", "B"]
    hours = [0, 1, 2, 3]
    windows = [0, 1]
    hour_to_window = {0: 0, 1: 0, 2: 1, 3: 1}

    load = {
        ("A", 0): 0.0,
        ("A", 1): 0.0,
        ("A", 2): 0.5,
        ("A", 3): 0.0,
        ("B", 0): 0.0,
        ("B", 1): 0.0,
        ("B", 2): 0.0,
        ("B", 3): 0.0,
    }
    price = {(i, t): 10.0 for i in sites for t in hours}
    grid_limit = {(i, t): 0.0 for i in sites for t in hours}
    penalty = {(i, t): 1000.0 for i in sites for t in hours}

    return CoreModelData(
        sites=sites,
        hours=hours,
        windows=windows,
        hour_to_window=hour_to_window,
        load_mw=load,
        price_yuan_per_mwh=price,
        grid_limit_mw=grid_limit,
        unserved_penalty_yuan_per_mwh=penalty,
        fixed_capacity_mwh={"A": 0.0, "B": 0.0},
        soc_initial_frac={"A": 0.0, "B": 1.0},
        soc_min_frac={"A": 0.0, "B": 0.0},
        soc_max_frac={"A": 1.0, "B": 1.0},
        charge_c_rate={"A": 1.0, "B": 1.0},
        discharge_c_rate={"A": 1.0, "B": 1.0},
        eta_charge=1.0,
        eta_discharge=1.0,
        dt_hours=1.0,
        m_total_mwh=1.0,
        delay_windows=0,
        active_init_mwh={},
        reconf_limit_mwh={0: 1e9, 1: 1e9},
        c_reconf_yuan_per_mwh=0.0,
    )


def _terminal_fixed_discharge_data() -> CoreModelData:
    return CoreModelData(
        sites=["A"],
        hours=[0, 1],
        windows=[0],
        hour_to_window={0: 0, 1: 0},
        load_mw={("A", 0): 0.0, ("A", 1): 1.0},
        price_yuan_per_mwh={("A", 0): 1000.0, ("A", 1): 1000.0},
        grid_limit_mw={("A", 0): 0.0, ("A", 1): 0.0},
        unserved_penalty_yuan_per_mwh={("A", 0): 10000.0, ("A", 1): 10000.0},
        fixed_capacity_mwh={"A": 1.0},
        soc_initial_frac={"A": 0.1},
        soc_min_frac={"A": 0.1},
        soc_max_frac={"A": 0.9},
        charge_c_rate={"A": 1.0},
        discharge_c_rate={"A": 1.0},
        eta_charge=1.0,
        eta_discharge=1.0,
        dt_hours=1.0,
        m_total_mwh=0.0,
        delay_windows=0,
        active_init_mwh={},
        reconf_limit_mwh={0: 1e9},
        c_reconf_yuan_per_mwh=0.0,
    )


def _terminal_mobile_discharge_data() -> CoreModelData:
    return CoreModelData(
        sites=["A"],
        hours=[0, 1],
        windows=[0],
        hour_to_window={0: 0, 1: 0},
        load_mw={("A", 0): 0.0, ("A", 1): 1.0},
        price_yuan_per_mwh={("A", 0): 1000.0, ("A", 1): 1000.0},
        grid_limit_mw={("A", 0): 0.0, ("A", 1): 0.0},
        unserved_penalty_yuan_per_mwh={("A", 0): 10000.0, ("A", 1): 10000.0},
        fixed_capacity_mwh={"A": 0.0},
        soc_initial_frac={"A": 0.1},
        soc_min_frac={"A": 0.1},
        soc_max_frac={"A": 0.9},
        charge_c_rate={"A": 1.0},
        discharge_c_rate={"A": 1.0},
        eta_charge=1.0,
        eta_discharge=1.0,
        dt_hours=1.0,
        m_total_mwh=1.0,
        delay_windows=0,
        active_init_mwh={},
        reconf_limit_mwh={0: 1e9},
        c_reconf_yuan_per_mwh=0.0,
    )


class TestCoreModel(unittest.TestCase):
    def test_build_and_solve(self) -> None:
        data = _demo_data()
        model = build_core_model(data)
        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        sol = extract_solution(model)
        self.assertGreaterEqual(sol["costs"]["C_total"], 0.0)

    def test_mobile_pool_constraint(self) -> None:
        data = _demo_data()
        model = build_core_model(data)
        solve_core_model(model, solver_name="highs")

        for w in model.W:
            total_x = sum(pyo.value(model.x[i, w]) for i in model.I)
            self.assertLessEqual(total_x, pyo.value(model.m_total) + 1e-6)

    def test_delay_and_reconf_hard_limit(self) -> None:
        data = _demo_data()
        model = build_core_model(data)
        solve_core_model(model, solver_name="highs")

        # Delay: M_active[i,w] == x[i,w-1] for w>=1 under delay=1
        for i in model.I:
            for w in model.W:
                w_int = int(w)
                if w_int == 0:
                    self.assertAlmostEqual(
                        pyo.value(model.m_active[i, w_int]),
                        data.active_init_mwh.get((i, w_int), 0.0),
                        places=6,
                    )
                else:
                    self.assertAlmostEqual(
                        pyo.value(model.m_active[i, w_int]),
                        pyo.value(model.x[i, w_int - 1]),
                        places=6,
                    )

        # Hard reconfiguration bound
        for w in model.WM:
            lhs = sum(
                abs(pyo.value(model.x[i, int(w)]) - pyo.value(model.x[i, int(w) - 1])) for i in model.I
            )
            rhs = pyo.value(model.reconf_limit[w])
            self.assertLessEqual(lhs, rhs + 1e-6)

    def test_charge_discharge_mutual_exclusivity(self) -> None:
        data = _efficiency_data()
        model = build_core_model(data)
        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        for i in model.I:
            for t in model.T:
                p_ch = pyo.value(model.p_ch[i, t])
                p_dis = pyo.value(model.p_dis[i, t])
                self.assertTrue(
                    p_ch <= 1e-7 or p_dis <= 1e-7,
                    msg=f"simultaneous charge/discharge at {(i, t)}: {p_ch}, {p_dis}",
                )

    def test_charge_discharge_efficiency_constraints(self) -> None:
        data = _efficiency_data()
        model = build_core_model(data)
        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        self.assertAlmostEqual(pyo.value(model.p_ch["A", 0]), 1.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc["A", 0]), 0.9, places=6)
        self.assertAlmostEqual(pyo.value(model.p_dis["A", 1]), 0.72, places=6)
        self.assertAlmostEqual(pyo.value(model.soc["A", 1]), 0.1, places=6)
        self.assertAlmostEqual(pyo.value(model.unserved["A", 1]), 0.0, places=6)

    def test_mobile_energy_conserved_across_reconfiguration(self) -> None:
        data = _mobile_transfer_data()
        model = build_core_model(data)

        model.x["A", 0].fix(1.0)
        model.x["B", 0].fix(0.0)
        model.x["A", 1].fix(0.0)
        model.x["B", 1].fix(1.0)

        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        # Mobile energy must move from A to B at the window boundary instead of
        # appearing/disappearing when active capacity is reallocated.
        self.assertAlmostEqual(pyo.value(model.soc_mobile["A", 1]), 0.5, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_mobile["B", 1]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_mobile["A", 2]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_mobile["B", 2]), 0.5, places=6)

        energy_shift_total = sum(
            pyo.value(model.energy_shift_pos[i, 1]) - pyo.value(model.energy_shift_neg[i, 1])
            for i in model.I
        )
        self.assertAlmostEqual(energy_shift_total, 0.0, places=6)

    def test_no_energy_shift_without_capacity_reconfiguration(self) -> None:
        data = _static_mobile_capacity_data()
        model = build_core_model(data)

        for w in model.W:
            model.x["A", w].fix(0.5)
            model.x["B", w].fix(0.5)

        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        for i in model.I:
            self.assertAlmostEqual(pyo.value(model.energy_shift_pos[i, 1]), 0.0, places=6)
            self.assertAlmostEqual(pyo.value(model.energy_shift_neg[i, 1]), 0.0, places=6)

        self.assertAlmostEqual(pyo.value(model.soc_mobile["A", 2]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_mobile["B", 2]), 0.5, places=6)
        self.assertAlmostEqual(pyo.value(model.unserved["A", 2]), 0.5, places=6)

    def test_terminal_hour_cannot_discharge_fixed_soc_below_floor(self) -> None:
        data = _terminal_fixed_discharge_data()
        model = build_core_model(data)

        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        self.assertAlmostEqual(pyo.value(model.p_dis_fixed["A", 1]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_fixed["A", 1]), 0.1, places=6)
        self.assertAlmostEqual(pyo.value(model.unserved["A", 1]), 1.0, places=6)

    def test_terminal_hour_cannot_discharge_mobile_soc_below_floor(self) -> None:
        data = _terminal_mobile_discharge_data()
        model = build_core_model(data)
        model.x["A", 0].fix(1.0)

        results = solve_core_model(model, solver_name="highs")
        self.assertEqual(str(results.solver.termination_condition).lower(), "optimal")

        self.assertAlmostEqual(pyo.value(model.p_dis_mobile["A", 1]), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(model.soc_mobile["A", 1]), 0.1, places=6)
        self.assertAlmostEqual(pyo.value(model.unserved["A", 1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
