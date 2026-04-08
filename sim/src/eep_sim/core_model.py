"""Core optimization model for MESS reconfiguration + hourly operation.

Implemented core constraints:

1) Mobile capacity pool:
   sum_i x[i,w] <= M_total
2) Delayed activation:
   M_active[i,w] = x[i,w-delay] (or initial active capacity for early windows)
3) Hard reconfiguration bound:
   sum_i |x[i,w] - x[i,w-1]| <= R_w
4) Hourly operation:
   - power balance
   - fixed/mobile SOC dynamics
   - mobile-energy conservation at window boundaries
   - charge/discharge mutual exclusivity
   - charge/discharge limits
   - grid import limits
   - unserved load
5) Objective:
   C_grid + C_unserved + C_reconf
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import monotonic
from typing import Dict, Mapping, Sequence, Tuple

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs


IndexIT = Tuple[str, int]
IndexIW = Tuple[str, int]
GAP_EPS = 1e-9


@dataclass(frozen=True)
class CoreModelData:
    """Input data container for the core optimization model."""

    sites: Sequence[str]
    hours: Sequence[int]
    windows: Sequence[int]
    hour_to_window: Mapping[int, int]

    load_mw: Mapping[IndexIT, float]
    price_yuan_per_mwh: Mapping[IndexIT, float]
    grid_limit_mw: Mapping[IndexIT, float]
    unserved_penalty_yuan_per_mwh: Mapping[IndexIT, float]

    fixed_capacity_mwh: Mapping[str, float]
    soc_initial_frac: Mapping[str, float]
    soc_min_frac: Mapping[str, float]
    soc_max_frac: Mapping[str, float]
    charge_c_rate: Mapping[str, float]
    discharge_c_rate: Mapping[str, float]

    eta_charge: float
    eta_discharge: float
    dt_hours: float

    m_total_mwh: float
    delay_windows: int
    active_init_mwh: Mapping[IndexIW, float]
    reconf_limit_mwh: Mapping[int, float]

    c_reconf_yuan_per_mwh: float
    c_storage_fixed_yuan_per_mwh: float = 0.0
    c_storage_mobile_yuan_per_mwh: float = 0.0
    objective_mode: str = "operational"


@dataclass(frozen=True)
class CoreSolveDiagnostics:
    solver_name: str
    solver_status: str
    termination_condition: str
    case_outcome: str
    best_incumbent: float | None
    best_bound: float | None
    gap: float | None
    elapsed_seconds: float
    lp_relaxation_objective: float | None


def _validate_contiguous_index(values: Sequence[int], label: str) -> None:
    expected = list(range(len(values)))
    if list(values) != expected:
        raise ValueError(
            f"{label} must be contiguous integer indices starting from 0. "
            f"Expected {expected}, got {list(values)}"
        )


def _check_bounds(name: str, value: float, lower: float, upper: float) -> None:
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")


def _build_param_initializer(
    values: Mapping[Tuple[str, int], float],
    default: float = 0.0,
):
    def _init(_m, i, t):
        return float(values.get((i, int(t)), default))

    return _init


def _build_site_param_initializer(values: Mapping[str, float], default: float = 0.0):
    def _init(_m, i):
        return float(values.get(i, default))

    return _init


def _build_window_param_initializer(values: Mapping[int, float], default: float = 0.0):
    def _init(_m, w):
        return float(values.get(int(w), default))

    return _init


def _build_active_init_initializer(values: Mapping[IndexIW, float], default: float = 0.0):
    def _init(_m, i, w):
        return float(values.get((i, int(w)), default))

    return _init


def _validate_data(data: CoreModelData) -> None:
    if len(data.sites) == 0:
        raise ValueError("sites cannot be empty")
    if len(data.hours) == 0:
        raise ValueError("hours cannot be empty")
    if len(data.windows) == 0:
        raise ValueError("windows cannot be empty")

    _validate_contiguous_index(data.hours, "hours")
    _validate_contiguous_index(data.windows, "windows")

    if data.dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    if data.m_total_mwh < 0:
        raise ValueError("m_total_mwh must be >= 0")
    if data.delay_windows < 0:
        raise ValueError("delay_windows must be >= 0")
    if data.eta_charge <= 0 or data.eta_charge > 1:
        raise ValueError("eta_charge must be in (0, 1]")
    if data.eta_discharge <= 0 or data.eta_discharge > 1:
        raise ValueError("eta_discharge must be in (0, 1]")
    if data.c_reconf_yuan_per_mwh < 0:
        raise ValueError("c_reconf_yuan_per_mwh must be >= 0")
    if data.objective_mode not in {"operational", "planning"}:
        raise ValueError("objective_mode must be either 'operational' or 'planning'")

    windows_set = set(data.windows)
    for t in data.hours:
        if t not in data.hour_to_window:
            raise ValueError(f"hour_to_window missing hour {t}")
        if data.hour_to_window[t] not in windows_set:
            raise ValueError(f"hour_to_window[{t}]={data.hour_to_window[t]} not in windows")

    for i in data.sites:
        _check_bounds(f"soc_initial_frac[{i}]", data.soc_initial_frac.get(i, 0.5), 0.0, 1.0)
        _check_bounds(f"soc_min_frac[{i}]", data.soc_min_frac.get(i, 0.0), 0.0, 1.0)
        _check_bounds(f"soc_max_frac[{i}]", data.soc_max_frac.get(i, 1.0), 0.0, 1.0)
        if data.soc_min_frac.get(i, 0.0) > data.soc_max_frac.get(i, 1.0):
            raise ValueError(f"soc_min_frac[{i}] cannot be larger than soc_max_frac[{i}]")
        if data.fixed_capacity_mwh.get(i, 0.0) < 0:
            raise ValueError(f"fixed_capacity_mwh[{i}] must be >= 0")
        if data.charge_c_rate.get(i, 0.0) < 0:
            raise ValueError(f"charge_c_rate[{i}] must be >= 0")
        if data.discharge_c_rate.get(i, 0.0) < 0:
            raise ValueError(f"discharge_c_rate[{i}] must be >= 0")


def build_core_model(data: CoreModelData) -> pyo.ConcreteModel:
    """Build the core linear optimization model."""
    _validate_data(data)

    m = pyo.ConcreteModel("eep_core_model")

    m.I = pyo.Set(initialize=list(data.sites), ordered=True)
    m.T = pyo.Set(initialize=list(data.hours), ordered=True)
    m.W = pyo.Set(initialize=list(data.windows), ordered=True)
    m.WM = pyo.Set(initialize=[w for w in data.windows if w > 0], ordered=True)

    # Core parameters
    m.hour_to_window = pyo.Param(
        m.T,
        initialize=lambda _m, t: int(data.hour_to_window[int(t)]),
        within=pyo.NonNegativeIntegers,
    )

    m.load_demand = pyo.Param(
        m.I, m.T, initialize=_build_param_initializer(data.load_mw), default=0.0
    )
    m.price = pyo.Param(
        m.I, m.T, initialize=_build_param_initializer(data.price_yuan_per_mwh), default=0.0
    )
    m.grid_limit = pyo.Param(
        m.I, m.T, initialize=_build_param_initializer(data.grid_limit_mw), default=0.0
    )
    m.penalty_unserved = pyo.Param(
        m.I,
        m.T,
        initialize=_build_param_initializer(data.unserved_penalty_yuan_per_mwh),
        default=0.0,
    )

    m.fixed_cap = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.fixed_capacity_mwh), default=0.0
    )
    m.soc_initial_frac = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.soc_initial_frac, 0.5), default=0.5
    )
    m.soc_min_frac = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.soc_min_frac, 0.0), default=0.0
    )
    m.soc_max_frac = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.soc_max_frac, 1.0), default=1.0
    )
    m.charge_c_rate = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.charge_c_rate, 0.0), default=0.0
    )
    m.discharge_c_rate = pyo.Param(
        m.I, initialize=_build_site_param_initializer(data.discharge_c_rate, 0.0), default=0.0
    )

    m.active_init = pyo.Param(
        m.I,
        m.W,
        initialize=_build_active_init_initializer(data.active_init_mwh, 0.0),
        default=0.0,
    )
    m.reconf_limit = pyo.Param(
        m.W,
        initialize=_build_window_param_initializer(data.reconf_limit_mwh, 1e9),
        default=1e9,
    )

    m.eta_charge = pyo.Param(initialize=float(data.eta_charge))
    m.eta_discharge = pyo.Param(initialize=float(data.eta_discharge))
    m.dt_hours = pyo.Param(initialize=float(data.dt_hours))
    m.m_total = pyo.Param(initialize=float(data.m_total_mwh))
    m.delay = pyo.Param(initialize=int(data.delay_windows), within=pyo.NonNegativeIntegers)

    m.c_reconf = pyo.Param(initialize=float(data.c_reconf_yuan_per_mwh))
    m.c_storage_fixed = pyo.Param(initialize=float(data.c_storage_fixed_yuan_per_mwh))
    m.c_storage_mobile = pyo.Param(initialize=float(data.c_storage_mobile_yuan_per_mwh))
    m.charge_mode_limit = pyo.Param(
        m.I,
        initialize=lambda _m, i: float(data.charge_c_rate.get(i, 0.0))
        * (float(data.fixed_capacity_mwh.get(i, 0.0)) + float(data.m_total_mwh)),
        default=0.0,
    )
    m.discharge_mode_limit = pyo.Param(
        m.I,
        initialize=lambda _m, i: float(data.discharge_c_rate.get(i, 0.0))
        * (float(data.fixed_capacity_mwh.get(i, 0.0)) + float(data.m_total_mwh)),
        default=0.0,
    )

    # Decision variables
    m.x = pyo.Var(m.I, m.W, domain=pyo.NonNegativeReals)  # deployed mobile capacity
    m.m_active = pyo.Var(m.I, m.W, domain=pyo.NonNegativeReals)  # active mobile capacity

    m.p_ch_fixed = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.p_dis_fixed = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.p_ch_mobile = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.p_dis_mobile = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.is_charging = pyo.Var(m.I, m.T, domain=pyo.UnitInterval)
    m.p_grid = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.unserved = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.soc_fixed = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
    m.soc_mobile = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

    m.reconf_pos = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)
    m.reconf_neg = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)
    m.active_reconf_pos = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)
    m.active_reconf_neg = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)
    m.active_reconf_dir = pyo.Var(m.I, m.WM, domain=pyo.Binary)
    m.energy_shift_pos = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)
    m.energy_shift_neg = pyo.Var(m.I, m.WM, domain=pyo.NonNegativeReals)

    def _energy_cap(_m, i, t):
        w = int(pyo.value(_m.hour_to_window[t]))
        return _m.fixed_cap[i] + _m.m_active[i, w]

    m.energy_cap = pyo.Expression(m.I, m.T, rule=_energy_cap)
    m.mobile_cap = pyo.Expression(
        m.I,
        m.T,
        rule=lambda _m, i, t: _m.m_active[i, int(pyo.value(_m.hour_to_window[t]))],
    )
    m.p_ch = pyo.Expression(m.I, m.T, rule=lambda _m, i, t: _m.p_ch_fixed[i, t] + _m.p_ch_mobile[i, t])
    m.p_dis = pyo.Expression(m.I, m.T, rule=lambda _m, i, t: _m.p_dis_fixed[i, t] + _m.p_dis_mobile[i, t])
    m.soc = pyo.Expression(m.I, m.T, rule=lambda _m, i, t: _m.soc_fixed[i, t] + _m.soc_mobile[i, t])

    # 1) Mobile capacity pool
    def _mobile_pool_rule(_m, w):
        return sum(_m.x[i, w] for i in _m.I) <= _m.m_total

    m.mobile_pool = pyo.Constraint(m.W, rule=_mobile_pool_rule)

    # 2) Delay activation
    def _delay_activation_rule(_m, i, w):
        d = int(pyo.value(_m.delay))
        if w >= d:
            return _m.m_active[i, w] == _m.x[i, w - d]
        return _m.m_active[i, w] == _m.active_init[i, w]

    m.delay_activation = pyo.Constraint(m.I, m.W, rule=_delay_activation_rule)

    # 3) Hard reconfiguration constraint: sum_i |x[i,w] - x[i,w-1]| <= R_w
    def _reconf_balance_rule(_m, i, w):
        return _m.x[i, w] - _m.x[i, w - 1] == _m.reconf_pos[i, w] - _m.reconf_neg[i, w]

    m.reconf_balance = pyo.Constraint(m.I, m.WM, rule=_reconf_balance_rule)

    def _reconf_hard_limit_rule(_m, w):
        return sum(_m.reconf_pos[i, w] + _m.reconf_neg[i, w] for i in _m.I) <= _m.reconf_limit[w]

    m.reconf_hard_limit = pyo.Constraint(m.WM, rule=_reconf_hard_limit_rule)

    def _active_reconf_balance_rule(_m, i, w):
        return (
            _m.m_active[i, w] - _m.m_active[i, w - 1]
            == _m.active_reconf_pos[i, w] - _m.active_reconf_neg[i, w]
        )

    m.active_reconf_balance = pyo.Constraint(m.I, m.WM, rule=_active_reconf_balance_rule)

    def _active_reconf_pos_gate_rule(_m, i, w):
        return _m.active_reconf_pos[i, w] <= _m.m_total * _m.active_reconf_dir[i, w]

    m.active_reconf_pos_gate = pyo.Constraint(m.I, m.WM, rule=_active_reconf_pos_gate_rule)

    def _active_reconf_neg_gate_rule(_m, i, w):
        return _m.active_reconf_neg[i, w] <= _m.m_total * (1 - _m.active_reconf_dir[i, w])

    m.active_reconf_neg_gate = pyo.Constraint(m.I, m.WM, rule=_active_reconf_neg_gate_rule)

    def _energy_shift_conservation_rule(_m, w):
        return sum(_m.energy_shift_pos[i, w] for i in _m.I) == sum(_m.energy_shift_neg[i, w] for i in _m.I)

    m.energy_shift_conservation = pyo.Constraint(m.WM, rule=_energy_shift_conservation_rule)

    def _energy_shift_in_limit_rule(_m, i, w):
        return _m.energy_shift_pos[i, w] <= _m.soc_max_frac[i] * _m.active_reconf_pos[i, w]

    m.energy_shift_in_limit = pyo.Constraint(m.I, m.WM, rule=_energy_shift_in_limit_rule)

    def _energy_shift_out_limit_rule(_m, i, w):
        return _m.energy_shift_neg[i, w] <= _m.soc_max_frac[i] * _m.active_reconf_neg[i, w]

    m.energy_shift_out_limit = pyo.Constraint(m.I, m.WM, rule=_energy_shift_out_limit_rule)

    # 4) Hourly operation
    # 4.1 power balance
    def _power_balance_rule(_m, i, t):
        return (
            _m.p_grid[i, t] + _m.p_dis_fixed[i, t] + _m.p_dis_mobile[i, t] + _m.unserved[i, t]
            == _m.load_demand[i, t] + _m.p_ch_fixed[i, t] + _m.p_ch_mobile[i, t]
        )

    m.power_balance = pyo.Constraint(m.I, m.T, rule=_power_balance_rule)

    # 4.2 fixed/mobile SOC dynamics
    # SOC[t] is modeled as the end-of-step energy state after dispatch at time t.
    # This avoids the terminal-step loophole where the last period could discharge
    # without any SOC consequence.
    def _soc_fixed_dyn_rule(_m, i, t):
        step_delta = _m.dt_hours * (
            _m.eta_charge * _m.p_ch_fixed[i, t] - _m.p_dis_fixed[i, t] / _m.eta_discharge
        )
        if t == 0:
            return _m.soc_fixed[i, t] == _m.soc_initial_frac[i] * _m.fixed_cap[i] + step_delta
        return _m.soc_fixed[i, t] == _m.soc_fixed[i, t - 1] + step_delta

    m.soc_fixed_dyn = pyo.Constraint(m.I, m.T, rule=_soc_fixed_dyn_rule)

    def _soc_mobile_dyn_rule(_m, i, t):
        w = int(pyo.value(_m.hour_to_window[t]))
        step_delta = _m.dt_hours * (
            _m.eta_charge * _m.p_ch_mobile[i, t] - _m.p_dis_mobile[i, t] / _m.eta_discharge
        )
        if t == 0:
            return _m.soc_mobile[i, t] == _m.soc_initial_frac[i] * _m.mobile_cap[i, t] + step_delta

        shift = 0.0
        if int(pyo.value(_m.hour_to_window[t - 1])) != w:
            shift = _m.energy_shift_pos[i, w] - _m.energy_shift_neg[i, w]
        return _m.soc_mobile[i, t] == _m.soc_mobile[i, t - 1] + shift + step_delta

    m.soc_mobile_dyn = pyo.Constraint(m.I, m.T, rule=_soc_mobile_dyn_rule)

    # 4.3 charge/discharge limits
    def _charge_limit_fixed_rule(_m, i, t):
        return _m.p_ch_fixed[i, t] <= _m.charge_c_rate[i] * _m.fixed_cap[i]

    m.charge_limit_fixed = pyo.Constraint(m.I, m.T, rule=_charge_limit_fixed_rule)

    def _charge_limit_mobile_rule(_m, i, t):
        return _m.p_ch_mobile[i, t] <= _m.charge_c_rate[i] * _m.mobile_cap[i, t]

    m.charge_limit_mobile = pyo.Constraint(m.I, m.T, rule=_charge_limit_mobile_rule)

    def _discharge_limit_fixed_rule(_m, i, t):
        return _m.p_dis_fixed[i, t] <= _m.discharge_c_rate[i] * _m.fixed_cap[i]

    m.discharge_limit_fixed = pyo.Constraint(m.I, m.T, rule=_discharge_limit_fixed_rule)

    def _discharge_limit_mobile_rule(_m, i, t):
        return _m.p_dis_mobile[i, t] <= _m.discharge_c_rate[i] * _m.mobile_cap[i, t]

    m.discharge_limit_mobile = pyo.Constraint(m.I, m.T, rule=_discharge_limit_mobile_rule)

    # 4.3b mutually exclusive charge/discharge mode
    def _charge_mode_rule(_m, i, t):
        return (_m.p_ch_fixed[i, t] + _m.p_ch_mobile[i, t]) <= _m.charge_mode_limit[i] * _m.is_charging[i, t]

    m.charge_mode_limit_con = pyo.Constraint(m.I, m.T, rule=_charge_mode_rule)

    def _discharge_mode_rule(_m, i, t):
        return (_m.p_dis_fixed[i, t] + _m.p_dis_mobile[i, t]) <= _m.discharge_mode_limit[i] * (1 - _m.is_charging[i, t])

    m.discharge_mode_limit_con = pyo.Constraint(m.I, m.T, rule=_discharge_mode_rule)

    # 4.4 SOC bounds
    def _soc_min_fixed_rule(_m, i, t):
        return _m.soc_fixed[i, t] >= _m.soc_min_frac[i] * _m.fixed_cap[i]

    m.soc_min_fixed_bound = pyo.Constraint(m.I, m.T, rule=_soc_min_fixed_rule)

    def _soc_max_fixed_rule(_m, i, t):
        return _m.soc_fixed[i, t] <= _m.soc_max_frac[i] * _m.fixed_cap[i]

    m.soc_max_fixed_bound = pyo.Constraint(m.I, m.T, rule=_soc_max_fixed_rule)

    def _soc_min_mobile_rule(_m, i, t):
        return _m.soc_mobile[i, t] >= _m.soc_min_frac[i] * _m.mobile_cap[i, t]

    m.soc_min_mobile_bound = pyo.Constraint(m.I, m.T, rule=_soc_min_mobile_rule)

    def _soc_max_mobile_rule(_m, i, t):
        return _m.soc_mobile[i, t] <= _m.soc_max_frac[i] * _m.mobile_cap[i, t]

    m.soc_max_mobile_bound = pyo.Constraint(m.I, m.T, rule=_soc_max_mobile_rule)

    # 4.5 grid limit
    def _grid_limit_rule(_m, i, t):
        return _m.p_grid[i, t] <= _m.grid_limit[i, t]

    m.grid_limit_con = pyo.Constraint(m.I, m.T, rule=_grid_limit_rule)

    # 4.6 unserved load cap
    def _unserved_cap_rule(_m, i, t):
        return _m.unserved[i, t] <= _m.load_demand[i, t]

    m.unserved_cap = pyo.Constraint(m.I, m.T, rule=_unserved_cap_rule)

    # 5) Objective
    m.C_grid = pyo.Expression(
        expr=sum(m.price[i, t] * m.p_grid[i, t] * m.dt_hours for i in m.I for t in m.T)
    )
    m.C_unserved = pyo.Expression(
        expr=sum(m.penalty_unserved[i, t] * m.unserved[i, t] * m.dt_hours for i in m.I for t in m.T)
    )
    m.C_reconf = pyo.Expression(
        expr=m.c_reconf
        * sum(m.reconf_pos[i, w] + m.reconf_neg[i, w] for i in m.I for w in m.WM)
    )
    m.C_storage = pyo.Expression(
        expr=m.c_storage_fixed * sum(m.fixed_cap[i] for i in m.I) + m.c_storage_mobile * m.m_total
    )
    m.C_total_primary = pyo.Expression(expr=m.C_grid + m.C_unserved + m.C_reconf)
    m.C_total_planning = pyo.Expression(expr=m.C_total_primary + m.C_storage)

    m.total_cost = pyo.Objective(
        expr=m.C_total_primary if data.objective_mode == "operational" else m.C_total_planning,
        sense=pyo.minimize,
    )

    return m


def solve_core_model(model: pyo.ConcreteModel, solver_name: str = "highs"):
    """Solve the model and return Pyomo results object."""
    solver = pyo.SolverFactory(solver_name)
    return solver.solve(model, tee=False)


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _relative_gap(best_incumbent: float | None, best_bound: float | None) -> float | None:
    if best_incumbent is None or best_bound is None:
        return None
    denominator = max(abs(float(best_incumbent)), GAP_EPS)
    raw_gap = max(0.0, float(best_incumbent) - float(best_bound))
    return raw_gap / denominator


def _classify_case_outcome(
    termination_condition: TerminationCondition,
    best_incumbent: float | None,
) -> str:
    if termination_condition == TerminationCondition.optimal:
        return "optimal"
    if termination_condition == TerminationCondition.maxTimeLimit:
        return "feasible_with_gap" if best_incumbent is not None else "timeout_no_incumbent"
    if best_incumbent is not None and termination_condition in {
        TerminationCondition.maxIterations,
        TerminationCondition.objectiveLimit,
        TerminationCondition.interrupted,
    }:
        return "feasible_with_gap"
    if termination_condition == TerminationCondition.infeasible:
        return "infeasible"
    if termination_condition == TerminationCondition.unbounded:
        return "unbounded"
    if termination_condition == TerminationCondition.infeasibleOrUnbounded:
        return "infeasible_or_unbounded"
    if best_incumbent is not None:
        return "feasible_with_gap"
    return "solver_error"


def _prepare_mip_start_from_relaxed_solution(model: pyo.ConcreteModel) -> None:
    for i in model.I:
        for w in model.WM:
            delta = pyo.value(model.m_active[i, w] - model.m_active[i, w - 1], exception=False)
            if delta is None:
                direction = pyo.value(model.active_reconf_dir[i, w], exception=False)
                if direction is None:
                    continue
                model.active_reconf_dir[i, w].value = 1.0 if float(direction) >= 0.5 else 0.0
                continue
            model.active_reconf_dir[i, w].value = 1.0 if float(delta) >= 0.0 else 0.0


def _solve_with_appsi_highs(
    model: pyo.ConcreteModel,
    *,
    time_limit_seconds: float,
    relax_integrality: bool,
    warmstart: bool,
):
    solver = Highs()
    solver.config.time_limit = max(0.0, float(time_limit_seconds))
    solver.config.relax_integrality = relax_integrality
    solver.config.warmstart = warmstart
    solver.config.load_solution = False
    solver.config.stream_solver = os.environ.get("EEP_STREAM_SOLVER", "0") == "1"
    solver.highs_options["threads"] = max(1, int(os.environ.get("EEP_SOLVER_THREADS", "1")))
    result = solver.solve(model)
    if result.best_feasible_objective is not None:
        result.solution_loader.load_vars()
    return result


def solve_core_model_with_diagnostics(
    data: CoreModelData,
    *,
    solver_name: str = "highs",
    time_limit_seconds: float = 5 * 60,
    lp_time_limit_seconds: float = 120.0,
) -> tuple[CoreSolveDiagnostics, Dict[str, Dict] | None]:
    if solver_name.lower() != "highs":
        raise ValueError(f"solver diagnostics are only implemented for highs, got {solver_name}")

    started_at = monotonic()
    model = build_core_model(data)
    remaining_seconds = max(0.0, float(time_limit_seconds) - (monotonic() - started_at))
    if remaining_seconds <= 0.0:
        diagnostics = CoreSolveDiagnostics(
            solver_name=solver_name,
            solver_status="ok",
            termination_condition=TerminationCondition.maxTimeLimit.name,
            case_outcome="timeout_no_incumbent",
            best_incumbent=None,
            best_bound=None,
            gap=None,
            elapsed_seconds=monotonic() - started_at,
            lp_relaxation_objective=None,
        )
        return diagnostics, None

    lp_result = _solve_with_appsi_highs(
        model,
        time_limit_seconds=min(remaining_seconds, float(lp_time_limit_seconds)),
        relax_integrality=True,
        warmstart=False,
    )
    lp_relaxation_objective = _safe_float(lp_result.best_feasible_objective)

    remaining_seconds = max(0.0, float(time_limit_seconds) - (monotonic() - started_at))
    if remaining_seconds <= 0.0:
        diagnostics = CoreSolveDiagnostics(
            solver_name=solver_name,
            solver_status="ok",
            termination_condition=TerminationCondition.maxTimeLimit.name,
            case_outcome="timeout_no_incumbent",
            best_incumbent=None,
            best_bound=lp_relaxation_objective,
            gap=None,
            elapsed_seconds=monotonic() - started_at,
            lp_relaxation_objective=lp_relaxation_objective,
        )
        return diagnostics, None

    if lp_relaxation_objective is not None:
        _prepare_mip_start_from_relaxed_solution(model)

    mip_result = _solve_with_appsi_highs(
        model,
        time_limit_seconds=remaining_seconds,
        relax_integrality=False,
        warmstart=lp_relaxation_objective is not None,
    )
    best_incumbent = _safe_float(mip_result.best_feasible_objective)
    best_bound = _safe_float(mip_result.best_objective_bound)
    termination_condition = mip_result.termination_condition
    gap = 0.0 if termination_condition == TerminationCondition.optimal and best_incumbent is not None else _relative_gap(best_incumbent, best_bound)
    diagnostics = CoreSolveDiagnostics(
        solver_name=solver_name,
        solver_status="ok",
        termination_condition=termination_condition.name,
        case_outcome=_classify_case_outcome(termination_condition, best_incumbent),
        best_incumbent=best_incumbent,
        best_bound=best_bound,
        gap=gap,
        elapsed_seconds=monotonic() - started_at,
        lp_relaxation_objective=lp_relaxation_objective,
    )
    solution = extract_solution(model) if best_incumbent is not None else None
    return diagnostics, solution


def extract_solution(model: pyo.ConcreteModel) -> Dict[str, Dict]:
    """Extract key variable values and objective components."""
    x = {(i, w): pyo.value(model.x[i, w]) for i in model.I for w in model.W}
    m_active = {(i, w): pyo.value(model.m_active[i, w]) for i in model.I for w in model.W}
    soc_mobile = {(i, t): pyo.value(model.soc_mobile[i, t]) for i in model.I for t in model.T}
    p_grid = {(i, t): pyo.value(model.p_grid[i, t]) for i in model.I for t in model.T}
    p_ch = {(i, t): pyo.value(model.p_ch[i, t]) for i in model.I for t in model.T}
    p_dis = {(i, t): pyo.value(model.p_dis[i, t]) for i in model.I for t in model.T}
    unserved = {(i, t): pyo.value(model.unserved[i, t]) for i in model.I for t in model.T}

    return {
        "x": x,
        "m_active": m_active,
        "soc_mobile": soc_mobile,
        "p_grid": p_grid,
        "p_ch": p_ch,
        "p_dis": p_dis,
        "unserved": unserved,
        "costs": {
            "C_grid": pyo.value(model.C_grid),
            "C_unserved": pyo.value(model.C_unserved),
            "C_reconf": pyo.value(model.C_reconf),
            "C_storage": pyo.value(model.C_storage),
            "C_total_primary": pyo.value(model.C_total_primary),
            "C_total_planning": pyo.value(model.C_total_planning),
            "C_total": pyo.value(model.total_cost),
            "C_total_with_storage": pyo.value(model.C_total_planning),
        },
    }
