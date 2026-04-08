"""Generic battery SOC estimator for simulation scaffolding.

Power sign convention:
- positive power: charge request (grid -> battery)
- negative power: discharge request (battery -> load/grid)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class BatterySpec:
    """Battery and converter limits used in SOC updates."""

    energy_capacity_mwh: float
    charge_power_max_mw: float
    discharge_power_max_mw: float
    eta_charge: float
    eta_discharge: float
    soc_min: float
    soc_max: float

    def validate(self) -> None:
        if self.energy_capacity_mwh <= 0:
            raise ValueError("energy_capacity_mwh must be > 0")
        if self.charge_power_max_mw < 0:
            raise ValueError("charge_power_max_mw must be >= 0")
        if self.discharge_power_max_mw < 0:
            raise ValueError("discharge_power_max_mw must be >= 0")
        if not (0 < self.eta_charge <= 1):
            raise ValueError("eta_charge must be in (0, 1]")
        if not (0 < self.eta_discharge <= 1):
            raise ValueError("eta_discharge must be in (0, 1]")
        if not (0 <= self.soc_min < self.soc_max <= 1):
            raise ValueError("SOC bounds must satisfy 0 <= soc_min < soc_max <= 1")


@dataclass(frozen=True)
class SocStepResult:
    """Result of one SOC update step."""

    soc_prev: float
    soc_next: float
    requested_power_mw: float
    applied_power_mw: float
    curtailed_power_mw: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def soc_step(
    soc_prev: float,
    requested_power_mw: float,
    dt_hours: float,
    spec: BatterySpec,
) -> SocStepResult:
    """Update SOC for one step with limits and efficiencies.

    Args:
        soc_prev: previous SOC in [0, 1].
        requested_power_mw: positive for charging, negative for discharging.
        dt_hours: simulation step length in hours.
        spec: battery specification and constraints.
    """
    spec.validate()
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    if not (0 <= soc_prev <= 1):
        raise ValueError("soc_prev must be in [0, 1]")

    soc = _clamp(soc_prev, spec.soc_min, spec.soc_max)
    applied_power = 0.0

    if requested_power_mw >= 0:
        p_charge = min(requested_power_mw, spec.charge_power_max_mw)
        requested_energy_to_battery = p_charge * dt_hours * spec.eta_charge
        max_energy_to_battery = (spec.soc_max - soc) * spec.energy_capacity_mwh
        energy_to_battery = min(requested_energy_to_battery, max_energy_to_battery)
        if dt_hours > 0 and spec.eta_charge > 0:
            applied_power = energy_to_battery / (dt_hours * spec.eta_charge)
        soc_next = soc + energy_to_battery / spec.energy_capacity_mwh
    else:
        p_discharge = min(-requested_power_mw, spec.discharge_power_max_mw)
        requested_energy_from_battery = p_discharge * dt_hours / spec.eta_discharge
        max_energy_from_battery = (soc - spec.soc_min) * spec.energy_capacity_mwh
        energy_from_battery = min(requested_energy_from_battery, max_energy_from_battery)
        if dt_hours > 0:
            applied_power = -energy_from_battery * spec.eta_discharge / dt_hours
        soc_next = soc - energy_from_battery / spec.energy_capacity_mwh

    soc_next = _clamp(soc_next, spec.soc_min, spec.soc_max)
    curtailed_power = requested_power_mw - applied_power

    return SocStepResult(
        soc_prev=soc_prev,
        soc_next=soc_next,
        requested_power_mw=requested_power_mw,
        applied_power_mw=applied_power,
        curtailed_power_mw=curtailed_power,
    )


def simulate_soc_series(
    soc_initial: float,
    requested_power_series_mw: Iterable[float],
    dt_hours: float,
    spec: BatterySpec,
) -> List[SocStepResult]:
    """Run SOC simulation over a power request series."""
    if not (0 <= soc_initial <= 1):
        raise ValueError("soc_initial must be in [0, 1]")

    results: List[SocStepResult] = []
    soc = soc_initial
    for requested_power in requested_power_series_mw:
        step = soc_step(
            soc_prev=soc,
            requested_power_mw=float(requested_power),
            dt_hours=dt_hours,
            spec=spec,
        )
        results.append(step)
        soc = step.soc_next
    return results
