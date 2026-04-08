"""EEP simulation package."""

from .core_model import CoreModelData, build_core_model, extract_solution, solve_core_model
from .experiments import prepare_capacity_fair_mobile_data, prepare_same_budget_data, run_all_experiments
from .g2_loader import G2LoadDiagnostics, G2LoaderOptions, load_g2_core_model_data
from .soc import BatterySpec, SocStepResult, simulate_soc_series, soc_step

__all__ = [
    "CoreModelData",
    "build_core_model",
    "solve_core_model",
    "extract_solution",
    "G2LoaderOptions",
    "G2LoadDiagnostics",
    "load_g2_core_model_data",
    "run_all_experiments",
    "prepare_capacity_fair_mobile_data",
    "prepare_same_budget_data",
    "BatterySpec",
    "SocStepResult",
    "soc_step",
    "simulate_soc_series",
]
