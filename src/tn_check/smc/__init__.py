"""
Statistical model checking (SMC) comparison module.

Implements importance-sampling SPRT for CSL properties and compares
with TN-Check's certified tensor-network approach.
"""

from tn_check.smc.statistical_checker import (
    SMCResult,
    ImportanceSamplingConfig,
    SMCComparison,
    run_smc_comparison,
    importance_sampling_sprt,
    gillespie_simulation,
)

__all__ = [
    "SMCResult",
    "ImportanceSamplingConfig",
    "SMCComparison",
    "run_smc_comparison",
    "importance_sampling_sprt",
    "gillespie_simulation",
]
