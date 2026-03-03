"""
Stiffness-aware adaptive TDVP integration.

Detects stiffness in CME generators and adapts the TDVP integration
accordingly with adaptive step-size control based on projection quality.
"""

from tn_check.stiffness.adaptive_tdvp import (
    StiffnessAnalysis,
    StiffnessAwareTDVP,
    ProjectionMonitor,
    StepSizeController,
    compute_stiffness_ratio,
    detect_fast_transients,
    analyze_generator_spectrum,
)

__all__ = [
    "StiffnessAnalysis",
    "StiffnessAwareTDVP",
    "ProjectionMonitor",
    "StepSizeController",
    "compute_stiffness_ratio",
    "detect_fast_transients",
    "analyze_generator_spectrum",
]
