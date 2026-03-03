"""
FSP-TT coupling analysis.

Analyzes how FSP truncation level M and TT bond dimension χ interact,
provides joint M-χ optimization, and tracks FSP non-conservativity.
"""

from tn_check.coupling.fsp_tt_coupling import (
    FSPTTCouplingAnalysis,
    CouplingResult,
    JointOptimizationResult,
    NonConservativityTracker,
    analyze_fsp_tt_coupling,
    joint_m_chi_optimization,
    track_nonconservativity,
    fsp_nonconservativity_bound,
    coupling_correlation_study,
)

__all__ = [
    "FSPTTCouplingAnalysis",
    "CouplingResult",
    "JointOptimizationResult",
    "NonConservativityTracker",
    "analyze_fsp_tt_coupling",
    "joint_m_chi_optimization",
    "track_nonconservativity",
    "fsp_nonconservativity_bound",
    "coupling_correlation_study",
]
