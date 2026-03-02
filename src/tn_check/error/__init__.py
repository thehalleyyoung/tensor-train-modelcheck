"""
Error certification and tracking for TT-compressed CSL model checking.

Implements:
- Non-negativity-preserving TT rounding (alternating projections)
- Clamping error bounds (Proposition 1)
- Error propagation through Markov semigroup (Theorem 1)
- Per-bond truncation error tracking
- Richardson extrapolation for convergence estimation
- Cumulative clamping error analysis across time steps
"""

from tn_check.error.certification import (
    ErrorCertificate,
    ErrorTracker,
    ClampingProof,
    ClampingProofIteration,
    ClampingCertificate,
    nonneg_preserving_round,
    certified_nonneg_round,
    clamping_error_bound,
    tight_clamping_bound,
    tight_clamping_bound_certified,
    verify_clamping_proposition,
    verify_clamping_reduces_error,
)
from tn_check.error.propagation import (
    semigroup_error_bound,
    csl_error_propagation,
    compose_fsp_tt_errors,
)
from tn_check.error.cumulative import (
    CumulativeErrorTracker,
    CumulativeErrorAnalysis,
    StepErrorRecord,
    run_cumulative_error_experiment,
)

__all__ = [
    "ErrorCertificate",
    "ErrorTracker",
    "ClampingProof",
    "ClampingProofIteration",
    "ClampingCertificate",
    "nonneg_preserving_round",
    "certified_nonneg_round",
    "clamping_error_bound",
    "tight_clamping_bound",
    "tight_clamping_bound_certified",
    "verify_clamping_proposition",
    "verify_clamping_reduces_error",
    "semigroup_error_bound",
    "csl_error_propagation",
    "compose_fsp_tt_errors",
    "CumulativeErrorTracker",
    "CumulativeErrorAnalysis",
    "StepErrorRecord",
    "run_cumulative_error_experiment",
]
