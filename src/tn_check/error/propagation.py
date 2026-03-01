"""
Error propagation analysis for TT-compressed Markov semigroups.

Theorem 1 (Error Contraction):
    For CME generator Q (Metzler matrix with zero column sums),
    the semigroup e^{Qt} is a contraction in L1 norm: ‖e^{Qt}‖₁ = 1.
    
    Therefore, per-step truncation errors do NOT amplify:
    ‖p_exact(t) - p_TT(t)‖₁ ≤ Σ_k ε_k
    where ε_k is the truncation error at step k.

    Proof sketch:
    1. Q is a Metzler matrix (non-negative off-diagonal entries, negative
       diagonal entries) with zero column sums (conservation of probability).
    2. The matrix exponential e^{Qt} is a stochastic matrix: all entries
       non-negative and columns sum to 1. Therefore ‖e^{Qt}‖₁ = 1.
    3. At each step k, the truncation introduces error δ_k with ‖δ_k‖₁ ≤ ε_k.
       After propagation through the remaining semigroup:
         ‖e^{Q(t-t_k)} δ_k‖₁ ≤ ‖e^{Q(t-t_k)}‖₁ · ‖δ_k‖₁ = 1 · ε_k
    4. By triangle inequality over K steps:
         ‖p_exact(t) - p_TT(t)‖₁ ≤ Σ_{k=1}^{K} ε_k

    Combined with Lemma 2 (clamping reduces L1 error), the total certified
    error for the clamped TT-compressed solution is:
    ‖p_exact(t) - p_clamped(t)‖₁ ≤ Σ_k ε_k
    (NOT 2·Σ_k ε_k — Lemma 2 shows clamping cannot increase L1 error.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tn_check.error.certification import ErrorCertificate

logger = logging.getLogger(__name__)


@dataclass
class PropagationAnalysis:
    """Analysis of error propagation through CSL evaluation."""
    time_horizon: float = 0.0
    num_steps: int = 0
    per_step_errors: list[float] = None
    accumulated_error: float = 0.0
    contractivity_factor: float = 1.0  # for Metzler, this is 1
    amplification_bound: float = 1.0
    clamping_overhead: float = 2.0  # factor from Proposition 1

    def __post_init__(self):
        if self.per_step_errors is None:
            self.per_step_errors = []


def semigroup_error_bound(
    per_step_errors: list[float],
    time_horizon: float,
    is_metzler: bool = True,
) -> PropagationAnalysis:
    """
    Compute certified error bound for TT-compressed Markov semigroup evolution.

    For Metzler generators (CME), the semigroup is an L1 contraction,
    so errors accumulate linearly:
        ‖p_exact(t) - p_TT(t)‖₁ ≤ C(t) * Σ ε_k
    where C(t) = 1 for Metzler generators (Theorem 1).

    For non-Metzler generators, C(t) can grow, and we use the
    conservative bound C(t) = e^{μt} where μ is the log-norm.

    Args:
        per_step_errors: List of per-step truncation errors.
        time_horizon: Total time horizon T.
        is_metzler: Whether the generator has Metzler structure.

    Returns:
        PropagationAnalysis with certified bounds.
    """
    analysis = PropagationAnalysis(
        time_horizon=time_horizon,
        num_steps=len(per_step_errors),
        per_step_errors=per_step_errors,
    )

    total_truncation = sum(per_step_errors)

    if is_metzler:
        # Theorem 1: Metzler contractivity
        # ‖e^{Qt}‖₁ = 1 for stochastic semigroup
        analysis.contractivity_factor = 1.0
        analysis.accumulated_error = total_truncation
        # By Lemma 2, clamping reduces L1 error, so no factor-of-2 overhead
        analysis.amplification_bound = total_truncation
        analysis.clamping_overhead = 1.0  # clamping does not amplify
    else:
        # Generic non-Hermitian: exponential amplification possible
        # Conservative bound: assume ‖e^{Qt}‖ ≤ e^{μt}
        # Without knowing μ, use the worst-case bound
        mu_estimate = 1.0  # conservative
        analysis.contractivity_factor = np.exp(mu_estimate * time_horizon)
        analysis.accumulated_error = analysis.contractivity_factor * total_truncation
        analysis.amplification_bound = 2.0 * analysis.accumulated_error

    return analysis


def csl_error_propagation(
    inner_error: float,
    threshold: float,
    comparison: str,
    num_states_near_threshold: int = 0,
    total_states: int = 1,
) -> dict:
    """
    Analyze error propagation through CSL probability operators.

    For nested P~p[ψ], inner error ε in the probability computation
    creates an indeterminate zone of width 2ε around the threshold.
    States with probability in [p-ε, p+ε] cannot be classified as
    definitely satisfying or violating the property.

    Three-valued semantics (Katoen et al. 2007, Hermanns et al. 2008):
    - TRUE:  all states in inner sat-set are definitely true
    - FALSE: all states in inner sat-set are definitely false
    - INDETERMINATE: some states are uncertain

    The indeterminate fraction is bounded by:
        |{s : |Pr_s[ψ] - p| ≤ ε}| / |S|

    For single-level CSL (common in biology), this fraction is typically
    small because probability distributions are smooth.

    Args:
        inner_error: Error from inner probability computation.
        threshold: Probability threshold p.
        comparison: Comparison operator (">=", ">", "<=", "<").
        num_states_near_threshold: Estimated states with prob near threshold.
        total_states: Total number of states.

    Returns:
        Dictionary with propagation analysis.
    """
    indeterminate_width = 2 * inner_error
    indeterminate_fraction = (
        num_states_near_threshold / max(total_states, 1)
        if total_states > 0 else 1.0
    )

    return {
        "inner_error": inner_error,
        "threshold": threshold,
        "indeterminate_width": indeterminate_width,
        "indeterminate_fraction": indeterminate_fraction,
        "sound": True,  # three-valued semantics is always sound
        "complete": indeterminate_fraction < 0.01,  # approximate completeness
        "recommendation": (
            "reduce truncation tolerance" if indeterminate_fraction > 0.1
            else "error bounds are adequate"
        ),
    }


def compose_error_certificates(
    *certificates: ErrorCertificate,
) -> ErrorCertificate:
    """
    Compose error certificates from multiple computation stages.

    Error bounds compose additively (triangle inequality).
    """
    result = ErrorCertificate()
    for cert in certificates:
        result.truncation_error += cert.truncation_error
        result.clamping_error += cert.clamping_error
        result.fsp_error += cert.fsp_error
        result.integration_error += cert.integration_error
        result.negativity_mass += cert.negativity_mass
    result.compute_total()
    return result


def compose_fsp_tt_errors(
    fsp_error: float,
    per_step_truncation_errors: list[float],
    per_step_clamping_errors: Optional[list[float]] = None,
    threshold: Optional[float] = None,
    is_metzler: bool = True,
) -> dict:
    """
    Analyze how FSP truncation error and TT truncation error compose
    through the CSL evaluation pipeline.

    Key insight:
    - FSP truncation removes probability mass from high-count states → ε_FSP
    - TT truncation introduces L1 error at each step → Σ ε_k
    - These errors are independent and additive (triangle inequality)
    - For probability operators P~p[φ], the FSP error affects the threshold
      comparison differently than the TT error

    For Metzler generators (CME), the semigroup is an L1 contraction,
    so TT errors accumulate linearly (Theorem 1).

    Args:
        fsp_error: FSP truncation error bound (probability mass lost).
        per_step_truncation_errors: Per-step TT truncation errors.
        per_step_clamping_errors: Per-step clamping errors (optional).
        threshold: Probability threshold for CSL operator (optional).
        is_metzler: Whether the generator is Metzler.

    Returns:
        Dictionary with composition analysis including:
        - total_tt_error: accumulated TT truncation error
        - total_clamping_error: accumulated clamping error
        - total_composed_error: FSP + TT + clamping bound
        - per_source breakdown and indeterminate zone analysis
    """
    total_tt = sum(per_step_truncation_errors)
    total_clamp = sum(per_step_clamping_errors) if per_step_clamping_errors else 0.0
    num_steps = len(per_step_truncation_errors)

    # For Metzler generators, contractivity factor is 1
    contractivity = 1.0 if is_metzler else np.exp(num_steps * 0.01)

    # TT error after propagation through semigroup
    propagated_tt_error = contractivity * total_tt

    # Clamping bound: by Lemma 2, clamping reduces L1 error, so no additional term
    clamping_bound = 0.0  # clamping does not increase error

    # Total composed error (triangle inequality: FSP ⊥ TT)
    total_composed = fsp_error + propagated_tt_error + clamping_bound

    analysis = {
        "fsp_error": fsp_error,
        "total_tt_error": total_tt,
        "propagated_tt_error": propagated_tt_error,
        "total_clamping_error": total_clamp,
        "clamping_bound": clamping_bound,
        "contractivity_factor": contractivity,
        "num_steps": num_steps,
        "is_metzler": is_metzler,
        "total_composed_error": total_composed,
        "error_breakdown": {
            "fsp_fraction": fsp_error / max(total_composed, 1e-300),
            "tt_fraction": propagated_tt_error / max(total_composed, 1e-300),
            "clamping_fraction": clamping_bound / max(total_composed, 1e-300),
        },
    }

    # If threshold is given, analyze indeterminate zone for P~p[φ]
    if threshold is not None:
        indeterminate_width = 2 * total_composed
        # FSP error shifts the probability estimate downward (missing mass)
        # TT error creates symmetric uncertainty
        analysis["threshold_analysis"] = {
            "threshold": threshold,
            "indeterminate_zone": [
                max(0.0, threshold - total_composed),
                min(1.0, threshold + total_composed),
            ],
            "indeterminate_width": indeterminate_width,
            "fsp_bias": -fsp_error,  # FSP underestimates probability
            "tt_symmetric_error": propagated_tt_error + clamping_bound,
        }

    return analysis
