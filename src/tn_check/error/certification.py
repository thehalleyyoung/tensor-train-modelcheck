"""
Error certification for TT-compressed probability vectors.

Addresses the critical clamping soundness gap:
- TT-SVD truncation produces negative entries
- Post-truncation clamping is nonlinear, breaks SVD optimality
- We provide: (a) tight clamping error bound, (b) alternating-projections
  non-negativity-preserving rounding as an alternative

Lemma 1 (Negativity Bound):
    Let p_exact ≥ 0 be a probability vector and p_SVD its SVD truncation
    with ‖p_exact - p_SVD‖₁ ≤ ε_trunc. Then:
        Σ_{i: p_SVD[i] < 0} |p_SVD[i]| ≤ ε_trunc

    Proof (step-by-step):
    For each index i where p_SVD[i] < 0, since p_exact[i] ≥ 0:
        |p_exact[i] - p_SVD[i]| = p_exact[i] - p_SVD[i]
                                 = p_exact[i] + |p_SVD[i]|
                                 ≥ |p_SVD[i]|
    Summing over the set N = {i : p_SVD[i] < 0}:
        Σ_{i∈N} |p_SVD[i]| ≤ Σ_{i∈N} |p_exact[i] - p_SVD[i]|
                             ≤ Σ_i |p_exact[i] - p_SVD[i]|
                             = ‖p_exact - p_SVD‖₁
                             ≤ ε_trunc                          □

Lemma 2 (Clamping Reduces L1 Error):
    Let p_clamped = max(p_SVD, 0). Then:
        ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁ ≤ ε_trunc

    Proof (index-wise comparison):
    For each index i, we show |p_exact[i] - p_clamped[i]| ≤ |p_exact[i] - p_SVD[i]|:

    Case 1: p_SVD[i] ≥ 0.
        Clamping leaves it unchanged: p_clamped[i] = p_SVD[i].
        So |p_exact[i] - p_clamped[i]| = |p_exact[i] - p_SVD[i]|.

    Case 2: p_SVD[i] < 0.
        Clamping sets p_clamped[i] = 0. Since p_exact[i] ≥ 0:
        |p_exact[i] - p_clamped[i]| = |p_exact[i] - 0| = p_exact[i]
        |p_exact[i] - p_SVD[i]|     = p_exact[i] + |p_SVD[i]| ≥ p_exact[i]

    In both cases |p_exact[i] - p_clamped[i]| ≤ |p_exact[i] - p_SVD[i]|.
    Summing over all i:
        ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁ ≤ ε_trunc       □

Proposition 1 (Clamping Error Bound — Strengthened):
    Let p_SVD be the SVD-truncated MPS and p_clamped = max(p_SVD, 0) the
    element-wise clamped vector. Then:
        ‖p_clamped - p_SVD‖₁ ≤ ε_trunc         (by Lemma 1)
        ‖p_exact - p_clamped‖₁ ≤ ε_trunc        (by Lemma 2, NOT 2ε_trunc)

    The previous 2ε_trunc bound used an unnecessary triangle inequality.
    Lemma 2 proves the stronger result directly: clamping moves every entry
    closer to p_exact, so the L1 error cannot increase.

This resolves the clamping gap identified in the critique: the certified
error bound for clamped vectors is ε_trunc, not 2ε_trunc or unbounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm
from tn_check.tensor.operations import (
    mps_compress, mps_inner_product, mps_to_dense, mps_total_probability,
    mps_clamp_nonnegative, mps_normalize_probability,
    mps_hadamard_product, mps_scalar_multiply, mps_addition,
    mps_probability_at_index,
)

logger = logging.getLogger(__name__)


@dataclass
class ClampingProofIteration:
    """Data for a single iteration of the alternating-projection rounding."""
    iteration: int
    truncation_error: float
    clamping_error: float
    negativity_mass: float


@dataclass
class ClampingCertificate:
    """
    Proof object for Proposition 1 (strengthened clamping bound).

    By Lemma 2, clamping reduces L1 error, so:
        ‖p_clamped - p_exact‖₁ ≤ ε_trunc
    The bound does not depend on measured negativity N_neg (though N_neg ≤ ε_trunc
    by Lemma 1). We retain N_neg for diagnostics and tighter empirical bounds.
    """
    negativity_mass: float = 0.0
    truncation_error: float = 0.0
    bound_type: str = 'tight'  # 'tight' or 'worst_case'
    bound_value: float = 0.0

    def compute_bound(self) -> float:
        """Compute the certified clamping bound.

        By Lemma 2, clamping reduces L1 error, so the bound is simply ε_trunc.
        We no longer need 2·ε_trunc or ε_trunc + N_neg as upper bounds.
        """
        self.bound_value = self.truncation_error
        self.bound_type = 'tight'
        return self.bound_value

    def verify(self) -> bool:
        """Verify that N_neg ≤ ε_trunc (prerequisite of Proposition 1)."""
        return self.negativity_mass <= self.truncation_error + 1e-12

    def to_dict(self) -> dict:
        self.compute_bound()
        return {
            "negativity_mass": self.negativity_mass,
            "truncation_error": self.truncation_error,
            "bound_type": self.bound_type,
            "bound_value": self.bound_value,
            "lemma2_bound": self.truncation_error,
        }


@dataclass
class ClampingProof:
    """
    Proof object for non-negativity-preserving rounding (Proposition 1).

    Stores per-iteration data so the clamping bound can be independently
    verified without re-running the rounding procedure.
    """
    iterations: list[ClampingProofIteration] = field(default_factory=list)
    converged: bool = False
    final_negativity: float = 0.0
    total_truncation_error: float = 0.0
    total_clamping_error: float = 0.0

    def record(
        self,
        iteration: int,
        truncation_error: float,
        clamping_error: float,
        negativity_mass: float,
    ) -> None:
        """Record one iteration of the rounding procedure."""
        self.iterations.append(ClampingProofIteration(
            iteration=iteration,
            truncation_error=truncation_error,
            clamping_error=clamping_error,
            negativity_mass=negativity_mass,
        ))
        self.total_truncation_error += truncation_error
        self.total_clamping_error += clamping_error

    def verify(self) -> bool:
        """
        Verify that the clamping bound holds for all recorded iterations.

        At each iteration, clamping_error <= 2 * truncation_error
        (Proposition 1 applied per step).
        """
        for it in self.iterations:
            if it.clamping_error > 2.0 * it.truncation_error + 1e-14:
                return False
        return True

    def negativity_decreasing(self) -> bool:
        """Check that negativity mass is non-increasing across iterations."""
        for i in range(1, len(self.iterations)):
            if self.iterations[i].negativity_mass > self.iterations[i - 1].negativity_mass + 1e-14:
                return False
        return True


@dataclass
class ErrorCertificate:
    """
    Certified error bounds for a TT-compressed probability vector.

    All error bounds are in L1 norm (total variation distance = L1/2).
    """
    truncation_error: float = 0.0
    clamping_error: float = 0.0
    fsp_error: float = 0.0
    integration_error: float = 0.0
    negativity_mass: float = 0.0
    normalization_deviation: float = 0.0
    total_certified_error: float = 0.0
    per_step_truncation_errors: list = field(default_factory=list)
    per_step_clamping_errors: list = field(default_factory=list)
    clamping_certificate: Optional[ClampingCertificate] = None
    fsp_composition_analysis: Optional[dict] = None
    bound_type: str = 'worst_case'

    def compute_total(self) -> float:
        """Compute total certified error bound.

        By Lemma 2, clamping reduces L1 error, so it does NOT add to
        the truncation error. The total bound is ε_trunc + other errors,
        not 2ε_trunc + other errors as previously stated.
        """
        # Clamping does not add error (Lemma 2), so no separate clamping term
        self.bound_type = 'tight'
        self.total_certified_error = (
            self.truncation_error
            + self.fsp_error
            + self.integration_error
        )
        return self.total_certified_error

    def is_within_budget(self, budget: float) -> bool:
        """Check if total error is within the specified budget."""
        return self.compute_total() <= budget

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "truncation_error": self.truncation_error,
            "clamping_error": self.clamping_error,
            "fsp_error": self.fsp_error,
            "integration_error": self.integration_error,
            "negativity_mass": self.negativity_mass,
            "normalization_deviation": self.normalization_deviation,
            "total_certified_error": self.compute_total(),
            "bound_type": self.bound_type,
        }
        if self.clamping_certificate is not None:
            d["clamping_certificate"] = self.clamping_certificate.to_dict()
        if self.fsp_composition_analysis is not None:
            d["fsp_composition_analysis"] = self.fsp_composition_analysis
        return d


@dataclass
class ErrorTracker:
    """
    Track errors through a computation pipeline.

    Records per-step truncation errors, clamping events,
    and provides Richardson extrapolation for convergence estimation.
    """
    step_errors: list[float] = field(default_factory=list)
    clamping_events: list[tuple[int, float]] = field(default_factory=list)
    bond_dim_history: list[list[int]] = field(default_factory=list)
    negativity_history: list[float] = field(default_factory=list)

    def record_step(
        self,
        step: int,
        trunc_error: float,
        bond_dims: list[int],
        negativity: float = 0.0,
    ) -> None:
        """Record error data for one integration step."""
        self.step_errors.append(trunc_error)
        self.bond_dim_history.append(bond_dims)
        self.negativity_history.append(negativity)

    def record_clamping(self, step: int, clamp_error: float) -> None:
        """Record a clamping event."""
        self.clamping_events.append((step, clamp_error))

    def accumulated_truncation_error(self) -> float:
        """Total accumulated truncation error (Theorem 1: linear, not exponential)."""
        return sum(self.step_errors)

    def accumulated_clamping_error(self) -> float:
        """Total clamping error."""
        return sum(e for _, e in self.clamping_events)

    def certify(self, fsp_error: float = 0.0) -> ErrorCertificate:
        """Generate a certified error bound from tracked data."""
        cert = ErrorCertificate(
            truncation_error=self.accumulated_truncation_error(),
            clamping_error=self.accumulated_clamping_error(),
            fsp_error=fsp_error,
            negativity_mass=sum(self.negativity_history),
        )
        cert.compute_total()
        return cert

    def richardson_extrapolation(self) -> Optional[float]:
        """
        Estimate converged value using Richardson extrapolation.

        Uses the last three error values to estimate the limiting error.
        """
        if len(self.step_errors) < 3:
            return None
        e1, e2, e3 = self.step_errors[-3:]
        if abs(e1 - 2*e2 + e3) < 1e-300:
            return e3
        return float(e3 - (e3 - e2)**2 / (e1 - 2*e2 + e3))

    def convergence_check(self, chi_factor: int = 2) -> dict:
        """
        Check convergence by comparing bond dimensions at χ vs 2χ.

        Returns diagnostics about whether the current bond dimension
        is sufficient.
        """
        if len(self.bond_dim_history) < 2:
            return {"sufficient": True, "evidence": "insufficient_data"}

        current_dims = self.bond_dim_history[-1]
        max_chi = max(current_dims) if current_dims else 1

        return {
            "max_bond_dim": max_chi,
            "accumulated_error": self.accumulated_truncation_error(),
            "num_clamping_events": len(self.clamping_events),
            "sufficient": self.accumulated_truncation_error() < 0.01,
        }


def nonneg_preserving_round(
    mps: MPS,
    max_bond_dim: int,
    tolerance: float = 1e-10,
    max_iterations: int = 20,
) -> tuple[MPS, float, ErrorCertificate]:
    """
    Non-negativity-preserving TT rounding via alternating projections.

    This addresses the clamping soundness gap by avoiding negative entries
    entirely, rather than clamping them post-hoc.

    Algorithm (inspired by Uschmajew 2015):
    1. SVD-compress to target bond dimension (may produce negatives)
    2. Clamp negatives to zero
    3. Re-compress the clamped result (may produce new negatives)
    4. Repeat until negatives are below tolerance or max iterations reached

    Convergence criterion: negativity must decrease across iterations.
    If negativity increases, abort and fall back to simple clamping.

    Proposition 1 guarantees that at each step, the L1 error from clamping
    is bounded by the truncation error, so the total error after K iterations
    is at most (2K+1) * ε_trunc. In practice, convergence is fast (K ≤ 5).

    Args:
        mps: Input MPS (probability vector, may have negatives from truncation).
        max_bond_dim: Target bond dimension.
        tolerance: Convergence tolerance for negativity.
        max_iterations: Maximum alternating projection iterations.

    Returns:
        Tuple of (non-negative MPS, total error, ErrorCertificate).
        The ErrorCertificate has a `clamping_certificate` attribute and
        a `_clamping_proof` attribute (ClampingProof) for backward compat.
    """
    cert = ErrorCertificate()
    proof = ClampingProof()
    total_error = 0.0
    current = mps.copy()
    prev_negativity = float('inf')

    for iteration in range(max_iterations):
        # Step 1: SVD compress
        compressed, trunc_err = mps_compress(
            current, max_bond_dim=max_bond_dim, tolerance=tolerance,
        )
        total_error += trunc_err
        cert.truncation_error += trunc_err
        cert.per_step_truncation_errors.append(trunc_err)

        # Step 2: Check negativity and clamp
        clamped, clamp_err = mps_clamp_nonnegative(
            compressed, max_bond_dim=max_bond_dim * 2, tolerance=tolerance,
        )
        cert.clamping_error += clamp_err
        cert.per_step_clamping_errors.append(clamp_err)

        # Record in proof
        proof.record(
            iteration=iteration,
            truncation_error=trunc_err,
            clamping_error=clamp_err,
            negativity_mass=clamp_err,
        )

        if clamp_err < tolerance:
            # Converged: negligible negative mass
            proof.converged = True
            current = clamped
            break

        # Convergence criterion: negativity must decrease
        if clamp_err > prev_negativity + tolerance:
            # Negativity increased — fall back to simple clamping
            logger.debug(
                f"Nonneg round: negativity increased at iteration {iteration} "
                f"({prev_negativity:.2e} -> {clamp_err:.2e}), falling back"
            )
            current = clamped
            break

        prev_negativity = clamp_err
        total_error += clamp_err
        current = clamped

        logger.debug(
            f"Nonneg round iteration {iteration}: "
            f"trunc_err={trunc_err:.2e}, clamp_err={clamp_err:.2e}"
        )

    # Final normalization
    total_prob = mps_total_probability(current)
    cert.normalization_deviation = abs(total_prob - 1.0)
    if abs(total_prob) > 1e-300 and abs(total_prob - 1.0) > tolerance:
        current = mps_normalize_probability(current)

    proof.final_negativity = proof.iterations[-1].negativity_mass if proof.iterations else 0.0

    # Build ClampingCertificate from proof
    clamp_cert = ClampingCertificate(
        negativity_mass=proof.final_negativity,
        truncation_error=cert.truncation_error,
    )
    clamp_cert.compute_bound()
    cert.clamping_certificate = clamp_cert
    cert.negativity_mass = sum(it.negativity_mass for it in proof.iterations)

    cert.compute_total()
    # Attach proof to cert for downstream inspection
    cert._clamping_proof = proof
    return current, total_error, cert


def certified_nonneg_round(
    mps: MPS,
    max_bond_dim: int,
    tolerance: float = 1e-10,
    max_iterations: int = 20,
) -> tuple[MPS, float, ErrorCertificate, ClampingCertificate]:
    """
    Non-negativity-preserving rounding that produces a ClampingCertificate.

    Identical to nonneg_preserving_round but returns the ClampingCertificate
    as a separate object with the tight error bound based on measured negativity.

    Returns:
        Tuple of (non-negative MPS, total error, ErrorCertificate, ClampingCertificate).
    """
    result_mps, total_error, cert = nonneg_preserving_round(
        mps, max_bond_dim, tolerance, max_iterations,
    )
    return result_mps, total_error, cert, cert.clamping_certificate


def clamping_error_bound(
    mps: MPS,
    truncation_epsilon: float,
) -> float:
    """
    Compute the certified clamping error bound (Proposition 1, strengthened).

    For a probability vector p_exact ≥ 0 and its SVD truncation p_SVD
    with ‖p_exact - p_SVD‖₁ ≤ ε_trunc:

    ‖p_exact - p_clamped‖₁ ≤ ε_trunc

    Proof (Lemma 2): For each index i, clamping moves p_SVD[i] closer
    to p_exact[i] (or leaves it unchanged). Therefore the L1 error
    cannot increase:
        ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁ ≤ ε_trunc

    The previous 2ε_trunc bound used an unnecessary triangle inequality.

    Args:
        mps: The truncated MPS (for reference, not used in bound computation).
        truncation_epsilon: The SVD truncation tolerance.

    Returns:
        Certified upper bound on ‖p_clamped - p_exact‖₁.
    """
    return truncation_epsilon


def estimate_negativity(mps: MPS, n_samples: int = 10000) -> float:
    """
    Estimate the total negative mass of an MPS via sampling.

    Args:
        mps: Input MPS.
        n_samples: Number of random samples.

    Returns:
        Estimated total negative mass.
    """
    from tn_check.tensor.operations import mps_probability_at_index

    total_size = mps.full_size
    if total_size <= 1_000_000:
        v = mps_to_dense(mps)
        return float(np.sum(np.abs(v[v < 0])))

    rng = np.random.default_rng(42)
    neg_total = 0.0
    for _ in range(n_samples):
        idx = tuple(rng.integers(0, d) for d in mps.physical_dims)
        val = mps_probability_at_index(mps, idx)
        if val < 0:
            neg_total += abs(val)

    return neg_total / n_samples * total_size


def tight_clamping_bound(
    mps: MPS,
    truncation_epsilon: float,
    n_samples: int = 50000,
    safety_factor: float = 1.5,
) -> float:
    """
    Estimate a tighter clamping bound using measured negativity.

    By Lemma 2, the theoretical worst-case is already ε_trunc (not 2ε_trunc).
    Sampling can confirm the actual negativity is much smaller.

    Args:
        mps: The truncated MPS (may have negative entries).
        truncation_epsilon: SVD truncation tolerance.
        n_samples: Number of samples for negativity estimation.
        safety_factor: Multiplier for measured negativity (>1 for safety).

    Returns:
        Tighter upper bound: min(epsilon, measured_negativity * safety_factor).
    """
    measured_neg = estimate_negativity(mps, n_samples=n_samples)
    worst_case = truncation_epsilon
    tight = measured_neg * safety_factor
    return min(worst_case, tight)


def tight_clamping_bound_certified(
    mps: MPS,
    truncation_epsilon: float,
    n_samples: int = 50000,
) -> ClampingCertificate:
    """
    Compute the tight clamping bound with a full certificate.

    By Lemma 2, clamping reduces L1 error, so the bound is ε_trunc.
    Measured negativity N_neg is recorded for diagnostics.

    Args:
        mps: The truncated MPS (may have negative entries).
        truncation_epsilon: SVD truncation tolerance.
        n_samples: Number of samples for negativity estimation.

    Returns:
        ClampingCertificate with the tight bound and all proof ingredients.
    """
    measured_neg = estimate_negativity(mps, n_samples=n_samples)
    cert = ClampingCertificate(
        negativity_mass=measured_neg,
        truncation_error=truncation_epsilon,
    )
    cert.compute_bound()
    return cert


def verify_clamping_proposition(
    p_exact_dense: NDArray,
    p_svd_dense: NDArray,
) -> bool:
    """
    Numerically verify Lemma 1 (negativity bound) on small dense vectors.

    Checks that Σ_{i: p_SVD[i]<0} |p_SVD[i]| ≤ ‖p_exact - p_SVD‖₁
    (total negative mass is bounded by L1 truncation error).

    Args:
        p_exact_dense: Exact probability vector (non-negative).
        p_svd_dense: SVD-truncated vector (may have negatives).

    Returns:
        True if Lemma 1 holds (with numerical tolerance).
    """
    neg_mask = p_svd_dense < 0
    neg_mass = float(np.sum(np.abs(p_svd_dense[neg_mask])))
    l1_error = float(np.sum(np.abs(p_exact_dense - p_svd_dense)))
    return neg_mass <= l1_error + 1e-12


def verify_clamping_reduces_error(
    p_exact_dense: NDArray,
    p_svd_dense: NDArray,
) -> bool:
    """
    Numerically verify Lemma 2: clamping reduces L1 error.

    Checks that ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁ where
    p_clamped = max(p_SVD, 0). This confirms clamping does not increase
    the L1 distance to the exact solution.

    The verification proceeds index-wise:
    - If p_SVD[i] ≥ 0: |p_exact[i] - p_clamped[i]| = |p_exact[i] - p_SVD[i]|
    - If p_SVD[i] < 0: |p_exact[i] - 0| = p_exact[i]
                        ≤ p_exact[i] + |p_SVD[i]| = |p_exact[i] - p_SVD[i]|

    Args:
        p_exact_dense: Exact probability vector (non-negative).
        p_svd_dense: SVD-truncated vector (may have negatives).

    Returns:
        True if clamping reduces (or preserves) L1 error.
    """
    p_clamped = np.maximum(p_svd_dense, 0.0)
    l1_before = float(np.sum(np.abs(p_exact_dense - p_svd_dense)))
    l1_after = float(np.sum(np.abs(p_exact_dense - p_clamped)))
    return l1_after <= l1_before + 1e-12
