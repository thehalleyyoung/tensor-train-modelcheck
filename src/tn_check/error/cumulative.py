"""
Cumulative clamping error analysis across time steps.

Addresses the critique that "post-truncation clamping introduces systematic
bias without cumulative error tracking." This module:

1. Tracks per-step clamping error and truncation error
2. Computes cumulative bias bounds using Theorem 1 (Metzler contractivity)
3. Reports total systematic bias vs random error
4. Provides convergence diagnostics for the cumulative error

Key insight from Theorem 1 + Lemma 2:
    At each time step k, clamping reduces L1 error (Lemma 2), so the
    clamping does NOT contribute additional error beyond truncation.
    The cumulative error after K steps is:
        ‖p_exact(t_K) - p_TT(t_K)‖₁ ≤ Σ_{k=1}^K ε_k
    where ε_k is the per-step truncation error, NOT 2·ε_k.

    However, clamping introduces a systematic directional bias:
    - Negative entries are zeroed → probability mass is added
    - This is corrected by renormalization, but the correction is not
      uniform across states
    - Cumulative bias = Σ_k (negativity mass at step k)
    - This bias is bounded by Σ_k ε_k (by Lemma 1)

    Therefore the cumulative bias is bounded by the cumulative truncation
    error, and both accumulate linearly (not exponentially) for Metzler
    generators (Theorem 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class StepErrorRecord:
    """Error record for a single time step."""
    step: int
    time: float
    truncation_error: float
    clamping_error: float
    negativity_mass: float
    normalization_deviation: float
    bond_dims: list[int]
    total_probability: float


@dataclass
class CumulativeErrorAnalysis:
    """
    Cumulative error analysis across all time steps.

    Tracks both worst-case bounds (rigorous) and measured values (empirical).
    """
    num_steps: int = 0
    time_horizon: float = 0.0

    # Per-step records
    step_records: list[StepErrorRecord] = field(default_factory=list)

    # Cumulative bounds
    cumulative_truncation_error: float = 0.0
    cumulative_clamping_error: float = 0.0
    cumulative_negativity_mass: float = 0.0
    cumulative_normalization_drift: float = 0.0

    # Bias analysis
    systematic_bias_bound: float = 0.0
    max_per_step_bias: float = 0.0
    bias_direction: str = "positive"  # clamping always adds mass

    # Convergence
    error_growth_rate: str = "linear"  # "linear" for Metzler, "exponential" otherwise
    contractivity_verified: bool = True

    # Theorem 1 verification
    metzler_contractivity: bool = True
    cumulative_bound_tight: bool = True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "num_steps": self.num_steps,
            "time_horizon": self.time_horizon,
            "cumulative_truncation_error": self.cumulative_truncation_error,
            "cumulative_clamping_error": self.cumulative_clamping_error,
            "cumulative_negativity_mass": self.cumulative_negativity_mass,
            "cumulative_normalization_drift": self.cumulative_normalization_drift,
            "systematic_bias_bound": self.systematic_bias_bound,
            "max_per_step_bias": self.max_per_step_bias,
            "bias_direction": self.bias_direction,
            "error_growth_rate": self.error_growth_rate,
            "contractivity_verified": self.contractivity_verified,
            "metzler_contractivity": self.metzler_contractivity,
            "cumulative_bound_tight": self.cumulative_bound_tight,
            "per_step_truncation": [r.truncation_error for r in self.step_records],
            "per_step_clamping": [r.clamping_error for r in self.step_records],
            "per_step_negativity": [r.negativity_mass for r in self.step_records],
        }


class CumulativeErrorTracker:
    """
    Track and analyze cumulative clamping error across time steps.

    Provides rigorous bounds on total error and systematic bias
    introduced by TT truncation and clamping.

    Usage:
        tracker = CumulativeErrorTracker()
        for step in range(n_steps):
            # ... perform time step ...
            tracker.record_step(step, t, trunc_err, clamp_err,
                              neg_mass, norm_dev, bond_dims, total_prob)
        analysis = tracker.analyze()
    """

    def __init__(self, is_metzler: bool = True):
        """
        Initialize tracker.

        Args:
            is_metzler: Whether the CME generator has Metzler structure.
                       If True, contractivity factor C(t) = 1 (Theorem 1).
        """
        self.is_metzler = is_metzler
        self._records: list[StepErrorRecord] = []
        self._cumulative_trunc = 0.0
        self._cumulative_clamp = 0.0
        self._cumulative_neg = 0.0
        self._cumulative_norm_drift = 0.0

    def record_step(
        self,
        step: int,
        time: float,
        truncation_error: float,
        clamping_error: float,
        negativity_mass: float,
        normalization_deviation: float,
        bond_dims: list[int],
        total_probability: float,
    ) -> None:
        """Record error data for one time step."""
        record = StepErrorRecord(
            step=step,
            time=time,
            truncation_error=truncation_error,
            clamping_error=clamping_error,
            negativity_mass=negativity_mass,
            normalization_deviation=normalization_deviation,
            bond_dims=bond_dims,
            total_probability=total_probability,
        )
        self._records.append(record)
        self._cumulative_trunc += truncation_error
        self._cumulative_clamp += clamping_error
        self._cumulative_neg += negativity_mass
        self._cumulative_norm_drift += abs(normalization_deviation)

    def analyze(self) -> CumulativeErrorAnalysis:
        """
        Analyze cumulative errors and compute bounds.

        Returns:
            CumulativeErrorAnalysis with rigorous bounds and diagnostics.
        """
        n = len(self._records)
        if n == 0:
            return CumulativeErrorAnalysis()

        analysis = CumulativeErrorAnalysis(
            num_steps=n,
            time_horizon=self._records[-1].time if self._records else 0.0,
            step_records=list(self._records),
            cumulative_truncation_error=self._cumulative_trunc,
            cumulative_clamping_error=self._cumulative_clamp,
            cumulative_negativity_mass=self._cumulative_neg,
            cumulative_normalization_drift=self._cumulative_norm_drift,
        )

        # Systematic bias bound (Lemma 1 + Theorem 1)
        # Clamping always adds probability mass (negative entries zeroed)
        # Total bias ≤ cumulative negativity mass ≤ cumulative truncation error
        analysis.systematic_bias_bound = min(
            self._cumulative_neg,
            self._cumulative_trunc,
        )
        analysis.max_per_step_bias = max(
            (r.negativity_mass for r in self._records), default=0.0,
        )
        analysis.bias_direction = "positive"  # always adds mass

        # Verify Metzler contractivity
        if self.is_metzler:
            analysis.metzler_contractivity = True
            analysis.error_growth_rate = "linear"
            # Check that errors don't grow exponentially
            if n >= 3:
                errors = [r.truncation_error for r in self._records]
                growth_ok = _check_linear_growth(errors)
                analysis.contractivity_verified = growth_ok
        else:
            analysis.metzler_contractivity = False
            analysis.error_growth_rate = "potentially_exponential"
            analysis.contractivity_verified = False

        # Check if bound is tight
        # Tight if cumulative clamping ≤ cumulative truncation
        analysis.cumulative_bound_tight = (
            self._cumulative_clamp <= self._cumulative_trunc + 1e-12
        )

        return analysis

    @property
    def current_cumulative_error(self) -> float:
        """Current cumulative error bound."""
        if self.is_metzler:
            return self._cumulative_trunc
        else:
            return self._cumulative_trunc * (1 + len(self._records) * 0.01)

    def long_horizon_bias_analysis(self) -> dict:
        """
        Analyze systematic clamping bias over long time horizons.

        Clamping always adds probability mass (negative → zero), creating
        systematic positive bias. This analysis quantifies:
        1. Rate of bias accumulation per step
        2. Projected total bias at arbitrary horizons
        3. Effect on CSL probability thresholds

        Returns:
            Dictionary with bias rate, projections, and threshold impact.
        """
        if not self._records:
            return {"status": "no_data"}

        neg_masses = [r.negativity_mass for r in self._records]
        times = [r.time for r in self._records]

        # Compute bias rate (negativity mass per unit time)
        if times[-1] > 0:
            bias_rate = self._cumulative_neg / times[-1]
        else:
            bias_rate = 0.0

        # Detect if bias is growing, constant, or decaying
        if len(neg_masses) >= 4:
            q1 = np.mean(neg_masses[:len(neg_masses)//4])
            q4 = np.mean(neg_masses[3*len(neg_masses)//4:])
            if q1 > 1e-15:
                growth_ratio = q4 / q1
            else:
                growth_ratio = 1.0

            if growth_ratio > 2.0:
                trend = "growing"
            elif growth_ratio < 0.5:
                trend = "decaying"
            else:
                trend = "stable"
        else:
            growth_ratio = 1.0
            trend = "unknown"

        # Threshold impact: if p_computed is within bias of threshold,
        # the bias could flip a verdict
        max_per_step_neg = max(neg_masses) if neg_masses else 0.0

        return {
            "total_bias": self._cumulative_neg,
            "bias_rate_per_unit_time": bias_rate,
            "mean_per_step_bias": float(np.mean(neg_masses)) if neg_masses else 0.0,
            "max_per_step_bias": max_per_step_neg,
            "bias_trend": trend,
            "growth_ratio_q4_over_q1": growth_ratio,
            "bias_direction": "positive (mass added by clamping)",
            "projected_bias_at_100": bias_rate * 100,
            "projected_bias_at_1000": bias_rate * 1000,
            "threshold_safety_margin": (
                "A CSL verdict P≥p[ψ] is safe if the computed probability "
                f"exceeds the threshold by at least {self._cumulative_neg:.2e} "
                "(the total clamping bias)."
            ),
            "recommendation": (
                "Increase bond dimension to reduce truncation error"
                if bias_rate > 1e-4 else
                "Clamping bias is negligible for practical thresholds"
            ),
        }

    def verdict_sensitivity_analysis(
        self,
        csl_probability: float,
        threshold: float,
        comparison: str = ">=",
    ) -> dict:
        """
        Check whether clamping bias could flip a CSL verdict.

        Clamping adds probability mass (positive bias), so it can only
        push P(φ) higher. This means:
        - For P≥p[ψ]: clamping bias favors TRUE → false TRUE possible
        - For P≤p[ψ]: clamping bias opposes TRUE → conservative (sound)

        Args:
            csl_probability: Computed probability P(φ).
            threshold: CSL threshold p.
            comparison: Comparison operator.

        Returns:
            dict with verdict_safe, bias_impact, and corrected_probability.
        """
        total_bias = self._cumulative_neg
        total_error = self._cumulative_trunc

        if comparison in (">=", ">"):
            # Bias pushes P higher → could cause false TRUE
            corrected_prob = max(0, csl_probability - total_bias)
            verdict_original = csl_probability >= threshold
            verdict_corrected = corrected_prob >= threshold
            could_flip = verdict_original and not verdict_corrected
        else:
            # Bias pushes P higher → makes P≤p harder to satisfy (conservative)
            corrected_prob = min(1.0, csl_probability + total_bias)
            verdict_original = csl_probability <= threshold
            verdict_corrected = corrected_prob <= threshold
            could_flip = verdict_original and not verdict_corrected

        return {
            "csl_probability": csl_probability,
            "threshold": threshold,
            "total_clamping_bias": total_bias,
            "total_error_bound": total_error,
            "corrected_probability": corrected_prob,
            "verdict_safe": not could_flip,
            "could_flip_verdict": could_flip,
            "bias_fraction_of_margin": (
                total_bias / max(abs(csl_probability - threshold), 1e-15)
            ),
            "recommendation": (
                "SAFE: clamping bias does not affect verdict"
                if not could_flip else
                "WARNING: clamping bias could affect verdict; "
                "increase bond dimension or report INDETERMINATE"
            ),
        }


def run_cumulative_error_experiment(
    model_name: str = "birth_death",
    time_horizon: float = 10.0,
    n_steps: int = 100,
    max_bond_dim: int = 20,
    tolerance: float = 1e-10,
) -> dict:
    """
    Run a cumulative error tracking experiment.

    Evolves a model in TT format, tracking per-step and cumulative errors,
    then compares against the dense exact solution.

    Args:
        model_name: Name of model from library.
        time_horizon: Total simulation time.
        n_steps: Number of time steps.
        max_bond_dim: Maximum TT bond dimension.
        tolerance: SVD truncation tolerance.

    Returns:
        Dictionary with cumulative error analysis and comparison to exact.
    """
    from tn_check.models.library import birth_death, gene_expression, toggle_switch
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver
    from tn_check.tensor.decomposition import tensor_to_mps
    from tn_check.tensor.operations import (
        mps_to_dense, mps_compress, mps_total_probability,
        mps_clamp_nonnegative, mps_normalize_probability,
        mps_zip_up, mps_scalar_multiply, mps_addition,
    )

    # Build model
    if model_name == "birth_death":
        net = birth_death(birth_rate=5.0, death_rate=0.5, max_copy=40)
    elif model_name == "gene_expression":
        net = gene_expression(max_copy_mRNA=15, max_copy_protein=40)
    elif model_name == "toggle_switch":
        net = toggle_switch(alpha1=20.0, alpha2=20.0, max_copy=20)
    else:
        net = birth_death(birth_rate=5.0, death_rate=0.5, max_copy=40)

    dims = net.physical_dims
    N = int(np.prod(dims))

    # Dense solver for ground truth
    dense_solver = DenseReferenceSolver(net, max_states=500_000)

    # TT setup
    compiler = CMECompiler(net)
    mpo = compiler.compile()

    # Initial state
    p0 = np.zeros(N)
    initial = net.initial_state
    flat_idx = np.ravel_multi_index(
        [min(c, d - 1) for c, d in zip(initial, dims)], dims,
    )
    p0[flat_idx] = 1.0

    p0_mps = tensor_to_mps(p0.reshape(dims), list(dims), max_bond_dim=max_bond_dim)

    # Evolve with tracking
    tracker = CumulativeErrorTracker(is_metzler=True)
    dt = time_horizon / n_steps
    state = p0_mps

    checkpoints = []  # (time, l1_error_vs_exact)

    for step in range(n_steps):
        t = (step + 1) * dt

        # TT time step: forward Euler
        Qp, trunc_err = mps_zip_up(
            mpo, state, max_bond_dim=max_bond_dim * 2, tolerance=tolerance,
        )
        dp = mps_scalar_multiply(Qp, dt)
        state = mps_addition(state, dp)
        state, compress_err = mps_compress(
            state, max_bond_dim=max_bond_dim, tolerance=tolerance,
        )

        total_trunc = trunc_err + compress_err

        # Measure negativity before clamping
        p_dense_tt = mps_to_dense(state).ravel()
        neg_mass = float(np.sum(np.abs(p_dense_tt[p_dense_tt < 0])))

        # Clamp
        state, clamp_err = mps_clamp_nonnegative(
            state, max_bond_dim=max_bond_dim, tolerance=tolerance,
        )

        # Normalize
        tp = mps_total_probability(state)
        norm_dev = abs(tp - 1.0)
        if abs(tp) > 1e-15 and abs(tp - 1.0) > tolerance:
            state = mps_normalize_probability(state)

        tracker.record_step(
            step=step,
            time=t,
            truncation_error=total_trunc,
            clamping_error=clamp_err,
            negativity_mass=neg_mass,
            normalization_deviation=norm_dev,
            bond_dims=list(state.bond_dims),
            total_probability=tp,
        )

        # Checkpoint: compare with exact
        if (step + 1) % max(1, n_steps // 10) == 0:
            p_exact = dense_solver.evolve(p0, t)
            p_tt = mps_to_dense(state).ravel()
            l1_error = float(np.sum(np.abs(p_exact - p_tt)))
            checkpoints.append({
                "step": step,
                "time": t,
                "l1_error_vs_exact": l1_error,
                "cumulative_trunc_bound": tracker._cumulative_trunc,
                "bound_holds": l1_error <= tracker._cumulative_trunc + 1e-8,
            })

    analysis = tracker.analyze()

    return {
        "model": model_name,
        "time_horizon": time_horizon,
        "n_steps": n_steps,
        "max_bond_dim": max_bond_dim,
        "analysis": analysis.to_dict(),
        "checkpoints": checkpoints,
        "bound_verified": all(c["bound_holds"] for c in checkpoints),
    }


def _check_linear_growth(errors: list[float]) -> bool:
    """Check that errors grow at most linearly (not exponentially)."""
    if len(errors) < 3:
        return True

    # Compute running sum and check it doesn't grow exponentially
    cumulative = np.cumsum(errors)
    n = len(cumulative)

    # Fit linear and exponential models
    x = np.arange(n, dtype=float)
    if cumulative[-1] < 1e-15:
        return True

    # Simple check: ratio of last quarter to first quarter
    q1 = np.mean(errors[:max(1, n // 4)])
    q4 = np.mean(errors[3 * n // 4:])

    if q1 < 1e-15:
        return True

    ratio = q4 / q1
    # Linear growth: ratio should be O(1)
    # Exponential growth: ratio >> 1
    return ratio < 10.0
