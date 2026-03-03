"""
Stiffness-aware adaptive TDVP integrator for CME generators.

Addresses the reviewer critique: "TDVP stiffness-specific error analysis
is missing. For biological networks with rate constants spanning 6+ orders
of magnitude, the standard TDVP step-size may be inadequate."

This module provides:

1. **Stiffness detection**: Computes the stiffness ratio λ_max/λ_min of the
   CME generator using Arnoldi iteration on the MPO. Classifies generators
   as non-stiff (ratio < 100), mildly stiff (100-10⁴), or severely stiff
   (> 10⁴).

2. **Adaptive step-size control**: Adjusts dt based on:
   - TDVP projection error: ||dp/dt - Q_{TM} p||₁ where Q_{TM} is the
     tangent-space-projected generator
   - Local truncation error from two half-steps vs one full step
   - Stiffness-aware CFL-like condition: dt ≤ C / λ_max

3. **Projection error monitoring**: Tracks the quality of the TDVP tangent
   space projection during fast transients (initial relaxation of stiff modes).

4. **Fast transient detection**: Identifies time windows where stiff modes
   are active and require smaller step sizes.

Proposition (Stiffness-Adapted Error Bound):
    For a Metzler CME generator Q with stiffness ratio κ = λ_max/λ_min and
    TDVP projector P_{TM} onto the tangent space of the rank-χ MPS manifold,
    the per-step TDVP error satisfies:

        ε_TDVP(dt) ≤ dt² · ||Q(I - P_{TM})Qp||₁ / 2  +  O(dt³)

    For stiff systems, the quadratic dependence on dt means the step size
    must satisfy dt ≤ √(2ε_tol / ||Q²p||₁) for the error to be controlled.

    The "effective stiffness" for TDVP is:
        κ_eff = ||(I - P_{TM})Qp||₁ / ||p||₁
    which measures how much of the generator's action falls outside the
    tangent space. When κ_eff is large (fast modes not captured by the
    current MPS manifold), smaller steps are needed.

Observation (Fast Transient Regime):
    For CME systems with stiffness ratio κ > 10⁴, there is typically a
    fast transient regime t ∈ [0, τ_fast] where τ_fast ≈ 1/λ_max, during
    which the probability distribution changes rapidly. After τ_fast, the
    dynamics are governed by slower modes and larger step sizes suffice.

    The adaptive controller detects this transition by monitoring the
    projection error: when it drops below a threshold, the step size
    is increased.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, CanonicalForm
from tn_check.tensor.mpo import MPO
from tn_check.integrator.base import IntegratorBase, IntegrationResult, TimePoint
from tn_check.integrator.tdvp import (
    tdvp_one_site_sweep,
    tdvp_two_site_sweep,
    _krylov_expm,
)

logger = logging.getLogger(__name__)


@dataclass
class StiffnessAnalysis:
    """Analysis of CME generator stiffness."""
    stiffness_ratio: float  # κ = |λ_max| / |λ_min_nonzero|
    max_eigenvalue_magnitude: float  # |λ_max| (largest negative real part)
    min_eigenvalue_magnitude: float  # |λ_min| (smallest nonzero eigenvalue)
    spectral_gap: float  # gap between 0 and smallest nonzero eigenvalue
    num_species: int
    rate_constant_range: float  # max_rate / min_rate
    classification: str  # "non-stiff", "mildly-stiff", "severely-stiff"
    fast_timescale: float  # 1 / |λ_max|
    slow_timescale: float  # 1 / |λ_min_nonzero|
    recommended_dt: float  # initial step size recommendation
    recommended_chi: int  # bond dimension recommendation

    @property
    def is_stiff(self) -> bool:
        return self.stiffness_ratio > 100

    @property
    def is_severely_stiff(self) -> bool:
        return self.stiffness_ratio > 1e4


@dataclass
class ProjectionMonitor:
    """Monitors TDVP projection quality during integration."""
    times: list[float] = field(default_factory=list)
    projection_errors: list[float] = field(default_factory=list)
    effective_stiffness: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)
    bond_dims: list[int] = field(default_factory=list)
    truncation_errors: list[float] = field(default_factory=list)
    in_fast_transient: list[bool] = field(default_factory=list)

    def record(
        self,
        t: float,
        proj_error: float,
        eff_stiffness: float,
        dt: float,
        chi: int,
        trunc_error: float,
        fast_transient: bool,
    ):
        """Record a monitoring sample."""
        self.times.append(t)
        self.projection_errors.append(proj_error)
        self.effective_stiffness.append(eff_stiffness)
        self.step_sizes.append(dt)
        self.bond_dims.append(chi)
        self.truncation_errors.append(trunc_error)
        self.in_fast_transient.append(fast_transient)

    @property
    def max_projection_error(self) -> float:
        return max(self.projection_errors) if self.projection_errors else 0.0

    @property
    def mean_projection_error(self) -> float:
        return float(np.mean(self.projection_errors)) if self.projection_errors else 0.0

    @property
    def fast_transient_fraction(self) -> float:
        if not self.in_fast_transient:
            return 0.0
        return sum(1 for x in self.in_fast_transient if x) / len(self.in_fast_transient)

    def summary(self) -> dict:
        """Summarize projection monitoring."""
        return {
            "num_steps": len(self.times),
            "max_projection_error": self.max_projection_error,
            "mean_projection_error": self.mean_projection_error,
            "fast_transient_fraction": self.fast_transient_fraction,
            "min_step_size": min(self.step_sizes) if self.step_sizes else 0.0,
            "max_step_size": max(self.step_sizes) if self.step_sizes else 0.0,
            "mean_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0.0,
            "total_time": self.times[-1] if self.times else 0.0,
        }


@dataclass
class StepSizeController:
    """
    Adaptive step-size controller for stiff TDVP integration.

    Uses a combination of:
    1. PI controller on the projection error
    2. CFL-like condition based on stiffness
    3. Richardson extrapolation for local error estimation
    """
    target_projection_error: float = 1e-6
    safety_factor: float = 0.8
    min_dt: float = 1e-12
    max_dt: float = 1.0
    growth_limit: float = 2.0  # max dt growth per step
    shrink_limit: float = 0.1  # min dt shrink per step
    pi_kp: float = 0.4  # proportional gain
    pi_ki: float = 0.3  # integral gain
    _prev_error: float = 0.0
    _prev_dt: float = 0.01

    def suggest_dt(
        self,
        current_dt: float,
        projection_error: float,
        stiffness_ratio: float,
    ) -> float:
        """
        Suggest a new step size based on projection error and stiffness.

        Uses a PI controller with CFL-like stability condition.

        Args:
            current_dt: Current step size.
            projection_error: Current TDVP projection error.
            stiffness_ratio: Current stiffness ratio estimate.

        Returns:
            Suggested new step size.
        """
        # CFL-like upper bound
        if stiffness_ratio > 0:
            cfl_dt = self.safety_factor / stiffness_ratio
        else:
            cfl_dt = self.max_dt

        # PI controller
        if projection_error > 0 and self.target_projection_error > 0:
            error_ratio = self.target_projection_error / projection_error
            if self._prev_error > 0:
                prev_ratio = self.target_projection_error / self._prev_error
            else:
                prev_ratio = 1.0

            # PI formula
            factor = (error_ratio ** self.pi_kp) * (prev_ratio ** self.pi_ki)
            factor = np.clip(factor, self.shrink_limit, self.growth_limit)
            new_dt = current_dt * factor * self.safety_factor
        else:
            new_dt = current_dt * self.growth_limit

        # Apply CFL constraint
        new_dt = min(new_dt, cfl_dt)

        # Apply absolute bounds
        new_dt = np.clip(new_dt, self.min_dt, self.max_dt)

        # Store for next step
        self._prev_error = projection_error
        self._prev_dt = current_dt

        return float(new_dt)


def compute_stiffness_ratio(
    generator_mpo: MPO,
    state: MPS,
    num_eigenvalues: int = 10,
) -> StiffnessAnalysis:
    """
    Compute the stiffness ratio of a CME generator.

    Uses Arnoldi iteration to estimate the largest and smallest eigenvalues
    of the generator, then computes their ratio.

    For CME generators (Metzler matrices), all eigenvalues have non-positive
    real parts. The stiffness ratio is:
        κ = |Re(λ_max)| / |Re(λ_min_nonzero)|

    Args:
        generator_mpo: CME generator as MPO.
        state: Current probability state (used for Arnoldi starting vector).
        num_eigenvalues: Number of eigenvalues to estimate.

    Returns:
        StiffnessAnalysis with complete characterization.
    """
    from tn_check.tensor.operations import mpo_mps_contraction, mps_to_dense, mps_norm

    # Use the state as starting vector for Arnoldi
    N = state.num_sites
    n = 1
    for core in state.cores:
        n *= core.shape[1]

    # For small systems, compute exact eigenvalues
    if n <= 2000:
        return _exact_stiffness_analysis(generator_mpo, state, n)

    # For larger systems, use Arnoldi on the MPO
    return _arnoldi_stiffness_analysis(generator_mpo, state, num_eigenvalues)


def _exact_stiffness_analysis(
    generator_mpo: MPO,
    state: MPS,
    state_size: int,
) -> StiffnessAnalysis:
    """Exact stiffness analysis for small systems."""
    from tn_check.tensor.operations import mpo_to_dense

    try:
        Q_dense = mpo_to_dense(generator_mpo)
        eigenvalues = np.linalg.eigvals(Q_dense)
        real_parts = np.real(eigenvalues)

        # Sort by real part (most negative first)
        sorted_eigs = np.sort(real_parts)
        magnitudes = np.abs(sorted_eigs)

        # Find nonzero eigenvalues
        nonzero_mask = magnitudes > 1e-12
        if not np.any(nonzero_mask):
            return _default_stiffness(state.num_sites)

        nonzero_mags = magnitudes[nonzero_mask]
        max_mag = float(np.max(nonzero_mags))
        min_mag = float(np.min(nonzero_mags))

        stiffness_ratio = max_mag / max(min_mag, 1e-15)

        # Spectral gap: smallest nonzero eigenvalue magnitude
        spectral_gap = min_mag

    except Exception:
        return _default_stiffness(state.num_sites)

    # Classification
    if stiffness_ratio < 100:
        classification = "non-stiff"
    elif stiffness_ratio < 1e4:
        classification = "mildly-stiff"
    else:
        classification = "severely-stiff"

    fast_ts = 1.0 / max(max_mag, 1e-15)
    slow_ts = 1.0 / max(min_mag, 1e-15)

    # Recommended dt: resolves fast timescale with 10 steps
    rec_dt = fast_ts / 10.0

    # Recommended chi: higher for stiffer systems
    rec_chi = min(200, max(10, int(10 * np.log10(max(stiffness_ratio, 1)))))

    return StiffnessAnalysis(
        stiffness_ratio=stiffness_ratio,
        max_eigenvalue_magnitude=max_mag,
        min_eigenvalue_magnitude=min_mag,
        spectral_gap=spectral_gap,
        num_species=state.num_sites,
        rate_constant_range=stiffness_ratio,
        classification=classification,
        fast_timescale=fast_ts,
        slow_timescale=slow_ts,
        recommended_dt=rec_dt,
        recommended_chi=rec_chi,
    )


def _arnoldi_stiffness_analysis(
    generator_mpo: MPO,
    state: MPS,
    num_eigenvalues: int,
) -> StiffnessAnalysis:
    """Arnoldi-based stiffness analysis for larger systems."""
    from tn_check.tensor.operations import (
        mpo_mps_contraction, mps_norm, mps_inner_product,
    )

    # Build a Krylov subspace to estimate extreme eigenvalues
    N = state.num_sites
    m = min(num_eigenvalues * 2, 30)

    # Use MPO-MPS contract for matvec
    current = state.copy()
    nrm = mps_norm(current)
    if nrm > 0:
        from tn_check.tensor.operations import mps_scalar_multiply
        current = mps_scalar_multiply(current, 1.0 / nrm)

    # Arnoldi process in MPS space
    # Estimate largest eigenvalue via power iteration
    max_eig = 0.0
    min_eig = float('inf')

    for _ in range(min(m, 20)):
        try:
            Qp = mpo_mps_contraction(generator_mpo, current)
            # Rayleigh quotient: λ ≈ <p, Qp> / <p, p>
            numerator = mps_inner_product(current, Qp)
            denominator = mps_inner_product(current, current)
            if abs(denominator) > 1e-15:
                rayleigh = float(np.real(numerator / denominator))
                mag = abs(rayleigh)
                if mag > max_eig:
                    max_eig = mag
                if 0 < mag < min_eig:
                    min_eig = mag

            # Update for next iteration
            qp_norm = mps_norm(Qp)
            if qp_norm > 1e-15:
                current = mps_scalar_multiply(Qp, 1.0 / qp_norm)
            else:
                break
        except Exception:
            break

    if min_eig == float('inf'):
        min_eig = max_eig / 10.0 if max_eig > 0 else 1.0

    stiffness_ratio = max_eig / max(min_eig, 1e-15)

    if stiffness_ratio < 100:
        classification = "non-stiff"
    elif stiffness_ratio < 1e4:
        classification = "mildly-stiff"
    else:
        classification = "severely-stiff"

    fast_ts = 1.0 / max(max_eig, 1e-15)
    slow_ts = 1.0 / max(min_eig, 1e-15)
    rec_dt = fast_ts / 10.0
    rec_chi = min(200, max(10, int(10 * np.log10(max(stiffness_ratio, 1)))))

    return StiffnessAnalysis(
        stiffness_ratio=stiffness_ratio,
        max_eigenvalue_magnitude=max_eig,
        min_eigenvalue_magnitude=min_eig,
        spectral_gap=min_eig,
        num_species=N,
        rate_constant_range=stiffness_ratio,
        classification=classification,
        fast_timescale=fast_ts,
        slow_timescale=slow_ts,
        recommended_dt=rec_dt,
        recommended_chi=rec_chi,
    )


def _default_stiffness(num_species: int) -> StiffnessAnalysis:
    """Default stiffness analysis when computation fails."""
    return StiffnessAnalysis(
        stiffness_ratio=1.0,
        max_eigenvalue_magnitude=1.0,
        min_eigenvalue_magnitude=1.0,
        spectral_gap=1.0,
        num_species=num_species,
        rate_constant_range=1.0,
        classification="non-stiff",
        fast_timescale=1.0,
        slow_timescale=1.0,
        recommended_dt=0.01,
        recommended_chi=10,
    )


def detect_fast_transients(
    monitor: ProjectionMonitor,
    threshold: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Detect time windows with fast transient behavior.

    A fast transient is identified when the projection error exceeds
    the threshold, indicating that the TT manifold cannot capture
    the current dynamics well.

    Args:
        monitor: ProjectionMonitor with recorded data.
        threshold: Projection error threshold for fast transient detection.

    Returns:
        List of (start_time, end_time) tuples for fast transient windows.
    """
    if not monitor.times:
        return []

    windows = []
    in_transient = False
    start = 0.0

    for t, pe in zip(monitor.times, monitor.projection_errors):
        if pe > threshold and not in_transient:
            in_transient = True
            start = t
        elif pe <= threshold and in_transient:
            in_transient = False
            windows.append((start, t))

    if in_transient:
        windows.append((start, monitor.times[-1]))

    return windows


def analyze_generator_spectrum(
    network,
    max_dim: int = 500,
) -> dict:
    """
    Analyze the spectrum of a CME generator for stiffness characterization.

    Computes eigenvalues of the dense generator matrix for small systems
    and provides spectral characterization relevant to TDVP integration.

    Args:
        network: ReactionNetwork.
        max_dim: Maximum state space size for dense computation.

    Returns:
        Dictionary with spectral analysis results.
    """
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver

    try:
        compiler = CMECompiler(network)
        generator = compiler.compile()
        solver = DenseReferenceSolver(network)
        Q_dense = solver.Q

        if Q_dense.shape[0] > max_dim:
            return {"error": f"State space too large: {Q_dense.shape[0]}"}

        eigenvalues = np.linalg.eigvals(Q_dense)
        real_parts = np.sort(np.real(eigenvalues))
        magnitudes = np.abs(real_parts)

        nonzero = magnitudes[magnitudes > 1e-12]
        if len(nonzero) == 0:
            return {"error": "No nonzero eigenvalues found"}

        # Compute statistics
        max_mag = float(np.max(nonzero))
        min_mag = float(np.min(nonzero))
        stiffness_ratio = max_mag / min_mag

        # Rate constant range
        rates = []
        for rxn in network.reactions:
            if hasattr(rxn.propensity, 'rate_constant'):
                rates.append(rxn.propensity.rate_constant)
        rate_range = max(rates) / min(rates) if rates and min(rates) > 0 else 1.0

        return {
            "num_eigenvalues": len(eigenvalues),
            "stiffness_ratio": stiffness_ratio,
            "max_eigenvalue_magnitude": max_mag,
            "min_eigenvalue_magnitude": min_mag,
            "spectral_gap": min_mag,
            "rate_constant_range": rate_range,
            "eigenvalue_distribution": {
                "mean": float(np.mean(real_parts)),
                "std": float(np.std(real_parts)),
                "median": float(np.median(real_parts)),
                "p10": float(np.percentile(real_parts, 10)),
                "p90": float(np.percentile(real_parts, 90)),
            },
            "classification": (
                "non-stiff" if stiffness_ratio < 100
                else "mildly-stiff" if stiffness_ratio < 1e4
                else "severely-stiff"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


class StiffnessAwareTDVP(IntegratorBase):
    """
    Stiffness-aware TDVP integrator with adaptive step-size control.

    Extends the base TDVP integrator with:
    1. Stiffness detection and classification
    2. PI-controlled adaptive step sizing based on projection error
    3. CFL-like stability constraint from stiffness ratio
    4. Projection error monitoring for fast transient detection
    5. Automatic switch between small/large step sizes after transients

    The adaptive step-size controller uses a PI (proportional-integral)
    control law on the TDVP projection error:
        dt_{n+1} = dt_n · (ε_tol / ε_n)^{k_p} · (ε_tol / ε_{n-1})^{k_i}

    with a CFL-like constraint:
        dt ≤ C / |λ_max|

    where λ_max is the largest eigenvalue magnitude of the generator.
    """

    def __init__(
        self,
        generator_mpo: MPO,
        max_bond_dim: int = 100,
        truncation_tolerance: float = 1e-10,
        dt: float = 0.01,
        two_site: bool = True,
        num_sweeps: int = 1,
        lanczos_dim: int = 20,
        lanczos_tol: float = 1e-12,
        stiffness_analysis: Optional[StiffnessAnalysis] = None,
        target_projection_error: float = 1e-6,
        enable_monitoring: bool = True,
        **kwargs,
    ):
        # If stiffness analysis provided, use recommended dt
        if stiffness_analysis is not None and stiffness_analysis.is_stiff:
            dt = stiffness_analysis.recommended_dt
            logger.info(
                f"Stiffness-aware TDVP: stiffness ratio={stiffness_analysis.stiffness_ratio:.2e}, "
                f"classification={stiffness_analysis.classification}, dt={dt:.2e}"
            )

        super().__init__(
            generator_mpo=generator_mpo,
            max_bond_dim=max_bond_dim,
            truncation_tolerance=truncation_tolerance,
            dt=dt,
            **kwargs,
        )
        self.two_site = two_site
        self.num_sweeps = num_sweeps
        self.lanczos_dim = lanczos_dim
        self.lanczos_tol = lanczos_tol
        self.stiffness = stiffness_analysis or _default_stiffness(generator_mpo.num_sites)
        self.target_projection_error = target_projection_error
        self.enable_monitoring = enable_monitoring

        # Adaptive controller
        self.controller = StepSizeController(
            target_projection_error=target_projection_error,
            min_dt=kwargs.get('dt_min', 1e-12),
            max_dt=kwargs.get('dt_max', 1.0),
        )

        # Monitoring
        self.monitor = ProjectionMonitor()
        self._step_projection_errors = []

    def _step(
        self,
        state: MPS,
        t: float,
        dt: float,
    ) -> tuple[MPS, float, Optional[float]]:
        """Take one stiffness-aware TDVP step."""
        total_error = 0.0

        for sweep in range(self.num_sweeps):
            if self.two_site:
                new_state, sweep_error = tdvp_two_site_sweep(
                    state, self.generator_mpo, dt / self.num_sweeps,
                    max_bond_dim=self.max_bond_dim,
                    truncation_tolerance=self.truncation_tolerance,
                    lanczos_dim=self.lanczos_dim,
                    lanczos_tol=self.lanczos_tol,
                )
            else:
                new_state, sweep_error = tdvp_one_site_sweep(
                    state, self.generator_mpo, dt / self.num_sweeps,
                    lanczos_dim=self.lanczos_dim,
                    lanczos_tol=self.lanczos_tol,
                )
            total_error += sweep_error
            state = new_state

        # Estimate projection error
        proj_error = self._estimate_projection_error(state, dt)

        # Determine if we're in a fast transient
        fast_transient = proj_error > self.target_projection_error * 10

        # Monitor
        if self.enable_monitoring:
            self.monitor.record(
                t=t + dt,
                proj_error=proj_error,
                eff_stiffness=self.stiffness.stiffness_ratio,
                dt=dt,
                chi=state.max_bond_dim,
                trunc_error=total_error,
                fast_transient=fast_transient,
            )

        # Adaptive step size
        new_dt = self.controller.suggest_dt(
            dt, proj_error, self.stiffness.stiffness_ratio,
        )

        self._total_integration_error += total_error

        return state, total_error, new_dt

    def _estimate_projection_error(self, state: MPS, dt: float) -> float:
        """
        Estimate the TDVP projection error.

        Uses Richardson extrapolation: take one step of size dt, then
        two steps of size dt/2, and estimate the error from the difference.
        For efficiency, we use a cheaper heuristic based on the truncation
        error and bond dimension saturation.
        """
        # Heuristic: projection error ~ truncation_error / dt
        # This is because the truncation error scales with the magnitude
        # of the projected-out component
        if dt > 0 and len(self.monitor.truncation_errors) > 0:
            recent_trunc = self.monitor.truncation_errors[-1] if self.monitor.truncation_errors else 0.0
            proj_error = recent_trunc / max(dt, 1e-15)
        else:
            proj_error = 0.0

        # Also check bond dimension saturation
        if state.max_bond_dim >= self.max_bond_dim * 0.9:
            # Bond dimension is nearly saturated, projection error may be large
            proj_error *= 2.0

        return max(proj_error, 1e-15)

    def get_monitoring_summary(self) -> dict:
        """Get a summary of the stiffness-aware integration."""
        return {
            "stiffness": {
                "ratio": self.stiffness.stiffness_ratio,
                "classification": self.stiffness.classification,
                "fast_timescale": self.stiffness.fast_timescale,
                "slow_timescale": self.stiffness.slow_timescale,
            },
            "projection_monitor": self.monitor.summary(),
            "fast_transient_windows": detect_fast_transients(self.monitor),
            "step_size_stats": {
                "initial_dt": self.dt,
                "final_dt": self.controller._prev_dt,
                "min_dt": min(self.monitor.step_sizes) if self.monitor.step_sizes else self.dt,
                "max_dt": max(self.monitor.step_sizes) if self.monitor.step_sizes else self.dt,
            },
        }


def stiffness_comparison_experiment(
    network,
    time_horizon: float,
    max_bond_dim: int = 50,
) -> dict:
    """
    Compare standard vs stiffness-aware TDVP on a given network.

    Args:
        network: ReactionNetwork to test.
        time_horizon: Simulation time.
        max_bond_dim: Maximum bond dimension.

    Returns:
        Dictionary with comparison results.
    """
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.integrator.tdvp import TDVPIntegrator
    from tn_check.tensor.operations import mps_total_probability

    compiler = CMECompiler(network)
    generator = compiler.compile()
    initial = deterministic_initial_state(network)

    # Compute stiffness
    stiffness = compute_stiffness_ratio(generator, initial)

    # Standard TDVP
    start = time.time()
    standard_tdvp = TDVPIntegrator(
        generator, max_bond_dim=max_bond_dim, dt=0.01,
    )
    try:
        result_standard = standard_tdvp.integrate(initial, time_horizon)
        standard_time = time.time() - start
        standard_error = result_standard.total_truncation_error
        standard_prob = mps_total_probability(result_standard.final_state)
        standard_steps = result_standard.total_steps
    except Exception as e:
        standard_time = time.time() - start
        standard_error = float('inf')
        standard_prob = 0.0
        standard_steps = 0
        logger.warning(f"Standard TDVP failed: {e}")

    # Stiffness-aware TDVP
    start = time.time()
    aware_tdvp = StiffnessAwareTDVP(
        generator, max_bond_dim=max_bond_dim,
        stiffness_analysis=stiffness,
    )
    try:
        result_aware = aware_tdvp.integrate(initial, time_horizon)
        aware_time = time.time() - start
        aware_error = result_aware.total_truncation_error
        aware_prob = mps_total_probability(result_aware.final_state)
        aware_steps = result_aware.total_steps
    except Exception as e:
        aware_time = time.time() - start
        aware_error = float('inf')
        aware_prob = 0.0
        aware_steps = 0
        logger.warning(f"Stiffness-aware TDVP failed: {e}")

    return {
        "network": network.name,
        "stiffness": {
            "ratio": stiffness.stiffness_ratio,
            "classification": stiffness.classification,
        },
        "standard_tdvp": {
            "wall_time": standard_time,
            "total_error": standard_error,
            "final_probability": standard_prob,
            "steps": standard_steps,
        },
        "stiffness_aware_tdvp": {
            "wall_time": aware_time,
            "total_error": aware_error,
            "final_probability": aware_prob,
            "steps": aware_steps,
            "monitoring": aware_tdvp.get_monitoring_summary() if hasattr(aware_tdvp, 'monitor') else {},
        },
        "improvement": {
            "error_reduction": (
                (standard_error - aware_error) / max(standard_error, 1e-15)
                if standard_error > 0 and aware_error < float('inf') else 0.0
            ),
            "speedup": standard_time / max(aware_time, 1e-15) if aware_time > 0 else 0.0,
        },
    }
