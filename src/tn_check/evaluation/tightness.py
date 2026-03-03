"""
Error bound tightness study.

Empirically compares Theorem 7's certified error bound to actual L1 error
on models with known exact solutions. Reports tightness ratios and identifies
conditions under which the bound is tight vs conservative.

Key question addressed: "The tightness of the error bound is uncharacterized.
How conservative is the certified bound compared to actual errors?"

The tightness ratio T = ε_certified / ε_actual measures how conservative
the bound is:
- T ≈ 1: bound is tight (good)
- T ≫ 1: bound is conservative (safe but wasteful)
- T < 1: would indicate a bug (bound is violated)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TightnessResult:
    """Result of error bound tightness analysis for a single model."""
    model_name: str
    num_species: int
    state_space_size: int
    time_horizon: float
    max_bond_dim: int

    # Error measurements
    certified_error: float  # Theorem 7 bound
    actual_l1_error: float  # ||p_exact - p_TT||₁
    actual_l2_error: float  # ||p_exact - p_TT||₂
    actual_linf_error: float  # ||p_exact - p_TT||_∞

    # Error source breakdown
    fsp_error_bound: float
    tt_error_bound: float
    tdvp_error_bound: float
    clamping_error_bound: float

    # Actual error sources (when measurable)
    actual_fsp_error: float = 0.0
    actual_tt_error: float = 0.0

    # Tightness
    tightness_ratio: float = 0.0  # certified / actual
    log_tightness: float = 0.0

    # Diagnostics
    bound_valid: bool = True  # certified >= actual (no violation)
    wall_time: float = 0.0

    def __post_init__(self):
        if self.actual_l1_error > 0:
            self.tightness_ratio = self.certified_error / self.actual_l1_error
            self.log_tightness = np.log10(max(self.tightness_ratio, 1e-15))
        self.bound_valid = self.certified_error >= self.actual_l1_error - 1e-12


@dataclass
class TightnessStudy:
    """Comprehensive error bound tightness study results."""
    results: list[TightnessResult]
    models_tested: int
    mean_tightness_ratio: float = 0.0
    median_tightness_ratio: float = 0.0
    min_tightness_ratio: float = 0.0
    max_tightness_ratio: float = 0.0
    bound_violations: int = 0  # should be 0
    tightest_model: str = ""
    loosest_model: str = ""
    analysis: dict = field(default_factory=dict)
    wall_time: float = 0.0

    def __post_init__(self):
        if self.results:
            ratios = [r.tightness_ratio for r in self.results if r.tightness_ratio > 0]
            if ratios:
                self.mean_tightness_ratio = float(np.mean(ratios))
                self.median_tightness_ratio = float(np.median(ratios))
                self.min_tightness_ratio = float(np.min(ratios))
                self.max_tightness_ratio = float(np.max(ratios))
                self.bound_violations = sum(1 for r in self.results if not r.bound_valid)
                self.tightest_model = min(self.results, key=lambda r: r.tightness_ratio if r.tightness_ratio > 0 else float('inf')).model_name
                self.loosest_model = max(self.results, key=lambda r: r.tightness_ratio).model_name


def run_tightness_study(
    bond_dims: Sequence[int] = (5, 10, 20, 50),
    time_horizon: float = 1.0,
    models: Optional[list] = None,
) -> TightnessStudy:
    """
    Run comprehensive error bound tightness study.

    For each model with known exact solution, computes:
    1. Exact solution via dense matrix exponential
    2. TT-compressed solution via TDVP
    3. Certified error bound from Theorem 7
    4. Actual L1 error
    5. Tightness ratio = certified / actual

    Args:
        bond_dims: Bond dimensions to test.
        time_horizon: Simulation time.
        models: Optional list of (name, network) tuples. If None, uses
                built-in models with exact solutions.

    Returns:
        TightnessStudy with per-model and aggregate results.
    """
    start_time = time.time()

    if models is None:
        models = _get_exact_solution_models()

    results = []

    for model_name, network in models:
        for chi in bond_dims:
            try:
                result = _tightness_single_model(
                    model_name, network, chi, time_horizon,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Tightness study failed for {model_name} χ={chi}: {e}")
                continue

    study = TightnessStudy(
        results=results,
        models_tested=len(models),
        wall_time=time.time() - start_time,
    )

    # Compute analysis
    study.analysis = _analyze_tightness(results)

    return study


def _tightness_single_model(
    model_name: str,
    network,
    max_bond_dim: int,
    time_horizon: float,
) -> TightnessResult:
    """Compute tightness for a single model and bond dimension."""
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.cme.fsp import compute_fsp_bounds
    from tn_check.solver.dense_reference import DenseReferenceSolver
    from tn_check.integrator.tdvp import TDVPIntegrator
    from tn_check.tensor.operations import mps_to_dense, mps_total_probability
    from tn_check.error.propagation import unified_error_theorem

    start = time.time()

    # Compute exact solution
    solver = DenseReferenceSolver(network)
    solver.compile()

    # Build initial probability vector
    dims = list(network.physical_dims)
    state_space_size = int(np.prod(dims))
    p0 = np.zeros(state_space_size)
    initial_counts = [sp.initial_count for sp in network.species]
    # Compute flat index
    flat_idx = 0
    multiplier = 1
    for k in range(len(dims) - 1, -1, -1):
        flat_idx += min(initial_counts[k], dims[k] - 1) * multiplier
        multiplier *= dims[k]
    if flat_idx < state_space_size:
        p0[flat_idx] = 1.0
    else:
        p0[0] = 1.0

    p_exact = solver.evolve(p0, time_horizon)

    # Compute TT solution
    compiler = CMECompiler(network)
    generator = compiler.compile()
    initial = deterministic_initial_state(network)

    integrator = TDVPIntegrator(
        generator, max_bond_dim=max_bond_dim, dt=0.01,
        two_site=True,
    )
    result = integrator.integrate(initial, time_horizon)

    # Convert TT to dense for comparison
    p_tt = mps_to_dense(result.final_state)

    # Ensure same size for comparison
    min_len = min(len(p_exact), len(p_tt))
    p_exact_trunc = p_exact[:min_len]
    p_tt_trunc = p_tt[:min_len]

    # Actual errors
    actual_l1 = float(np.sum(np.abs(p_exact_trunc - p_tt_trunc)))
    actual_l2 = float(np.sqrt(np.sum((p_exact_trunc - p_tt_trunc) ** 2)))
    actual_linf = float(np.max(np.abs(p_exact_trunc - p_tt_trunc)))

    # Compute FSP error
    fsp_bounds = compute_fsp_bounds(network, time_horizon)
    fsp_error = fsp_bounds.probability_loss_bound

    # Certified error from Theorem 7
    # Include both truncation and integration errors
    num_steps = max(result.total_steps, 1)
    per_step_trunc = result.total_truncation_error / num_steps
    per_step_int = result.total_integration_error / num_steps if hasattr(result, 'total_integration_error') else 0.0
    per_step_errors = [per_step_trunc] * num_steps
    frob_residuals = [per_step_int / max(0.01, 1e-15)] * num_steps
    effective_support = max(1, int(np.sum(p_exact_trunc > 1e-10)))

    # The total certified error should encompass all error sources
    # including the integration error from TDVP
    integration_error = result.total_integration_error if hasattr(result, 'total_integration_error') else 0.0

    theorem7 = unified_error_theorem(
        fsp_error=fsp_error,
        per_step_truncation_errors=per_step_errors,
        frobenius_residuals=frob_residuals,
        effective_support=effective_support,
        dt=0.01,
        is_metzler=True,
        threshold=0.5,
    )

    # Add integration error that Theorem 7 might not fully capture
    certified_error = theorem7["total_certified_error"] + integration_error

    return TightnessResult(
        model_name=model_name,
        num_species=network.num_species,
        state_space_size=int(np.prod(list(network.physical_dims))),
        time_horizon=time_horizon,
        max_bond_dim=max_bond_dim,
        certified_error=certified_error,
        actual_l1_error=actual_l1,
        actual_l2_error=actual_l2,
        actual_linf_error=actual_linf,
        fsp_error_bound=fsp_error,
        tt_error_bound=result.total_truncation_error,
        tdvp_error_bound=theorem7["error_sources"]["tdvp_projection"],
        clamping_error_bound=0.0,
        actual_fsp_error=abs(1.0 - float(np.sum(p_exact_trunc))),
        actual_tt_error=result.total_truncation_error,
        wall_time=time.time() - start,
    )


def _get_exact_solution_models() -> list[tuple[str, object]]:
    """Get models with known exact solutions for tightness study."""
    from tn_check.models.library import (
        birth_death,
        gene_expression,
        toggle_switch,
    )

    models = []

    # Birth-death: exact Poisson stationary distribution
    bd = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=30)
    models.append(("birth-death", bd))

    # Gene expression (2 species)
    try:
        ge = gene_expression(
            k_txn=0.5,
            k_tln=5.0,
            gamma_m=0.1,
            gamma_p=0.005,
            max_copy_mRNA=20,
            max_copy_protein=30,
        )
        models.append(("gene-expression", ge))
    except Exception:
        pass

    # Toggle switch (2 genes)
    try:
        ts = toggle_switch(max_copy=15)
        models.append(("toggle-switch", ts))
    except Exception:
        pass

    return models


def _analyze_tightness(results: list[TightnessResult]) -> dict:
    """Analyze tightness results and identify patterns."""
    if not results:
        return {"error": "No results to analyze"}

    valid = [r for r in results if r.tightness_ratio > 0]
    if not valid:
        return {"error": "No valid tightness ratios"}

    ratios = np.array([r.tightness_ratio for r in valid])
    log_ratios = np.log10(np.maximum(ratios, 1e-15))

    # Group by model
    by_model = {}
    for r in valid:
        if r.model_name not in by_model:
            by_model[r.model_name] = []
        by_model[r.model_name].append(r)

    # Group by bond dimension
    by_chi = {}
    for r in valid:
        if r.max_bond_dim not in by_chi:
            by_chi[r.max_bond_dim] = []
        by_chi[r.max_bond_dim].append(r)

    return {
        "overall": {
            "mean_tightness": float(np.mean(ratios)),
            "median_tightness": float(np.median(ratios)),
            "std_tightness": float(np.std(ratios)),
            "mean_log_tightness": float(np.mean(log_ratios)),
            "bound_violations": sum(1 for r in valid if not r.bound_valid),
        },
        "by_model": {
            name: {
                "mean_tightness": float(np.mean([r.tightness_ratio for r in rs])),
                "num_tests": len(rs),
            }
            for name, rs in by_model.items()
        },
        "by_bond_dim": {
            chi: {
                "mean_tightness": float(np.mean([r.tightness_ratio for r in rs])),
                "num_tests": len(rs),
            }
            for chi, rs in by_chi.items()
        },
        "dominant_error_source": _identify_dominant_source(valid),
        "recommendations": _tightness_recommendations(valid),
    }


def _identify_dominant_source(results: list[TightnessResult]) -> dict:
    """Identify which error source dominates the certified bound."""
    fsp_dominated = 0
    tt_dominated = 0
    tdvp_dominated = 0

    for r in results:
        sources = {
            "fsp": r.fsp_error_bound,
            "tt": r.tt_error_bound,
            "tdvp": r.tdvp_error_bound,
        }
        dominant = max(sources, key=sources.get)
        if dominant == "fsp":
            fsp_dominated += 1
        elif dominant == "tt":
            tt_dominated += 1
        else:
            tdvp_dominated += 1

    total = len(results)
    return {
        "fsp_dominated_fraction": fsp_dominated / total if total > 0 else 0,
        "tt_dominated_fraction": tt_dominated / total if total > 0 else 0,
        "tdvp_dominated_fraction": tdvp_dominated / total if total > 0 else 0,
    }


def _tightness_recommendations(results: list[TightnessResult]) -> list[str]:
    """Generate recommendations based on tightness analysis."""
    recs = []

    ratios = [r.tightness_ratio for r in results if r.tightness_ratio > 0]
    if not ratios:
        return ["Insufficient data for recommendations"]

    mean_ratio = np.mean(ratios)

    if mean_ratio > 100:
        recs.append(
            "Certified bound is very conservative (>100x actual error). "
            "Consider using effective-support-aware Frobenius-to-L1 conversion."
        )
    elif mean_ratio > 10:
        recs.append(
            "Certified bound is moderately conservative (10-100x). "
            "The bound is useful for certification but not for optimization."
        )
    else:
        recs.append(
            "Certified bound is reasonably tight (<10x actual error). "
            "The bound accurately reflects the true error."
        )

    # Check if FSP dominates
    fsp_dominant = sum(1 for r in results if r.fsp_error_bound > r.tt_error_bound) / len(results)
    if fsp_dominant > 0.5:
        recs.append(
            "FSP error dominates in most cases. "
            "Consider adaptive FSP expansion or tighter FSP bounds."
        )

    return recs


def tightness_vs_bond_dim(
    network,
    bond_dims: Sequence[int] = (5, 10, 20, 50, 100),
    time_horizon: float = 1.0,
) -> dict:
    """
    Study how tightness varies with bond dimension for a single model.

    As χ increases, both certified and actual errors decrease, but
    the tightness ratio may change non-monotonically.

    Args:
        network: ReactionNetwork.
        bond_dims: Bond dimensions to test.
        time_horizon: Simulation time.

    Returns:
        Dictionary with per-χ tightness results.
    """
    results = []

    for chi in bond_dims:
        try:
            r = _tightness_single_model(
                network.name, network, chi, time_horizon,
            )
            results.append({
                "bond_dim": chi,
                "certified_error": r.certified_error,
                "actual_l1_error": r.actual_l1_error,
                "tightness_ratio": r.tightness_ratio,
                "bound_valid": r.bound_valid,
            })
        except Exception as e:
            logger.warning(f"Tightness test failed for χ={chi}: {e}")
            continue

    return {
        "network": network.name,
        "results": results,
        "num_tested": len(results),
    }
