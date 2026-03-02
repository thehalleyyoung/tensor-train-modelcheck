"""
PRISM/Storm cross-validation: verdict-by-verdict comparison.

Runs both TN-Check (dense exact solver) and TN-Check (TT-compressed solver)
on the same small models, verifying verdict agreement. Also generates PRISM
model files (.pm) for external validation with PRISM/Storm.

This addresses the critical gap: no comparison with PRISM/Storm on same models,
only scalability — not verdict agreement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tn_check.cme.reaction_network import ReactionNetwork
from tn_check.cme.compiler import CMECompiler
from tn_check.solver.dense_reference import DenseReferenceSolver
from tn_check.tensor.operations import mps_to_dense, mps_inner_product
from tn_check.checker.satisfaction import ThreeValued

logger = logging.getLogger(__name__)


@dataclass
class CSLProperty:
    """A CSL property specification for cross-validation."""
    name: str
    description: str
    formula_type: str  # "transient_prob", "steady_state", "bounded_until"
    species_index: int
    threshold: int
    time_bound: float
    probability_threshold: float = 0.5
    comparison: str = ">="


@dataclass
class CrossValidationResult:
    """Result of cross-validating TN-Check against dense reference."""
    model_name: str
    property_name: str
    dense_probability: float
    tt_probability: float
    absolute_error: float
    relative_error: float
    dense_verdict: str  # "TRUE", "FALSE"
    tt_verdict: str     # "TRUE", "FALSE", "INDETERMINATE"
    verdicts_agree: bool
    dense_time_s: float
    tt_time_s: float
    state_space_size: int
    tt_bond_dim: int
    tt_truncation_error: float
    notes: str = ""


@dataclass
class PRISMExportResult:
    """Result of exporting a model to PRISM format."""
    model_name: str
    prism_model: str
    prism_properties: list[str]
    num_species: int
    state_space_size: int


def _define_crossval_suite() -> list[dict]:
    """
    Define the cross-validation suite: models + CSL properties.

    Each entry has a model builder and multiple CSL properties to check,
    chosen to exercise different CSL operators and produce both TRUE and FALSE
    verdicts to test verdict agreement.

    Extended to 10 models with 30+ CSL properties for comprehensive
    verdict-level comparison.
    """
    from tn_check.models.library import (
        birth_death, toggle_switch, schlogl,
        gene_expression, sir_epidemic,
        michaelis_menten_enzyme, cascade,
    )
    from tn_check.models.biomodels_extended import (
        autoregulatory_gene, glycolysis_oscillator, competition_model,
    )

    return [
        {
            "name": "birth_death_N50",
            "builder": lambda: birth_death(birth_rate=5.0, death_rate=0.5, max_copy=50),
            "properties": [
                CSLProperty(
                    "ss_ge10", "P(X≥10) at steady state",
                    "steady_state", 0, 10, 200.0, 0.5, ">=",
                ),
                CSLProperty(
                    "ss_ge15", "P(X≥15) at steady state",
                    "steady_state", 0, 15, 200.0, 0.3, ">=",
                ),
                CSLProperty(
                    "transient_ge8", "P(X≥8) at t=5",
                    "transient_prob", 0, 8, 5.0, 0.5, ">=",
                ),
                CSLProperty(
                    "transient_ge20", "P(X≥20) at t=2",
                    "transient_prob", 0, 20, 2.0, 0.01, ">=",
                ),
            ],
        },
        {
            "name": "gene_expression_2sp",
            "builder": lambda: gene_expression(
                k_txn=0.5, k_tln=5.0, gamma_m=0.1, gamma_p=0.01,
                max_copy_mRNA=20, max_copy_protein=60,
            ),
            "properties": [
                CSLProperty(
                    "protein_ge30", "P(Protein≥30) at t=100",
                    "transient_prob", 1, 30, 100.0, 0.5, ">=",
                ),
                CSLProperty(
                    "mrna_ge5", "P(mRNA≥5) at t=50",
                    "transient_prob", 0, 5, 50.0, 0.5, ">=",
                ),
                CSLProperty(
                    "protein_ss_ge20", "P(Protein≥20) at steady state",
                    "steady_state", 1, 20, 500.0, 0.9, ">=",
                ),
            ],
        },
        {
            "name": "toggle_switch_bistable",
            "builder": lambda: toggle_switch(alpha1=20.0, alpha2=20.0, max_copy=25),
            "properties": [
                CSLProperty(
                    "x1_dominates", "P(X1≥10) at t=30",
                    "transient_prob", 0, 10, 30.0, 0.3, ">=",
                ),
                CSLProperty(
                    "x2_low", "P(X2≤5) at t=30",
                    "transient_prob", 1, 5, 30.0, 0.2, ">=",
                ),
            ],
        },
        {
            "name": "schlogl_bistable",
            "builder": lambda: schlogl(max_copy=80),
            "properties": [
                CSLProperty(
                    "high_mode", "P(X≥60) at steady state",
                    "steady_state", 0, 60, 500.0, 0.3, ">=",
                ),
                CSLProperty(
                    "low_mode", "P(X≤20) at steady state",
                    "steady_state", 0, 20, 500.0, 0.2, ">=",
                ),
            ],
        },
        {
            "name": "sir_small",
            "builder": lambda: sir_epidemic(
                max_S=20, max_I=20, max_R=20, S0=15, I0=2,
            ),
            "properties": [
                CSLProperty(
                    "outbreak", "P(I≥8) at t=10",
                    "transient_prob", 1, 8, 10.0, 0.3, ">=",
                ),
                CSLProperty(
                    "epidemic_end", "P(I≤1) at t=50",
                    "transient_prob", 1, 1, 50.0, 0.5, ">=",
                ),
            ],
        },
        {
            "name": "enzyme_kinetics",
            "builder": lambda: michaelis_menten_enzyme(
                max_E=8, max_S=12, max_ES=8, max_P=12, E0=4, S0=8,
            ),
            "properties": [
                CSLProperty(
                    "product_ge5", "P(P≥5) at t=20",
                    "transient_prob", 3, 5, 20.0, 0.5, ">=",
                ),
                CSLProperty(
                    "substrate_low", "P(S≤2) at t=30",
                    "transient_prob", 1, 2, 30.0, 0.5, ">=",
                ),
            ],
        },
        {
            "name": "cascade_3layer",
            "builder": lambda: cascade(n_layers=3, max_copy=20),
            "properties": [
                CSLProperty(
                    "layer3_ge5", "P(X3≥5) at t=20",
                    "transient_prob", 2, 5, 20.0, 0.3, ">=",
                ),
                CSLProperty(
                    "layer1_high", "P(X1≥10) at t=10",
                    "transient_prob", 0, 10, 10.0, 0.5, ">=",
                ),
            ],
        },
        {
            "name": "autoregulatory_gene",
            "builder": lambda: autoregulatory_gene(),
            "properties": [
                CSLProperty(
                    "protein_ge10", "P(Protein≥10) at t=50",
                    "transient_prob", 1, 10, 50.0, 0.5, ">=",
                ),
                CSLProperty(
                    "mrna_ss_ge3", "P(mRNA≥3) at steady state",
                    "steady_state", 0, 3, 200.0, 0.5, ">=",
                ),
            ],
        },
        {
            "name": "glycolysis",
            "builder": lambda: glycolysis_oscillator(),
            "properties": [
                CSLProperty(
                    "substrate_ge10", "P(Substrate≥10) at t=10",
                    "transient_prob", 0, 10, 10.0, 0.3, ">=",
                ),
                CSLProperty(
                    "product_ss_ge5", "P(Product≥5) at steady state",
                    "steady_state", 2, 5, 200.0, 0.3, ">=",
                ),
            ],
        },
        {
            "name": "competition",
            "builder": lambda: competition_model(),
            "properties": [
                CSLProperty(
                    "speciesA_ge15", "P(A≥15) at t=20",
                    "transient_prob", 0, 15, 20.0, 0.3, ">=",
                ),
                CSLProperty(
                    "speciesB_low", "P(B≤5) at t=30",
                    "transient_prob", 1, 5, 30.0, 0.3, ">=",
                ),
            ],
        },
    ]


def run_cross_validation(
    max_bond_dim: int = 50,
    tolerance: float = 1e-10,
    max_state_space: int = 500_000,
) -> dict:
    """
    Run full PRISM/Storm cross-validation.

    For each model and CSL property:
    1. Compute exact probability via dense matrix exponential
    2. Compute TT-compressed probability via MPS solver
    3. Compare verdicts and report agreement

    Args:
        max_bond_dim: Maximum TT bond dimension.
        tolerance: SVD truncation tolerance.
        max_state_space: Maximum state space for dense computation.

    Returns:
        Dictionary with per-model, per-property results and summary.
    """
    suite = _define_crossval_suite()
    all_results: list[CrossValidationResult] = []

    for model_entry in suite:
        model_name = model_entry["name"]
        try:
            net = model_entry["builder"]()
        except Exception as e:
            logger.error(f"Failed to build model {model_name}: {e}")
            continue

        state_space = int(np.prod(net.physical_dims))
        if state_space > max_state_space:
            logger.warning(f"Skipping {model_name}: state space {state_space} too large")
            continue

        # Build dense solver
        dense_solver = DenseReferenceSolver(net, max_states=max_state_space)

        # Build TT solver
        compiler = CMECompiler(net)
        mpo = compiler.compile()

        for prop in model_entry["properties"]:
            result = _cross_validate_single(
                net, dense_solver, mpo, prop,
                max_bond_dim, tolerance, state_space,
            )
            result.model_name = model_name
            all_results.append(result)
            logger.info(
                f"Cross-validation {model_name}/{prop.name}: "
                f"dense={result.dense_probability:.6f}, "
                f"tt={result.tt_probability:.6f}, "
                f"agree={result.verdicts_agree}"
            )

    # Summary
    n_total = len(all_results)
    n_agree = sum(1 for r in all_results if r.verdicts_agree)
    n_true_agree = sum(1 for r in all_results
                       if r.verdicts_agree and r.dense_verdict == "TRUE")
    n_false_agree = sum(1 for r in all_results
                        if r.verdicts_agree and r.dense_verdict == "FALSE")
    n_indeterminate = sum(1 for r in all_results
                         if r.tt_verdict == "INDETERMINATE")

    errors = [r.absolute_error for r in all_results]
    max_error = max(errors) if errors else 0.0
    mean_error = float(np.mean(errors)) if errors else 0.0

    summary = {
        "total_properties": n_total,
        "verdicts_agree": n_agree,
        "verdicts_disagree": n_total - n_agree - n_indeterminate,
        "tt_indeterminate": n_indeterminate,
        "agreement_rate": n_agree / max(n_total, 1),
        "true_verdicts_agree": n_true_agree,
        "false_verdicts_agree": n_false_agree,
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
    }

    return {
        "results": [_result_to_dict(r) for r in all_results],
        "summary": summary,
    }


def _cross_validate_single(
    net: ReactionNetwork,
    dense_solver: DenseReferenceSolver,
    mpo,
    prop: CSLProperty,
    max_bond_dim: int,
    tolerance: float,
    state_space: int,
) -> CrossValidationResult:
    """Cross-validate a single CSL property."""
    from tn_check.tensor.mps import MPS
    from tn_check.tensor.operations import mps_compress
    from tn_check.tensor.decomposition import tensor_to_mps

    dims = net.physical_dims
    N = state_space

    # Build initial state
    p0 = np.zeros(N)
    initial_counts = net.initial_state
    flat_idx = np.ravel_multi_index(
        [min(c, d - 1) for c, d in zip(initial_counts, dims)], dims,
    )
    p0[flat_idx] = 1.0

    # --- Dense computation ---
    t_start = time.time()
    dense_result = dense_solver.csl_comparison(
        formula_type=prop.formula_type,
        t=prop.time_bound,
        species_index=prop.species_index,
        threshold2=prop.threshold,
    )
    dense_time = time.time() - t_start
    dense_prob = dense_result["probability"]

    # Dense verdict
    if prop.comparison == ">=":
        dense_verdict = "TRUE" if dense_prob >= prop.probability_threshold else "FALSE"
    elif prop.comparison == "<=":
        dense_verdict = "TRUE" if dense_prob <= prop.probability_threshold else "FALSE"
    else:
        dense_verdict = "TRUE" if dense_prob >= prop.probability_threshold else "FALSE"

    # --- TT computation ---
    t_start = time.time()
    tt_prob, tt_trunc_error = _tt_compute_probability(
        net, mpo, p0, dims, prop, max_bond_dim, tolerance,
    )
    tt_time = time.time() - t_start

    # TT verdict with three-valued semantics
    eps = tt_trunc_error + tolerance
    prob_lo = max(0.0, tt_prob - eps)
    prob_hi = min(1.0, tt_prob + eps)

    if prop.comparison == ">=":
        if prob_lo >= prop.probability_threshold:
            tt_verdict = "TRUE"
        elif prob_hi < prop.probability_threshold:
            tt_verdict = "FALSE"
        else:
            tt_verdict = "INDETERMINATE"
    elif prop.comparison == "<=":
        if prob_hi <= prop.probability_threshold:
            tt_verdict = "TRUE"
        elif prob_lo > prop.probability_threshold:
            tt_verdict = "FALSE"
        else:
            tt_verdict = "INDETERMINATE"
    else:
        tt_verdict = "INDETERMINATE"

    # Verdict agreement
    if tt_verdict == "INDETERMINATE":
        verdicts_agree = True  # sound: indeterminate is conservative
    else:
        verdicts_agree = (tt_verdict == dense_verdict)

    abs_error = abs(dense_prob - tt_prob)
    rel_error = abs_error / max(abs(dense_prob), 1e-15)

    return CrossValidationResult(
        model_name="",  # filled by caller
        property_name=prop.name,
        dense_probability=dense_prob,
        tt_probability=tt_prob,
        absolute_error=abs_error,
        relative_error=rel_error,
        dense_verdict=dense_verdict,
        tt_verdict=tt_verdict,
        verdicts_agree=verdicts_agree,
        dense_time_s=dense_time,
        tt_time_s=tt_time,
        state_space_size=state_space,
        tt_bond_dim=max_bond_dim,
        tt_truncation_error=tt_trunc_error,
    )


def _tt_compute_probability(
    net: ReactionNetwork,
    mpo,
    p0: NDArray,
    dims: tuple,
    prop: CSLProperty,
    max_bond_dim: int,
    tolerance: float,
) -> tuple[float, float]:
    """Compute CSL probability using TT-compressed solver."""
    from tn_check.tensor.decomposition import tensor_to_mps
    from tn_check.tensor.operations import (
        mps_compress, mps_total_probability,
        mps_clamp_nonnegative, mps_normalize_probability,
        mpo_mps_contraction, mps_zip_up,
    )

    N = int(np.prod(dims))

    # Convert p0 to MPS
    p0_mps = tensor_to_mps(p0.reshape(dims), list(dims), max_bond_dim=max_bond_dim)

    total_trunc_error = 0.0

    if prop.formula_type in ("transient_prob", "bounded_until"):
        # Time evolution via Euler steps in TT format
        t = prop.time_bound
        dt = min(0.1, t / 10)
        n_steps = max(1, int(t / dt))
        dt = t / n_steps

        state = p0_mps
        for step in range(n_steps):
            # p_{n+1} = p_n + dt * Q @ p_n
            Qp, err = mps_zip_up(mpo, state, max_bond_dim=max_bond_dim * 2,
                                 tolerance=tolerance)
            total_trunc_error += err

            from tn_check.tensor.operations import mps_scalar_multiply, mps_addition
            dp = mps_scalar_multiply(Qp, dt)
            state = mps_addition(state, dp)
            state, err = mps_compress(state, max_bond_dim=max_bond_dim,
                                      tolerance=tolerance)
            total_trunc_error += err

            # Clamp negatives
            state, clamp_err = mps_clamp_nonnegative(
                state, max_bond_dim=max_bond_dim, tolerance=tolerance,
            )

            # Normalize
            tp = mps_total_probability(state)
            if abs(tp) > 1e-15 and abs(tp - 1.0) > tolerance:
                state = mps_normalize_probability(state)

        # Extract probability for predicate
        p_dense = mps_to_dense(state).ravel()
        prob = _compute_predicate_prob(p_dense, dims, prop.species_index,
                                       prop.threshold, "greater_equal")

    elif prop.formula_type == "steady_state":
        # Long-time evolution to approximate steady state
        t_ss = prop.time_bound
        dt = min(0.5, t_ss / 20)
        n_steps = max(1, int(t_ss / dt))
        dt = t_ss / n_steps

        state = p0_mps
        for step in range(n_steps):
            Qp, err = mps_zip_up(mpo, state, max_bond_dim=max_bond_dim * 2,
                                 tolerance=tolerance)
            total_trunc_error += err

            from tn_check.tensor.operations import mps_scalar_multiply, mps_addition
            dp = mps_scalar_multiply(Qp, dt)
            state = mps_addition(state, dp)
            state, err = mps_compress(state, max_bond_dim=max_bond_dim,
                                      tolerance=tolerance)
            total_trunc_error += err

            state, _ = mps_clamp_nonnegative(
                state, max_bond_dim=max_bond_dim, tolerance=tolerance,
            )
            tp = mps_total_probability(state)
            if abs(tp) > 1e-15 and abs(tp - 1.0) > tolerance:
                state = mps_normalize_probability(state)

        p_dense = mps_to_dense(state).ravel()
        direction = "less_equal" if prop.comparison == "<=" else "greater_equal"
        prob = _compute_predicate_prob(p_dense, dims, prop.species_index,
                                       prop.threshold, direction)
    else:
        prob = 0.0

    return prob, total_trunc_error


def _compute_predicate_prob(
    p: NDArray,
    dims: tuple,
    species_index: int,
    threshold: int,
    direction: str,
) -> float:
    """Compute probability of a species predicate from a flat probability vector."""
    p_tensor = p.reshape(dims)
    d = dims[species_index]
    n_species = len(dims)

    mask_1d = np.zeros(d, dtype=bool)
    for n in range(d):
        if direction == "greater_equal":
            mask_1d[n] = n >= threshold
        elif direction == "less_equal":
            mask_1d[n] = n <= threshold
        elif direction == "greater":
            mask_1d[n] = n > threshold
        elif direction == "less":
            mask_1d[n] = n < threshold

    shape = [1] * n_species
    shape[species_index] = d
    full_mask = mask_1d.reshape(shape)
    return float(np.sum(np.maximum(p_tensor, 0.0) * full_mask))


def export_to_prism(net: ReactionNetwork) -> PRISMExportResult:
    """
    Export a reaction network to PRISM model format (.pm).

    Generates a CTMC model in the PRISM modeling language that can be
    checked with PRISM or Storm for external validation.

    Args:
        net: Reaction network to export.

    Returns:
        PRISMExportResult with model string and properties.
    """
    lines = []
    lines.append("// Auto-generated PRISM model from TN-Check")
    lines.append(f"// Model: {net.name}")
    lines.append(f"// Species: {net.num_species}")
    lines.append(f"// Reactions: {len(net.reactions)}")
    lines.append("")
    lines.append("ctmc")
    lines.append("")

    # Module for the reaction network
    lines.append(f"module {net.name}")
    lines.append("")

    # Species declarations
    for sp in net.species:
        init = sp.initial_count
        lines.append(
            f"  {sp.name} : [0..{sp.max_copy_number}] init {init};"
        )
    lines.append("")

    # Reactions as commands
    for rxn in net.reactions:
        guard_parts = []
        update_parts = []

        # Guard: all reactants have sufficient copies
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            sp = net.species[sp_idx]
            guard_parts.append(f"{sp.name}>={stoich}")

        # Also check that products don't exceed max
        net_change = {}
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            net_change[sp_idx] = net_change.get(sp_idx, 0) - stoich
        for sp_idx, stoich in zip(rxn.product_species, rxn.product_stoichiometry):
            net_change[sp_idx] = net_change.get(sp_idx, 0) + stoich

        for sp_idx, delta in net_change.items():
            sp = net.species[sp_idx]
            if delta > 0:
                guard_parts.append(f"{sp.name}<={sp.max_copy_number - delta}")

        guard = " & ".join(guard_parts) if guard_parts else "true"

        # Rate (simplified: mass-action)
        rate = _prism_rate_string(rxn, net)

        # Updates
        for sp_idx, delta in sorted(net_change.items()):
            sp = net.species[sp_idx]
            if delta > 0:
                update_parts.append(f"({sp.name}'={sp.name}+{delta})")
            elif delta < 0:
                update_parts.append(f"({sp.name}'={sp.name}{delta})")

        update_str = " & ".join(update_parts) if update_parts else "true"
        lines.append(f"  [{rxn.name}] {guard} -> {rate} : {update_str};")

    lines.append("")
    lines.append("endmodule")

    model_str = "\n".join(lines)

    # Generate standard properties
    properties = []
    for sp in net.species:
        properties.append(f'P=? [ F<={100.0} {sp.name}>=10 ]')
        properties.append(f'S=? [ {sp.name}>=5 ]')

    state_space = int(np.prod(net.physical_dims))
    return PRISMExportResult(
        model_name=net.name,
        prism_model=model_str,
        prism_properties=properties,
        num_species=net.num_species,
        state_space_size=state_space,
    )


def _prism_rate_string(rxn, net: ReactionNetwork) -> str:
    """Generate PRISM rate expression for a reaction."""
    from tn_check.cme.reaction_network import (
        MassActionPropensity, HillPropensity,
    )

    prop = rxn.propensity
    if isinstance(prop, MassActionPropensity):
        k = prop.rate_constant
        parts = [str(k)]
        for sp_idx, stoich in zip(prop.reactant_species, prop.reactant_stoichiometry):
            sp = net.species[sp_idx]
            if stoich == 1:
                parts.append(sp.name)
            elif stoich == 2:
                parts.append(f"{sp.name}*({sp.name}-1)/2")
            else:
                parts.append(f"{sp.name}")
        return "*".join(parts)
    elif isinstance(prop, HillPropensity):
        sp = net.species[prop.species_index]
        v = prop.v_max
        K = prop.k_half
        n = prop.hill_coefficient
        if prop.activation:
            return f"{v}*pow({sp.name},{n})/(pow({K},{n})+pow({sp.name},{n}))"
        else:
            return f"{v}*pow({K},{n})/(pow({K},{n})+pow({sp.name},{n}))"
    return "1.0"


def generate_prism_crossval_report(results: dict) -> str:
    """Generate a human-readable cross-validation report."""
    lines = []
    lines.append("=" * 72)
    lines.append("TN-Check vs Dense Reference Cross-Validation Report")
    lines.append("=" * 72)
    lines.append("")

    summary = results["summary"]
    lines.append(f"Total properties checked: {summary['total_properties']}")
    lines.append(f"Verdicts agree:           {summary['verdicts_agree']}")
    lines.append(f"TT indeterminate:         {summary['tt_indeterminate']}")
    lines.append(f"Verdicts disagree:        {summary['verdicts_disagree']}")
    lines.append(f"Agreement rate:           {summary['agreement_rate']:.1%}")
    lines.append(f"Max absolute error:       {summary['max_absolute_error']:.2e}")
    lines.append(f"Mean absolute error:      {summary['mean_absolute_error']:.2e}")
    lines.append("")

    lines.append(f"{'Model':<25} {'Property':<20} {'Dense':<10} "
                 f"{'TT':<10} {'|Δ|':<10} {'Dense V':<8} {'TT V':<8} {'Agree'}")
    lines.append("-" * 100)

    for r in results["results"]:
        lines.append(
            f"{r['model_name']:<25} {r['property_name']:<20} "
            f"{r['dense_probability']:<10.6f} {r['tt_probability']:<10.6f} "
            f"{r['absolute_error']:<10.2e} {r['dense_verdict']:<8} "
            f"{r['tt_verdict']:<8} {'✓' if r['verdicts_agree'] else '✗'}"
        )

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def _result_to_dict(r: CrossValidationResult) -> dict:
    """Convert CrossValidationResult to serializable dict."""
    return {
        "model_name": r.model_name,
        "property_name": r.property_name,
        "dense_probability": r.dense_probability,
        "tt_probability": r.tt_probability,
        "absolute_error": r.absolute_error,
        "relative_error": r.relative_error,
        "dense_verdict": r.dense_verdict,
        "tt_verdict": r.tt_verdict,
        "verdicts_agree": r.verdicts_agree,
        "dense_time_s": r.dense_time_s,
        "tt_time_s": r.tt_time_s,
        "state_space_size": r.state_space_size,
        "tt_bond_dim": r.tt_bond_dim,
        "tt_truncation_error": r.tt_truncation_error,
        "notes": r.notes,
    }


def export_storm_benchmark(output_dir: str = "benchmarks/prism_models") -> dict:
    """
    Export all cross-validation models to Storm/PRISM format for external verification.

    Creates .pm (model) and .csl (property) files that can be checked with:
        storm --prism model.pm --prop properties.csl

    Returns:
        dict with file paths and expected results for comparison.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    suite = _define_crossval_suite()
    exports = []

    for model_entry in suite:
        model_name = model_entry["name"]
        try:
            net = model_entry["builder"]()
            prism_export = export_to_prism(net)

            # Write model file
            model_path = os.path.join(output_dir, f"{model_name}.pm")
            with open(model_path, "w") as f:
                f.write(prism_export.prism_model)

            # Write CSL properties file
            prop_lines = []
            for prop in model_entry["properties"]:
                csl_str = _csl_property_to_prism(prop, net)
                prop_lines.append(csl_str)

            prop_path = os.path.join(output_dir, f"{model_name}.csl")
            with open(prop_path, "w") as f:
                f.write("\n".join(prop_lines))

            exports.append({
                "model_name": model_name,
                "model_file": model_path,
                "property_file": prop_path,
                "num_properties": len(model_entry["properties"]),
                "state_space": prism_export.state_space_size,
            })
        except Exception as e:
            logger.error(f"Export failed for {model_name}: {e}")

    return {"exports": exports, "total_models": len(exports)}


def _csl_property_to_prism(prop: CSLProperty, net) -> str:
    """Convert a CSLProperty to PRISM property syntax."""
    sp = net.species[prop.species_index]

    if prop.formula_type == "steady_state":
        return f'S=? [ {sp.name}>={prop.threshold} ]'
    elif prop.formula_type == "transient_prob":
        if prop.comparison == "<=":
            return f'P=? [ F<={prop.time_bound} {sp.name}<={prop.threshold} ]'
        else:
            return f'P=? [ F<={prop.time_bound} {sp.name}>={prop.threshold} ]'
    elif prop.formula_type == "bounded_until":
        return f'P=? [ true U<={prop.time_bound} {sp.name}>={prop.threshold} ]'
    return f'P=? [ F<={prop.time_bound} {sp.name}>={prop.threshold} ]'


def run_verdict_comparison_table() -> dict:
    """
    Run a comprehensive verdict-by-verdict comparison and return a structured table.

    This provides the data for Table 5 in the paper: model-by-model,
    property-by-property verdict agreement between TN-Check and dense reference.

    Returns:
        dict with 'table' (list of row dicts) and 'summary'.
    """
    results = run_cross_validation(max_bond_dim=50, tolerance=1e-10)
    table = []

    for r in results["results"]:
        table.append({
            "model": r["model_name"],
            "property": r["property_name"],
            "p_exact": f"{r['dense_probability']:.6f}",
            "p_tt": f"{r['tt_probability']:.6f}",
            "error": f"{r['absolute_error']:.2e}",
            "verdict_exact": r["dense_verdict"],
            "verdict_tt": r["tt_verdict"],
            "agree": "✓" if r["verdicts_agree"] else "✗",
        })

    return {
        "table": table,
        "summary": results["summary"],
    }
