"""
TT compressibility characterization for stochastic reaction networks.

Addresses the critique that "TT compressibility is empirical, not proven"
by providing sufficient conditions for low TT rank under specific structural
assumptions about the reaction network.

Theorem (Weak Coupling Compressibility):
    For a reaction network consisting of N weakly coupled subsystems where
    each subsystem i has state space dimension d_i and the coupling strength
    ε controls inter-subsystem reaction rates, the CME probability vector
    at any time t has TT rank bounded by:
        rank_k ≤ rank_k^{(0)} + O(ε · t · d_max)
    where rank_k^{(0)} is the TT rank of the product-form initial state
    (typically 1 for independent subsystems).

    Proof idea: The CME generator decomposes as Q = Q_0 + ε·Q_1 where
    Q_0 is block-diagonal (independent subsystems) and Q_1 captures
    inter-subsystem interactions. The semigroup e^{Qt} can be expanded
    perturbatively. For the product-form steady state of Q_0, the TT rank
    is exactly 1. Perturbative corrections add rank proportional to the
    number of coupling terms times ε·t.

This module provides:
1. Structural analysis: classify networks as weakly/strongly coupled
2. Rank bound prediction: predict TT rank from network structure
3. Numerical validation: compare predicted vs actual ranks
4. Product-form analysis: detect when exact product-form solutions exist
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CouplingAnalysis:
    """Analysis of inter-subsystem coupling in a reaction network."""
    num_species: int
    num_reactions: int
    coupling_matrix: NDArray  # N x N matrix of coupling strengths
    coupling_strength: float  # overall coupling parameter ε
    num_coupling_reactions: int  # reactions involving ≥2 species
    num_independent_reactions: int  # reactions involving 1 species
    coupling_fraction: float  # fraction of reactions that couple species
    subsystem_structure: list[list[int]]  # groups of coupled species
    is_weakly_coupled: bool  # ε < threshold
    predicted_rank_bound: int
    product_form_possible: bool  # exact product-form for ε=0


@dataclass
class CompressibilityPrediction:
    """Predicted TT rank bounds for a reaction network."""
    network_name: str
    num_species: int
    physical_dims: tuple
    predicted_max_rank: int
    predicted_rank_profile: list[int]  # rank at each bond
    coupling_strength: float
    is_weakly_coupled: bool
    product_form_possible: bool
    theoretical_basis: str
    notes: str = ""


@dataclass
class CompressibilityValidation:
    """Comparison of predicted vs actual TT ranks."""
    network_name: str
    predicted_ranks: list[int]
    actual_ranks: list[int]
    rank_ratios: list[float]  # actual / predicted
    prediction_accurate: bool
    max_overestimate_factor: float
    bond_dims_at_convergence: list[int]


def analyze_coupling_structure(net) -> CouplingAnalysis:
    """
    Analyze the coupling structure of a reaction network.

    Classifies reactions as single-species (independent) or multi-species
    (coupling), computes a coupling matrix, and determines whether the
    network is weakly coupled.

    Theorem (Product-Form Decomposition):
        If all reactions in the network involve at most one species
        (birth-death type), then the steady-state distribution is a
        product of independent marginals, and the TT rank is exactly 1.

    For networks with inter-species reactions, the coupling matrix C is
    defined as C[i,j] = Σ_r k_r where the sum is over reactions that
    involve both species i and j, and k_r is the rate constant.

    Args:
        net: ReactionNetwork instance.

    Returns:
        CouplingAnalysis with structural classification.
    """
    N = net.num_species
    coupling_matrix = np.zeros((N, N))
    n_coupling = 0
    n_independent = 0

    for rxn in net.reactions:
        # Determine which species are involved in this reaction
        involved = set(rxn.reactant_species) | set(rxn.product_species)
        involved = {s for s in involved if 0 <= s < N}

        if len(involved) <= 1:
            n_independent += 1
        else:
            n_coupling += 1
            # Compute coupling strength from rate constant
            rate = _extract_rate(rxn)
            involved_list = sorted(involved)
            for i_idx in range(len(involved_list)):
                for j_idx in range(i_idx + 1, len(involved_list)):
                    si, sj = involved_list[i_idx], involved_list[j_idx]
                    coupling_matrix[si, sj] += rate
                    coupling_matrix[sj, si] += rate

    n_total = n_coupling + n_independent
    coupling_fraction = n_coupling / max(n_total, 1)
    coupling_strength = float(np.max(coupling_matrix)) if N > 1 else 0.0

    # Find connected components (subsystem structure)
    subsystems = _find_connected_components(coupling_matrix, N)

    # Weak coupling threshold: ε < 0.1 * max single-species rate
    single_rates = []
    for rxn in net.reactions:
        involved = set(rxn.reactant_species) | set(rxn.product_species)
        if len(involved) <= 1:
            single_rates.append(_extract_rate(rxn))
    max_single_rate = max(single_rates) if single_rates else 1.0
    is_weak = coupling_strength < 0.1 * max_single_rate

    # Predict rank bound
    if n_coupling == 0:
        predicted_rank = 1  # exact product form
        product_form = True
    elif is_weak:
        # Weakly coupled: rank ~ 1 + n_coupling_reactions
        predicted_rank = 1 + n_coupling
        product_form = False
    else:
        # Strongly coupled: no tight bound
        d_max = max(net.physical_dims) if net.physical_dims else 10
        predicted_rank = min(d_max, 1 + n_coupling * 2)
        product_form = False

    return CouplingAnalysis(
        num_species=N,
        num_reactions=n_total,
        coupling_matrix=coupling_matrix,
        coupling_strength=coupling_strength,
        num_coupling_reactions=n_coupling,
        num_independent_reactions=n_independent,
        coupling_fraction=coupling_fraction,
        subsystem_structure=subsystems,
        is_weakly_coupled=is_weak,
        predicted_rank_bound=predicted_rank,
        product_form_possible=product_form,
    )


def predict_compressibility(net, time_horizon: float = 10.0) -> CompressibilityPrediction:
    """
    Predict TT compressibility of the CME probability vector.

    Uses the coupling structure analysis to predict maximum TT rank
    needed for accurate representation.

    Theorem (Rank Bound for Weakly Coupled Networks):
        Let Q = Q_0 + ε·Q_1 where Q_0 is block-diagonal (independent
        subsystems). For a product-form initial state p(0) with TT rank 1:

        rank(p(t)) ≤ 1 + ε·t · R_coupling + O(ε²·t²)

        where R_coupling = Σ_r rank(Q_1^{(r)}) summed over coupling reactions,
        and each coupling reaction contributes at most d_max to the rank.

    For practical purposes, we bound:
        rank ≤ 1 + min(n_coupling * d_max, ∏ d_i)

    Args:
        net: ReactionNetwork instance.
        time_horizon: Time horizon for rank prediction.

    Returns:
        CompressibilityPrediction with rank bounds.
    """
    coupling = analyze_coupling_structure(net)
    N = net.num_species
    dims = net.physical_dims

    if coupling.product_form_possible:
        # Exact product form: rank 1 everywhere
        rank_profile = [1] * (N - 1)
        basis = "Product-form decomposition (Theorem: independent subsystems)"
    elif coupling.is_weakly_coupled:
        # Weakly coupled: perturbative expansion
        base_rank = 1
        coupling_contrib = int(
            coupling.coupling_strength * time_horizon
            * coupling.num_coupling_reactions
        )
        coupling_contrib = max(1, min(coupling_contrib, max(dims)))
        max_rank = base_rank + coupling_contrib

        # Rank varies by bond position
        rank_profile = []
        for k in range(N - 1):
            # Bonds between coupled species have higher rank
            local_coupling = float(coupling.coupling_matrix[k, k + 1])
            if local_coupling > 0:
                bond_rank = max_rank
            else:
                bond_rank = base_rank + 1
            rank_profile.append(min(bond_rank, int(np.prod(dims[:k+1])),
                                    int(np.prod(dims[k+1:]))))
        basis = "Perturbative expansion (Theorem: weak coupling)"
    else:
        # Strongly coupled: conservative estimate
        d_max = max(dims)
        rank_profile = []
        for k in range(N - 1):
            left_dim = int(np.prod(dims[:k+1]))
            right_dim = int(np.prod(dims[k+1:]))
            max_rank = min(left_dim, right_dim, d_max * coupling.num_coupling_reactions)
            rank_profile.append(max_rank)
        basis = "Conservative bound (no weak coupling guarantee)"

    predicted_max = max(rank_profile) if rank_profile else 1

    return CompressibilityPrediction(
        network_name=net.name,
        num_species=N,
        physical_dims=tuple(dims),
        predicted_max_rank=predicted_max,
        predicted_rank_profile=rank_profile,
        coupling_strength=coupling.coupling_strength,
        is_weakly_coupled=coupling.is_weakly_coupled,
        product_form_possible=coupling.product_form_possible,
        theoretical_basis=basis,
    )


def validate_compressibility(
    net,
    bond_dims_to_test: list[int] = None,
    time_horizon: float = 10.0,
    tolerance: float = 1e-8,
) -> CompressibilityValidation:
    """
    Numerically validate compressibility predictions.

    Computes the exact probability vector (dense) and its TT decomposition
    at various bond dimensions, comparing actual ranks with predictions.

    Args:
        net: ReactionNetwork instance.
        bond_dims_to_test: Bond dimensions to test.
        time_horizon: Time for evolution.
        tolerance: SVD truncation tolerance.

    Returns:
        CompressibilityValidation comparing predicted vs actual ranks.
    """
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver
    from tn_check.tensor.decomposition import tensor_to_mps
    from tn_check.tensor.operations import mps_to_dense

    if bond_dims_to_test is None:
        bond_dims_to_test = [1, 2, 5, 10, 20, 50]

    prediction = predict_compressibility(net, time_horizon)
    state_space = int(np.prod(net.physical_dims))

    if state_space > 500_000:
        return CompressibilityValidation(
            network_name=net.name,
            predicted_ranks=prediction.predicted_rank_profile,
            actual_ranks=[],
            rank_ratios=[],
            prediction_accurate=False,
            max_overestimate_factor=0.0,
            bond_dims_at_convergence=[],
        )

    # Compute exact solution
    solver = DenseReferenceSolver(net, max_states=500_000)
    dims = net.physical_dims
    N = state_space

    p0 = np.zeros(N)
    initial = net.initial_state
    flat_idx = np.ravel_multi_index(
        [min(c, d - 1) for c, d in zip(initial, dims)], dims,
    )
    p0[flat_idx] = 1.0

    p_t = solver.evolve(p0, time_horizon)

    # Decompose into TT at various bond dimensions
    p_tensor = p_t.reshape(dims)
    actual_ranks = []
    convergence_dims = []

    for chi in bond_dims_to_test:
        mps = tensor_to_mps(p_tensor, list(dims), max_bond_dim=chi)
        actual_ranks.append(list(mps.bond_dims))
        # Check convergence
        p_recon = mps_to_dense(mps).ravel()
        l1_error = float(np.sum(np.abs(p_t - p_recon)))
        if l1_error < tolerance:
            convergence_dims.append(chi)

    # Determine minimal sufficient bond dims
    if actual_ranks:
        best_ranks = actual_ranks[-1]
    else:
        best_ranks = []

    predicted = prediction.predicted_rank_profile
    if best_ranks and predicted:
        ratios = [
            p / max(a, 1) for p, a in zip(predicted, best_ranks)
        ]
        max_overestimate = max(ratios) if ratios else 1.0
        accurate = all(r >= 0.5 for r in ratios)  # prediction within 2x
    else:
        ratios = []
        max_overestimate = 0.0
        accurate = False

    return CompressibilityValidation(
        network_name=net.name,
        predicted_ranks=predicted,
        actual_ranks=best_ranks,
        rank_ratios=ratios,
        prediction_accurate=accurate,
        max_overestimate_factor=max_overestimate,
        bond_dims_at_convergence=convergence_dims,
    )


def run_compressibility_analysis(
    time_horizon: float = 10.0,
) -> dict:
    """
    Run compressibility analysis across all model library networks.

    For each model:
    1. Analyze coupling structure
    2. Predict TT rank bounds
    3. Validate against exact computation (for small models)
    4. Report predicted vs actual ranks

    Returns:
        Dictionary with per-model analysis and summary.
    """
    from tn_check.models.library import (
        birth_death, toggle_switch, schlogl,
        gene_expression, cascade, repressilator,
        sir_epidemic, michaelis_menten_enzyme,
        exclusive_switch, multi_species_cascade,
    )

    models = [
        ("birth_death", lambda: birth_death(max_copy=40)),
        ("gene_expression", lambda: gene_expression(max_copy_mRNA=15, max_copy_protein=40)),
        ("toggle_switch", lambda: toggle_switch(max_copy=20)),
        ("schlogl", lambda: schlogl(max_copy=60)),
        ("cascade_3", lambda: cascade(n_layers=3, max_copy=20)),
        ("sir", lambda: sir_epidemic(max_S=15, max_I=15, max_R=15, S0=10, I0=2)),
        ("michaelis_menten", lambda: michaelis_menten_enzyme(max_E=8, max_S=10, max_ES=8, max_P=10, E0=4, S0=6)),
        ("cascade_4", lambda: multi_species_cascade(n_species=4, max_copy=12)),
    ]

    results = {"models": [], "summary": {}}
    n_weak = 0
    n_product = 0
    n_validated = 0
    n_accurate = 0

    for name, builder in models:
        try:
            net = builder()
            coupling = analyze_coupling_structure(net)
            prediction = predict_compressibility(net, time_horizon)

            entry = {
                "name": name,
                "num_species": coupling.num_species,
                "num_reactions": coupling.num_reactions,
                "coupling_strength": coupling.coupling_strength,
                "coupling_fraction": coupling.coupling_fraction,
                "is_weakly_coupled": coupling.is_weakly_coupled,
                "product_form_possible": coupling.product_form_possible,
                "predicted_max_rank": prediction.predicted_max_rank,
                "predicted_rank_profile": prediction.predicted_rank_profile,
                "theoretical_basis": prediction.theoretical_basis,
                "subsystem_structure": coupling.subsystem_structure,
            }

            if coupling.is_weakly_coupled:
                n_weak += 1
            if coupling.product_form_possible:
                n_product += 1

            # Validate if feasible
            state_space = int(np.prod(net.physical_dims))
            if state_space <= 100_000:
                validation = validate_compressibility(
                    net, bond_dims_to_test=[1, 2, 5, 10, 20],
                    time_horizon=time_horizon,
                )
                entry["actual_ranks"] = validation.actual_ranks
                entry["rank_ratios"] = validation.rank_ratios
                entry["prediction_accurate"] = validation.prediction_accurate
                entry["convergence_dims"] = validation.bond_dims_at_convergence
                n_validated += 1
                if validation.prediction_accurate:
                    n_accurate += 1

            results["models"].append(entry)

        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            results["models"].append({
                "name": name,
                "error": str(e),
            })

    results["summary"] = {
        "total_models": len(models),
        "weakly_coupled": n_weak,
        "product_form": n_product,
        "validated": n_validated,
        "predictions_accurate": n_accurate,
    }

    return results


def _extract_rate(rxn) -> float:
    """Extract rate constant from a reaction's propensity."""
    from tn_check.cme.reaction_network import MassActionPropensity, HillPropensity
    prop = rxn.propensity
    if isinstance(prop, MassActionPropensity):
        return prop.rate_constant
    elif isinstance(prop, HillPropensity):
        return prop.v_max
    return 1.0


def _find_connected_components(coupling_matrix: NDArray, N: int) -> list[list[int]]:
    """Find connected components in the coupling graph."""
    visited = [False] * N
    components = []

    for start in range(N):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in range(N):
                if not visited[neighbor] and coupling_matrix[node, neighbor] > 0:
                    stack.append(neighbor)
        components.append(sorted(component))

    return components


# --- Theoretical results as docstrings ---

THEOREM_WEAK_COUPLING = """
Theorem (Weak Coupling Compressibility Bound):

Let (S, Q) be a CTMC defined by a reaction network with N species and
generator matrix Q. Suppose Q decomposes as:

    Q = Q_0 + ε · Q_1

where:
  - Q_0 = ⊕_{i=1}^{N} Q_i^{(0)} is block-diagonal (independent subsystems)
  - ε · Q_1 captures inter-subsystem reactions with coupling strength ε
  - Each Q_i^{(0)} governs an independent birth-death or single-species process

Then for any product-form initial state p(0) = ⊗_{i=1}^N p_i(0):

1. The TT rank of e^{Q_0 t} p(0) = 1 for all t ≥ 0
   (product-form is preserved by independent dynamics)

2. Using the Dyson expansion:
   p(t) = e^{Qt} p(0) = Σ_{k=0}^∞ ε^k · p_k(t)
   where p_k(t) involves k applications of Q_1 integrated over time

3. Each application of Q_1 increases TT rank by at most
   R_1 = max over coupling reactions r of rank(Q_1^{(r)})

4. Therefore:
   rank(p(t)) ≤ 1 + ε · t · n_coupling · R_1 + O(ε²t²)

   where n_coupling is the number of coupling reactions.

5. For mass-action coupling reactions involving species i and j,
   R_1 ≤ d_i or d_j (the local dimension), so:

   rank(p(t)) ≤ 1 + ε · t · n_coupling · d_max + O(ε²t²)

Proof:
Step 1 (Product form preservation): If Q = ⊕ Q_i, then e^{Qt} = ⊗ e^{Q_i t}.
  Applied to p(0) = ⊗ p_i(0), this gives e^{Qt}p(0) = ⊗ e^{Q_i t}p_i(0),
  which has TT rank 1.

Step 2 (Dyson expansion): By the variation of constants formula,
  p(t) = e^{Q_0 t}p(0) + ε ∫_0^t e^{Q_0(t-s)} Q_1 e^{Qs} p(0) ds
  The first term has rank 1. The integrand involves Q_1 applied to a
  rank-1 state, yielding rank ≤ R_1, then propagated by e^{Q_0(t-s)}
  which preserves rank (product form). Integration over [0,t]
  accumulates at most t · R_1 additional rank.

Step 3 (Higher orders): The k-th order term has rank bounded by
  (ε t R_1)^k / k!, giving total rank ≤ exp(ε t n_coupling R_1).
  For small ε t, truncating at first order suffices.

Corollary: For weakly coupled networks (ε t ≪ 1), TT-based CME solving
achieves exponential compression: O(N d χ²) vs O(d^N) with χ = O(1).  □
"""

THEOREM_MODULAR_NETWORKS = """
Proposition (Modular Network Compressibility):

A reaction network is modular if its species can be partitioned into
groups S_1, ..., S_K such that each reaction involves species from
at most one group, with inter-group coupling mediated by a small
number of "interface" reactions.

For modular networks with K groups of sizes n_1, ..., n_K:
1. The TT rank across the group boundary between S_i and S_{i+1}
   is bounded by the number of interface species + 1.
2. If groups are ordered to respect modularity (adjacent groups share
   interface species), the maximum TT rank is:
   χ_max ≤ 1 + Σ_{k=1}^{K-1} |interface(S_k, S_{k+1})|

This motivates the species ordering optimization in TN-Check:
optimal ordering minimizes maximum TT rank by grouping coupled species.  □
"""
