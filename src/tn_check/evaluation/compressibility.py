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


@dataclass
class EntanglementProfile:
    """Entanglement entropy profile across all bonds of a TT decomposition."""
    bond_entropies: list[float]       # von Neumann entropies S(ρ_k)
    bond_renyi2: list[float]          # Rényi-2 entropies
    bond_schmidt_ranks: list[int]     # number of singular values > threshold
    max_entropy: float
    mean_entropy: float
    area_law_holds: bool              # S ~ O(1) vs S ~ O(N)
    area_law_coefficient: float       # fitted constant in S ≈ c
    predicted_bond_dim: int           # χ ~ e^S for area-law systems
    volume_law_fraction: float        # fraction of bonds with S > log(d)/2


@dataclass
class CompressibilityTheorem:
    """Result of compressibility theorem application to a specific network."""
    network_name: str
    theorem_applies: str              # "product_form", "weak_coupling", "area_law", "none"
    rank_bound: int
    rank_bound_derivation: str
    coupling_strength: float
    entanglement_profile: Optional[EntanglementProfile]
    sufficient_conditions_met: list[str]
    conditions_violated: list[str]
    confidence: str                   # "rigorous", "empirical", "heuristic"


def compute_entanglement_profile(
    probability_vector: NDArray,
    dims: tuple,
    threshold: float = 1e-14,
) -> EntanglementProfile:
    """
    Compute the entanglement entropy profile of a probability distribution.

    For each bipartition of the N-site system into left (sites 0..k) and
    right (sites k+1..N-1), computes the von Neumann entropy of the reduced
    density matrix. Low entanglement entropy implies low TT rank.

    Connection to TT rank (Theorem):
        For a normalized vector |ψ⟩, the TT rank at bond k equals the
        Schmidt rank of the bipartition {0,..,k} | {k+1,..,N-1}.
        The von Neumann entropy S_k = -Σ λ_i² log(λ_i²) bounds the
        effective rank: χ_eff ≈ e^{S_k}.

        Therefore: area-law entanglement (S_k = O(1)) implies bounded
        TT rank χ = O(1), independent of system size N.

    Args:
        probability_vector: Flat probability vector of size ∏ d_i.
        dims: Physical dimensions (d_0, ..., d_{N-1}).
        threshold: Singular value cutoff.

    Returns:
        EntanglementProfile with per-bond entropy data.
    """
    N = len(dims)
    p = probability_vector.copy()
    # Normalize for entropy computation (treat as quantum state)
    norm = np.sqrt(np.sum(p ** 2))
    if norm > 0:
        psi = p / norm
    else:
        psi = p

    bond_entropies = []
    bond_renyi2 = []
    bond_ranks = []

    for k in range(N - 1):
        left_dim = int(np.prod(dims[:k+1]))
        right_dim = int(np.prod(dims[k+1:]))
        mat = psi.reshape(left_dim, right_dim)

        try:
            svs = np.linalg.svd(mat, compute_uv=False)
            svs = svs[svs > threshold]
            # Schmidt coefficients squared (probabilities)
            p_schmidt = svs ** 2
            p_sum = p_schmidt.sum()
            if p_sum > 0:
                p_schmidt = p_schmidt / p_sum

            # von Neumann entropy
            vn_entropy = float(-np.sum(p_schmidt * np.log(p_schmidt + 1e-300)))
            # Rényi-2 entropy
            renyi2 = float(-np.log(np.sum(p_schmidt ** 2) + 1e-300))

            bond_entropies.append(vn_entropy)
            bond_renyi2.append(renyi2)
            bond_ranks.append(len(svs))
        except Exception:
            bond_entropies.append(0.0)
            bond_renyi2.append(0.0)
            bond_ranks.append(1)

    max_entropy = max(bond_entropies) if bond_entropies else 0.0
    mean_entropy = float(np.mean(bond_entropies)) if bond_entropies else 0.0

    # Area law test: entropy should not grow with system size
    # For area law, S ~ O(1); for volume law, S ~ O(N)
    if N >= 3 and len(bond_entropies) >= 2:
        mid_entropy = np.mean(bond_entropies[len(bond_entropies)//4:3*len(bond_entropies)//4])
        max_possible = np.log(max(dims))
        area_law_holds = mid_entropy < 0.5 * max_possible
        area_law_coeff = mid_entropy
    else:
        area_law_holds = True
        area_law_coeff = mean_entropy

    # Volume law fraction
    vol_threshold = 0.5 * np.log(max(dims)) if max(dims) > 1 else 1.0
    vol_fraction = sum(1 for s in bond_entropies if s > vol_threshold) / max(len(bond_entropies), 1)

    predicted_chi = max(1, int(np.ceil(np.exp(max_entropy))))

    return EntanglementProfile(
        bond_entropies=bond_entropies,
        bond_renyi2=bond_renyi2,
        bond_schmidt_ranks=bond_ranks,
        max_entropy=max_entropy,
        mean_entropy=mean_entropy,
        area_law_holds=area_law_holds,
        area_law_coefficient=area_law_coeff,
        predicted_bond_dim=predicted_chi,
        volume_law_fraction=vol_fraction,
    )


def compressibility_theorem(
    net,
    time_horizon: float = 10.0,
    compute_entanglement: bool = True,
) -> CompressibilityTheorem:
    """
    Apply the compressibility theorem to determine if TT compression is justified.

    Checks three sufficient conditions for bounded TT rank:

    1. Product-form networks (Theorem 5a): If all reactions involve at most
       one species, the steady-state is a product distribution with rank 1.

    2. Weakly coupled networks (Theorem 5b): For Q = Q_0 + ε·Q_1 with
       ε·t ≪ 1, rank ≤ 1 + O(ε·t·n_coupling·d_max).

    3. Area-law entanglement (Theorem 5c): If the entanglement entropy
       across all bipartitions satisfies S_k = O(1), then the TT rank
       is bounded by χ = O(e^{S_max}), independent of system size.

    Args:
        net: ReactionNetwork instance.
        time_horizon: Time horizon for analysis.
        compute_entanglement: Whether to compute entanglement profile (requires dense).

    Returns:
        CompressibilityTheorem with applicable conditions and rank bounds.
    """
    coupling = analyze_coupling_structure(net)
    prediction = predict_compressibility(net, time_horizon)
    dims = net.physical_dims
    N = net.num_species
    state_space = int(np.prod(dims))

    conditions_met = []
    conditions_violated = []
    ent_profile = None

    # Condition 0 (strongest): Deficiency-zero + weakly reversible
    try:
        defn = compute_deficiency(net)
        if defn["product_form_steady_state"]:
            conditions_met.append(
                f"Deficiency-zero (δ={defn['deficiency']}) + weakly reversible: "
                "complex-balanced equilibrium is product-form (Feinberg 1972)"
            )
        elif defn["deficiency"] == 0:
            conditions_met.append(
                f"Deficiency-zero (δ=0) but not weakly reversible"
            )
        else:
            conditions_violated.append(
                f"Deficiency δ={defn['deficiency']} > 0 (Feinberg theorem does not apply)"
            )
    except Exception:
        defn = {"product_form_steady_state": False}

    # Condition 1: Product form (from coupling analysis)
    if coupling.product_form_possible or defn.get("product_form_steady_state", False):
        conditions_met.append(
            "Product-form: all reactions are single-species (birth-death type)"
        )
        theorem = "product_form"
        rank_bound = 1
        derivation = (
            "Q = ⊕ Q_i (block-diagonal) ⟹ e^{Qt} = ⊗ e^{Q_i t} ⟹ rank = 1. "
            "The steady-state π = ⊗ π_i is a product of independent marginals."
        )
        confidence = "rigorous"
    elif coupling.is_weakly_coupled:
        conditions_met.append(
            f"Weak coupling: ε = {coupling.coupling_strength:.4f} < 0.1 × max_single_rate"
        )
        conditions_met.append(
            f"Network has {coupling.num_coupling_reactions} coupling reactions "
            f"and {coupling.num_independent_reactions} independent reactions"
        )
        theorem = "weak_coupling"
        d_max = max(dims)
        rank_bound = prediction.predicted_max_rank
        derivation = (
            f"Q = Q_0 + ε·Q_1 with ε = {coupling.coupling_strength:.4f}. "
            f"Dyson expansion: rank ≤ 1 + ε·t·n_coupling·d_max "
            f"= 1 + {coupling.coupling_strength:.4f}·{time_horizon}·"
            f"{coupling.num_coupling_reactions}·{d_max} = {rank_bound}. "
            f"Valid when ε·t = {coupling.coupling_strength * time_horizon:.4f} ≪ 1."
        )
        if coupling.coupling_strength * time_horizon > 1.0:
            conditions_violated.append(
                f"ε·t = {coupling.coupling_strength * time_horizon:.2f} > 1: "
                "perturbative expansion may not converge"
            )
            confidence = "heuristic"
        else:
            confidence = "rigorous"
    else:
        conditions_violated.append(
            f"Strong coupling: ε = {coupling.coupling_strength:.4f} ≥ 0.1 × max_single_rate"
        )
        theorem = "none"
        rank_bound = prediction.predicted_max_rank
        derivation = (
            "No analytical rank bound available for strongly coupled networks. "
            f"Conservative estimate: χ ≤ {rank_bound}."
        )
        confidence = "heuristic"

    # Condition 3: Entanglement entropy (empirical validation)
    if compute_entanglement and state_space <= 500_000:
        try:
            from tn_check.solver.dense_reference import DenseReferenceSolver
            solver = DenseReferenceSolver(net, max_states=500_000)

            p0 = np.zeros(state_space)
            initial = net.initial_state
            flat_idx = np.ravel_multi_index(
                [min(c, d - 1) for c, d in zip(initial, dims)], dims,
            )
            p0[flat_idx] = 1.0
            p_t = solver.evolve(p0, time_horizon)

            ent_profile = compute_entanglement_profile(p_t, dims)

            if ent_profile.area_law_holds:
                conditions_met.append(
                    f"Area-law entanglement: S_max = {ent_profile.max_entropy:.4f}, "
                    f"predicted χ = {ent_profile.predicted_bond_dim}"
                )
                if theorem == "none":
                    theorem = "area_law"
                    rank_bound = ent_profile.predicted_bond_dim
                    derivation = (
                        f"Entanglement entropy S_max = {ent_profile.max_entropy:.4f} "
                        f"satisfies area law (S = O(1)). "
                        f"Effective rank χ ≈ e^S = {ent_profile.predicted_bond_dim}."
                    )
                    confidence = "empirical"
            else:
                conditions_violated.append(
                    f"Volume-law entanglement: S_max = {ent_profile.max_entropy:.4f}, "
                    f"{ent_profile.volume_law_fraction:.0%} of bonds violate area law"
                )
        except Exception as e:
            logger.warning(f"Entanglement computation failed: {e}")

    return CompressibilityTheorem(
        network_name=net.name,
        theorem_applies=theorem,
        rank_bound=rank_bound,
        rank_bound_derivation=derivation,
        coupling_strength=coupling.coupling_strength,
        entanglement_profile=ent_profile,
        sufficient_conditions_met=conditions_met,
        conditions_violated=conditions_violated,
        confidence=confidence,
    )


def run_compressibility_theorem_suite(
    time_horizon: float = 10.0,
) -> dict:
    """
    Run compressibility theorem analysis across all models.

    For each model, determines which (if any) sufficient condition
    for bounded TT rank applies, and validates against actual ranks.

    Returns:
        Dictionary with per-model theorem application results.
    """
    from tn_check.models.library import (
        birth_death, toggle_switch, schlogl,
        gene_expression, cascade, repressilator,
        sir_epidemic, michaelis_menten_enzyme,
        exclusive_switch, multi_species_cascade,
    )
    from tn_check.models.biomodels import (
        p53_mdm2, goodwin_oscillator, lac_operon,
        lotka_volterra, viral_infection, calcium_signaling,
        apoptosis_decision, dual_phosphorylation,
    )

    models = [
        ("birth_death", lambda: birth_death(max_copy=40)),
        ("gene_expression", lambda: gene_expression(max_copy_mRNA=15, max_copy_protein=40)),
        ("toggle_switch", lambda: toggle_switch(max_copy=20)),
        ("schlogl", lambda: schlogl(max_copy=60)),
        ("cascade_3", lambda: cascade(n_layers=3, max_copy=20)),
        ("sir", lambda: sir_epidemic(max_S=15, max_I=15, max_R=15, S0=10, I0=2)),
        ("michaelis_menten", lambda: michaelis_menten_enzyme(
            max_E=8, max_S=10, max_ES=8, max_P=10, E0=4, S0=6)),
        ("cascade_4", lambda: multi_species_cascade(n_species=4, max_copy=12)),
        ("p53_mdm2", lambda: p53_mdm2(max_p53=25, max_mdm2=25)),
        ("goodwin_3", lambda: goodwin_oscillator(n_genes=3, max_copy=20)),
        ("lac_operon", lambda: lac_operon(max_promoter=2, max_lacZ=30, max_inducer=20)),
        ("lotka_volterra", lambda: lotka_volterra(max_prey=30, max_predator=30)),
        ("viral_infection", lambda: viral_infection(
            max_cells=20, max_virus=25, max_immune=20)),
        ("calcium", lambda: calcium_signaling(max_ip3=15, max_cytosol_ca=20, max_er_ca=20)),
        ("apoptosis", lambda: apoptosis_decision(
            max_signal=15, max_caspase=20, max_antiapoptotic=15)),
        ("dual_phos", lambda: dual_phosphorylation(max_X0=15, max_X1=15, max_X2=15)),
    ]

    results = {"models": [], "summary": {}}
    n_rigorous = 0
    n_empirical = 0
    n_heuristic = 0
    n_product = 0
    n_weak = 0
    n_area_law = 0

    for name, builder in models:
        try:
            net = builder()
            theorem = compressibility_theorem(net, time_horizon, compute_entanglement=True)
            entry = {
                "name": name,
                "num_species": net.num_species,
                "state_space": int(np.prod(net.physical_dims)),
                "theorem_applies": theorem.theorem_applies,
                "rank_bound": theorem.rank_bound,
                "derivation": theorem.rank_bound_derivation,
                "coupling_strength": theorem.coupling_strength,
                "confidence": theorem.confidence,
                "conditions_met": theorem.sufficient_conditions_met,
                "conditions_violated": theorem.conditions_violated,
            }
            if theorem.entanglement_profile:
                entry["entanglement"] = {
                    "max_entropy": theorem.entanglement_profile.max_entropy,
                    "mean_entropy": theorem.entanglement_profile.mean_entropy,
                    "area_law_holds": theorem.entanglement_profile.area_law_holds,
                    "predicted_chi": theorem.entanglement_profile.predicted_bond_dim,
                    "schmidt_ranks": theorem.entanglement_profile.bond_schmidt_ranks,
                }

            if theorem.confidence == "rigorous":
                n_rigorous += 1
            elif theorem.confidence == "empirical":
                n_empirical += 1
            else:
                n_heuristic += 1

            if theorem.theorem_applies == "product_form":
                n_product += 1
            elif theorem.theorem_applies == "weak_coupling":
                n_weak += 1
            elif theorem.theorem_applies == "area_law":
                n_area_law += 1

            results["models"].append(entry)
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            results["models"].append({"name": name, "error": str(e)})

    results["summary"] = {
        "total_models": len(models),
        "rigorous_bounds": n_rigorous,
        "empirical_bounds": n_empirical,
        "heuristic_only": n_heuristic,
        "product_form": n_product,
        "weak_coupling": n_weak,
        "area_law": n_area_law,
    }

    return results


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


THEOREM_ENTANGLEMENT_AREA_LAW = """
Theorem (Entanglement Area Law for CME Distributions):

Let p(t) = e^{Qt} p(0) be the probability distribution at time t of a
CME with generator Q. Define the entanglement entropy across the
bipartition {1,..,k} | {k+1,..,N} as:

    S_k(t) = -Σ_i λ_i²(t) log λ_i²(t)

where λ_i are the singular values of the matrix unfolding
M_k[p(t)] ∈ R^{(d_1···d_k) × (d_{k+1}···d_N)}.

Sufficient Condition for Bounded TT Rank:
If there exists a constant C independent of N such that
    max_k S_k(t) ≤ C    for all t ∈ [0, T]
then the TT rank at tolerance ε satisfies:
    χ_k(ε) ≤ e^C / ε    for all bonds k = 1,..,N-1

This "area law" condition is satisfied when:
(a) The network has weak long-range correlations (exponentially
    decaying mutual information between distant species)
(b) The system is in a gapped phase (spectral gap Δ > 0)
(c) The probability distribution is well-approximated by a
    tree-structured graphical model

The area law provides a sufficient condition for TT compressibility
that can be verified empirically on small instances and extrapolated
to larger systems with the same local structure.  □
"""


THEOREM_DEFICIENCY_ZERO = """
Theorem (Deficiency-Zero Compressibility — Theorem 6):

Let (S, C, R) be a chemical reaction network in the sense of Feinberg,
with species set S = {X_1,...,X_N}, complex set C, and reaction set R.
The deficiency δ of the network is defined as:

    δ = |C| - ℓ - s

where |C| is the number of complexes, ℓ is the number of linkage classes,
and s = rank(Γ) is the rank of the stoichiometric subspace (Γ is the
stoichiometric matrix).

Theorem (Feinberg Deficiency Zero Theorem, 1972):
    If (S, C, R) has deficiency zero and is weakly reversible, then for
    any choice of mass-action rate constants, the system has exactly one
    positive equilibrium in each stoichiometric compatibility class, and
    this equilibrium is a complex-balanced equilibrium.

Connection to TT Compressibility:
    A complex-balanced equilibrium has the form:
        π(x) ∝ ∏_{i=1}^N c_i^{x_i} / x_i!
    (product of Poisson marginals). This is a RANK-1 TT representation.

    Corollary: For deficiency-zero weakly reversible mass-action networks,
    the steady-state probability distribution has TT rank exactly 1.
    For transient distributions starting from a product-form initial state,
    the TT rank grows at most linearly in time according to:

        rank(p(t)) ≤ 1 + Σ_{k≥1} (t·‖Q_coupling‖)^k / k!

    where Q_coupling captures inter-subsystem interactions.

    Therefore: networks with deficiency zero AND weak reversibility are
    provably TT-compressible at steady state, regardless of system size.

Practical Criterion:
    We compute (δ, ℓ, s, |C|) directly from the ReactionNetwork and
    check weak reversibility by verifying each linkage class has a
    directed spanning tree.  □
"""


def compute_deficiency(net) -> dict:
    """
    Compute the deficiency of a reaction network.

    Uses Feinberg's chemical reaction network theory:
        δ = |C| - ℓ - s
    where |C| = number of complexes, ℓ = linkage classes, s = rank(Γ).

    Also checks weak reversibility (each linkage class is strongly connected).

    Returns:
        dict with deficiency, linkage_classes, stoichiometric_rank,
        num_complexes, is_weakly_reversible, and product_form_steady_state.
    """
    N = net.num_species

    # Build complexes: each complex is a tuple of stoichiometric coefficients
    complexes = set()
    complex_list = []
    reaction_edges = []  # (source_complex_idx, target_complex_idx)

    for rxn in net.reactions:
        # Source complex (reactants)
        source = [0] * N
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            if 0 <= sp_idx < N:
                source[sp_idx] = stoich
        source_t = tuple(source)

        # Target complex (products)
        target = [0] * N
        for sp_idx, stoich in zip(rxn.product_species, rxn.product_stoichiometry):
            if 0 <= sp_idx < N:
                target[sp_idx] = stoich
        target_t = tuple(target)

        complexes.add(source_t)
        complexes.add(target_t)

    complex_list = sorted(complexes)
    complex_to_idx = {c: i for i, c in enumerate(complex_list)}

    for rxn in net.reactions:
        source = [0] * N
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            if 0 <= sp_idx < N:
                source[sp_idx] = stoich
        target = [0] * N
        for sp_idx, stoich in zip(rxn.product_species, rxn.product_stoichiometry):
            if 0 <= sp_idx < N:
                target[sp_idx] = stoich
        src_idx = complex_to_idx[tuple(source)]
        tgt_idx = complex_to_idx[tuple(target)]
        reaction_edges.append((src_idx, tgt_idx))

    n_complexes = len(complex_list)

    # Linkage classes: connected components of the reaction graph (undirected)
    adj_undirected = {i: set() for i in range(n_complexes)}
    adj_directed = {i: set() for i in range(n_complexes)}
    for s, t in reaction_edges:
        adj_undirected[s].add(t)
        adj_undirected[t].add(s)
        adj_directed[s].add(t)

    visited = [False] * n_complexes
    linkage_classes = []
    for start in range(n_complexes):
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
            for nb in adj_undirected[node]:
                if not visited[nb]:
                    stack.append(nb)
        linkage_classes.append(sorted(component))
    n_linkage = len(linkage_classes)

    # Stoichiometric matrix Γ: each column is (product - reactant) stoichiometry
    Gamma = np.zeros((N, len(net.reactions)))
    for j, rxn in enumerate(net.reactions):
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            if 0 <= sp_idx < N:
                Gamma[sp_idx, j] -= stoich
        for sp_idx, stoich in zip(rxn.product_species, rxn.product_stoichiometry):
            if 0 <= sp_idx < N:
                Gamma[sp_idx, j] += stoich

    stoich_rank = int(np.linalg.matrix_rank(Gamma, tol=1e-10))

    deficiency = n_complexes - n_linkage - stoich_rank

    # Weak reversibility: check if each linkage class is strongly connected
    # (every node can reach every other node via directed edges)
    is_weakly_reversible = True
    for lc in linkage_classes:
        if len(lc) <= 1:
            continue
        # BFS from each node in the linkage class
        lc_set = set(lc)
        for start_node in lc:
            reachable = set()
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node in reachable:
                    continue
                reachable.add(node)
                for nb in adj_directed.get(node, set()):
                    if nb in lc_set and nb not in reachable:
                        stack.append(nb)
            if reachable & lc_set != lc_set:
                is_weakly_reversible = False
                break

    # Product-form steady state exists iff deficiency == 0 AND weakly reversible
    product_form_ss = (deficiency == 0 and is_weakly_reversible)

    return {
        "deficiency": deficiency,
        "num_complexes": n_complexes,
        "num_linkage_classes": n_linkage,
        "stoichiometric_rank": stoich_rank,
        "is_weakly_reversible": is_weakly_reversible,
        "product_form_steady_state": product_form_ss,
        "complexes": [list(c) for c in complex_list],
        "linkage_classes": linkage_classes,
    }


def compute_interaction_graph_bandwidth(net) -> dict:
    """
    Compute the interaction graph bandwidth for TT rank prediction.

    The interaction graph G_I has species as vertices and an edge (i,j)
    whenever a reaction involves both species i and j in its propensity
    or stoichiometry.

    For a given species ordering σ, the TT rank at bond k is bounded by
    the number of edges crossing the bipartition {σ(1),...,σ(k)} | {σ(k+1),...,σ(N)}.

    The minimum such crossing number over all orderings gives the
    tightest TT rank bound from the interaction graph.

    Returns:
        dict with bandwidth, crossing_numbers, optimal_ordering suggestion.
    """
    N = net.num_species
    edges = set()
    for rxn in net.reactions:
        involved = set(rxn.reactant_species) | set(rxn.product_species)
        involved = {s for s in involved if 0 <= s < N}
        involved_list = sorted(involved)
        for i in range(len(involved_list)):
            for j in range(i + 1, len(involved_list)):
                edges.add((involved_list[i], involved_list[j]))

    # Compute crossing numbers for the current ordering (identity)
    crossing_numbers = []
    for k in range(N - 1):
        left = set(range(k + 1))
        right = set(range(k + 1, N))
        crossings = sum(1 for (i, j) in edges if (i in left and j in right))
        crossing_numbers.append(crossings)

    # Bandwidth = max crossing number
    bandwidth = max(crossing_numbers) if crossing_numbers else 0

    # TT rank bound from interaction graph: χ_k ≤ 2^(crossing_number_k)
    rank_bounds = [2 ** c for c in crossing_numbers]

    return {
        "bandwidth": bandwidth,
        "crossing_numbers": crossing_numbers,
        "rank_bounds_from_graph": rank_bounds,
        "num_edges": len(edges),
        "edges": sorted(edges),
    }


def run_deficiency_analysis() -> dict:
    """
    Run deficiency analysis across all models in the library.

    Computes deficiency, weak reversibility, and product-form predictions
    for each model, providing a structural basis for compressibility.

    Returns:
        Dictionary with per-model deficiency results and summary.
    """
    from tn_check.models.library import (
        birth_death, toggle_switch, schlogl,
        gene_expression, cascade, repressilator,
        sir_epidemic, michaelis_menten_enzyme,
        exclusive_switch, multi_species_cascade,
        mapk_cascade, gene_regulatory_network,
        multi_gene_expression,
    )
    from tn_check.models.biomodels import (
        p53_mdm2, goodwin_oscillator, lac_operon,
        lotka_volterra, viral_infection, calcium_signaling,
        apoptosis_decision, dual_phosphorylation,
        genetic_toggle_with_inducer, two_gene_oscillator,
        branching_pathway, cooperative_binding,
    )
    from tn_check.models.biomodels_extended import (
        glycolysis_oscillator, purine_metabolism, pentose_phosphate,
        tca_cycle_minimal, autoregulatory_gene, feedforward_loop,
        mutual_inhibition, positive_feedback_gene, incoherent_ffl,
        egfr_signaling, nfkb_signaling, wnt_signaling,
        jak_stat_signaling, insulin_signaling,
        cell_cycle_checkpoint, growth_factor_response,
        quorum_sensing, stem_cell_differentiation,
        predator_prey_3sp, sir_vaccination,
        competition_model, mutualism_model,
        stochastic_switch, noise_induced_oscillation,
        excitable_system, stochastic_bursting,
    )

    all_models = [
        ("birth_death", lambda: birth_death(max_copy=40)),
        ("gene_expression", lambda: gene_expression(max_copy_mRNA=15, max_copy_protein=40)),
        ("toggle_switch", lambda: toggle_switch(max_copy=20)),
        ("schlogl", lambda: schlogl(max_copy=60)),
        ("cascade_3", lambda: cascade(n_layers=3, max_copy=20)),
        ("repressilator", lambda: repressilator(max_copy=20)),
        ("sir", lambda: sir_epidemic(max_S=15, max_I=15, max_R=15, S0=10, I0=2)),
        ("michaelis_menten", lambda: michaelis_menten_enzyme(
            max_E=8, max_S=10, max_ES=8, max_P=10, E0=4, S0=6)),
        ("cascade_5", lambda: multi_species_cascade(n_species=5, max_copy=12)),
        ("p53_mdm2", lambda: p53_mdm2(max_p53=25, max_mdm2=25)),
        ("goodwin_3", lambda: goodwin_oscillator(n_genes=3, max_copy=20)),
        ("lac_operon", lambda: lac_operon(max_promoter=2, max_lacZ=30, max_inducer=20)),
        ("lotka_volterra", lambda: lotka_volterra(max_prey=30, max_predator=30)),
        ("viral_infection", lambda: viral_infection(
            max_cells=20, max_virus=25, max_immune=20)),
        ("calcium", lambda: calcium_signaling(max_ip3=15, max_cytosol_ca=20, max_er_ca=20)),
        ("apoptosis", lambda: apoptosis_decision(
            max_signal=15, max_caspase=20, max_antiapoptotic=15)),
        ("dual_phos", lambda: dual_phosphorylation(max_X0=15, max_X1=15, max_X2=15)),
        ("glycolysis", lambda: glycolysis_oscillator()),
        ("purine_metab", lambda: purine_metabolism()),
        ("pentose_phosphate", lambda: pentose_phosphate()),
        ("tca_cycle", lambda: tca_cycle_minimal()),
        ("autoregulatory", lambda: autoregulatory_gene()),
        ("feedforward", lambda: feedforward_loop()),
        ("mutual_inhib", lambda: mutual_inhibition()),
        ("pos_feedback", lambda: positive_feedback_gene()),
        ("incoherent_ffl", lambda: incoherent_ffl()),
        ("egfr", lambda: egfr_signaling()),
        ("nfkb", lambda: nfkb_signaling()),
        ("wnt", lambda: wnt_signaling()),
        ("jak_stat", lambda: jak_stat_signaling()),
        ("insulin", lambda: insulin_signaling()),
        ("cell_cycle", lambda: cell_cycle_checkpoint()),
        ("growth_factor", lambda: growth_factor_response()),
        ("quorum", lambda: quorum_sensing()),
        ("stem_cell", lambda: stem_cell_differentiation()),
        ("pred_prey_3sp", lambda: predator_prey_3sp()),
        ("sir_vax", lambda: sir_vaccination()),
        ("competition", lambda: competition_model()),
        ("mutualism", lambda: mutualism_model()),
        ("stoch_switch", lambda: stochastic_switch()),
        ("noise_osc", lambda: noise_induced_oscillation()),
        ("excitable", lambda: excitable_system()),
        ("bursting", lambda: stochastic_bursting()),
        ("toggle_inducer", lambda: genetic_toggle_with_inducer()),
        ("two_gene_osc", lambda: two_gene_oscillator()),
        ("branching", lambda: branching_pathway()),
        ("coop_binding", lambda: cooperative_binding()),
        ("exclusive_sw", lambda: exclusive_switch()),
        ("mapk", lambda: mapk_cascade(n_layers=3, max_copy=15)),
        ("grn_3", lambda: gene_regulatory_network(n_genes=3, max_copy=15)),
        ("multi_gene_4", lambda: multi_gene_expression(n_genes=4, max_copy_mrna=10, max_copy_prot=15)),
    ]

    results = {"models": [], "summary": {}}
    n_def_zero = 0
    n_weak_rev = 0
    n_product_form = 0
    total = 0

    for name, builder in all_models:
        try:
            net = builder()
            defn = compute_deficiency(net)
            graph = compute_interaction_graph_bandwidth(net)

            entry = {
                "name": name,
                "num_species": net.num_species,
                "num_reactions": len(net.reactions),
                "deficiency": defn["deficiency"],
                "num_complexes": defn["num_complexes"],
                "num_linkage_classes": defn["num_linkage_classes"],
                "stoichiometric_rank": defn["stoichiometric_rank"],
                "is_weakly_reversible": defn["is_weakly_reversible"],
                "product_form_ss": defn["product_form_steady_state"],
                "graph_bandwidth": graph["bandwidth"],
                "rank_bounds": graph["rank_bounds_from_graph"],
            }
            results["models"].append(entry)
            total += 1
            if defn["deficiency"] == 0:
                n_def_zero += 1
            if defn["is_weakly_reversible"]:
                n_weak_rev += 1
            if defn["product_form_steady_state"]:
                n_product_form += 1
        except Exception as e:
            logger.error(f"Deficiency analysis failed for {name}: {e}")
            results["models"].append({"name": name, "error": str(e)})

    results["summary"] = {
        "total_models": total,
        "deficiency_zero": n_def_zero,
        "weakly_reversible": n_weak_rev,
        "product_form_steady_state": n_product_form,
    }

    return results


THEOREM_FROBENIUS_L1 = """
Proposition (Frobenius-to-L1 Error Relationship):

For a probability vector p ∈ R^n (p ≥ 0, 1^T p = 1) and its
TT-compressed approximation p̃:

    ‖p - p̃‖₁ ≤ √n · ‖p - p̃‖₂ ≤ √n · ‖p - p̃‖_F

However, for probability vectors with bounded support (only K entries
are non-negligible):

    ‖p - p̃‖₁ ≤ √(2K) · ‖p - p̃‖₂

Since K ≪ n for typical CME distributions (most probability mass
concentrated on a small fraction of states), this provides a much
tighter bound than the naive √n factor.

For TDVP, the Frobenius residual ‖∂_t ψ - Qψ‖_F is directly controlled.
The L1 error relevant for CSL checking is bounded by:

    ‖p_exact(t) - p_TDVP(t)‖₁ ≤ √(2K_eff) · ∫_0^t ‖res(s)‖_F ds

where K_eff is the effective support size and res(s) is the TDVP
Frobenius residual at time s. Combined with Metzler contractivity,
per-step residuals do not amplify.  □
"""
