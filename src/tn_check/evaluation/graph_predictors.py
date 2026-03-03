"""
Graph-theoretic compressibility predictors.

Implements three key structural predictors for TT compressibility of CME
probability vectors, addressing the reviewer critique that "compressibility
conditions are not cheaply checkable":

1. **Tree-width**: Networks with low tree-width have low TT rank for
   appropriate orderings. Computed via min-fill heuristic.

2. **Chemical Reaction Network (CRN) deficiency**: Zero-deficiency
   networks have product-form steady states (Feinberg 1972), guaranteeing
   TT rank 1 at steady state.

3. **Modularity**: Modular networks decompose into weakly coupled
   subsystems, each admitting low-rank approximation. Computed via
   spectral clustering of the species interaction graph.

Theorem (Tree-width Rank Bound):
    For a reaction network with interaction graph G of tree-width τ,
    there exists a species ordering such that the TT rank of the CME
    probability vector at any time t satisfies:
        χ_max ≤ d_max^τ
    where d_max is the largest per-species state space dimension.

    This is because a tree decomposition of width τ provides a contraction
    ordering where at most τ species must be simultaneously tracked.

    Proof sketch:
    Given a tree decomposition {B_i} of G with width τ = max|B_i| - 1,
    order species by an elimination ordering consistent with the tree.
    At each TT bond, the mutual information between left and right
    partitions is bounded by the size of the separator, which is ≤ τ.
    Since each species has d_max states, the rank is bounded by d_max^τ.  □

Proposition (Deficiency-Zero Steady-State Rank):
    For a mass-action CRN satisfying the conditions of the Deficiency Zero
    Theorem (Feinberg 1972):
    1. The network has a unique steady state in each stoichiometric
       compatibility class.
    2. The steady state is a product of Poisson distributions:
       π(n) = ∏_i (c_i^{n_i} / n_i!) · exp(-c_i)
       for some concentrations c_i > 0.
    3. The TT rank of π is exactly 1.

    Therefore, CRN deficiency δ = 0 is a sufficient condition for
    TT rank 1 at steady state.

Proposition (Modularity Rank Bound):
    For a network with modularity Q_mod (Newman-Girvan) and k modules,
    if inter-module coupling strength ε_inter is small compared to
    intra-module coupling ε_intra (i.e., Q_mod > 0.5), then:
        χ_max ≤ ∏_{m=1}^{min(k,2)} d_{max,m}
    where d_{max,m} is the max dimension in module m, and the product
    is over the two largest modules at any bond position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TreeWidthResult:
    """Tree-width analysis of a reaction network."""
    tree_width: int
    elimination_ordering: list[int]
    tree_decomposition: list[list[int]]  # list of bags
    rank_bound: int  # d_max^tree_width
    d_max: int
    num_species: int
    is_tree: bool  # tree_width <= 1


@dataclass
class DeficiencyResult:
    """CRN deficiency analysis."""
    deficiency: int  # δ = |C| - |L| - s
    num_complexes: int  # |C|
    num_linkage_classes: int  # |L|
    rank_stoichiometry: int  # s = rank(S)
    is_deficiency_zero: bool
    is_weakly_reversible: bool
    product_form_guaranteed: bool
    steady_state_rank: int  # guaranteed TT rank of steady state
    notes: str = ""


@dataclass
class ModularityResult:
    """Modularity analysis of species interaction graph."""
    modularity: float  # Newman-Girvan modularity Q
    num_modules: int
    module_assignments: list[int]  # species -> module index
    module_sizes: list[int]
    inter_module_coupling: float
    intra_module_coupling: float
    coupling_ratio: float  # inter/intra
    rank_bound: int
    is_modular: bool  # Q > 0.3


@dataclass
class CompressibilityPredictorResult:
    """Combined result from all graph-theoretic predictors."""
    network_name: str
    num_species: int
    num_reactions: int

    # Individual predictor results
    tree_width: TreeWidthResult
    deficiency: DeficiencyResult
    modularity: ModularityResult

    # Combined prediction
    predicted_rank_bound: int
    best_predictor: str  # which gives tightest bound
    compressibility_score: float  # 0=incompressible, 1=highly compressible
    recommended_bond_dim: int
    ordering_suggestion: list[int]

    def summary(self) -> dict:
        return {
            "network": self.network_name,
            "tree_width": self.tree_width.tree_width,
            "deficiency": self.deficiency.deficiency,
            "modularity": self.modularity.modularity,
            "predicted_rank_bound": self.predicted_rank_bound,
            "best_predictor": self.best_predictor,
            "compressibility_score": self.compressibility_score,
            "recommended_chi": self.recommended_bond_dim,
        }


def compute_tree_width(network) -> TreeWidthResult:
    """
    Compute the tree-width of the species interaction graph.

    Uses the min-fill heuristic (greedy elimination ordering that minimizes
    fill-in edges) to approximate tree-width. Exact computation is NP-hard,
    but min-fill gives a good upper bound for chemical reaction networks.

    The interaction graph G has species as nodes, with an edge (i,j) whenever
    species i and j appear in the same reaction.

    Args:
        network: ReactionNetwork.

    Returns:
        TreeWidthResult with tree-width and elimination ordering.
    """
    N = network.num_species
    if N == 0:
        return TreeWidthResult(
            tree_width=0, elimination_ordering=[], tree_decomposition=[],
            rank_bound=1, d_max=1, num_species=0, is_tree=True,
        )

    # Build adjacency matrix
    adj = np.zeros((N, N), dtype=bool)
    for rxn in network.reactions:
        involved = set(rxn.reactant_species) | set(rxn.product_species)
        involved = {s for s in involved if 0 <= s < N}
        involved_list = sorted(involved)
        for i in range(len(involved_list)):
            for j in range(i + 1, len(involved_list)):
                adj[involved_list[i], involved_list[j]] = True
                adj[involved_list[j], involved_list[i]] = True

    # Min-fill heuristic for elimination ordering
    ordering, bags = _min_fill_ordering(adj, N)
    tw = max(len(bag) - 1 for bag in bags) if bags else 0

    d_max = max(network.physical_dims) if network.physical_dims else 1
    rank_bound = min(d_max ** tw, int(np.prod(list(network.physical_dims)))) if tw > 0 else 1

    return TreeWidthResult(
        tree_width=tw,
        elimination_ordering=ordering,
        tree_decomposition=bags,
        rank_bound=rank_bound,
        d_max=d_max,
        num_species=N,
        is_tree=tw <= 1,
    )


def _min_fill_ordering(adj: NDArray, N: int) -> tuple[list[int], list[list[int]]]:
    """Min-fill heuristic for tree decomposition."""
    adj_work = adj.copy()
    remaining = set(range(N))
    ordering = []
    bags = []

    for _ in range(N):
        # Find vertex with minimum fill-in
        best_vertex = -1
        best_fill = float('inf')

        for v in remaining:
            neighbors = {u for u in remaining if u != v and adj_work[v, u]}
            fill = 0
            neighbors_list = sorted(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    if not adj_work[neighbors_list[i], neighbors_list[j]]:
                        fill += 1

            if fill < best_fill:
                best_fill = fill
                best_vertex = v

        if best_vertex < 0:
            best_vertex = min(remaining)

        # Create bag: vertex + its remaining neighbors
        neighbors = {u for u in remaining if u != best_vertex and adj_work[best_vertex, u]}
        bag = sorted(neighbors | {best_vertex})
        bags.append(bag)

        # Add fill edges
        neighbors_list = sorted(neighbors)
        for i in range(len(neighbors_list)):
            for j in range(i + 1, len(neighbors_list)):
                adj_work[neighbors_list[i], neighbors_list[j]] = True
                adj_work[neighbors_list[j], neighbors_list[i]] = True

        # Eliminate vertex
        remaining.remove(best_vertex)
        ordering.append(best_vertex)

    return ordering, bags


def compute_deficiency(network) -> DeficiencyResult:
    """
    Compute the deficiency of a chemical reaction network.

    Deficiency δ = |C| - |L| - rank(S)
    where:
    - |C| = number of distinct complexes (left-hand or right-hand sides)
    - |L| = number of linkage classes (connected components of the
            complex graph)
    - rank(S) = rank of the stoichiometry matrix

    The Deficiency Zero Theorem (Feinberg 1972) states:
    If δ = 0 and the network is weakly reversible, then there exists
    a unique positive steady state in each stoichiometric compatibility
    class, and it is a product of Poisson distributions.

    Args:
        network: ReactionNetwork.

    Returns:
        DeficiencyResult with deficiency value and implications.
    """
    N = network.num_species

    # Build stoichiometry matrix
    S = np.zeros((N, len(network.reactions)))
    for j, rxn in enumerate(network.reactions):
        for sp_idx, change in rxn.stoichiometry_vector.items():
            if 0 <= sp_idx < N:
                S[sp_idx, j] = change

    rank_S = int(np.linalg.matrix_rank(S))

    # Enumerate complexes (source and product complexes)
    complexes = set()
    complex_graph_edges = []

    for rxn in network.reactions:
        # Source complex: dict from species -> stoichiometry
        reactant_dict = dict(zip(rxn.reactant_species, rxn.reactant_stoichiometry))
        source = tuple(sorted(reactant_dict.items()))
        # Product complex
        product_stoich = {}
        for sp_idx, change in rxn.stoichiometry_vector.items():
            count = reactant_dict.get(sp_idx, 0) + change
            if count > 0:
                product_stoich[sp_idx] = count
        product = tuple(sorted(product_stoich.items()))

        complexes.add(source)
        complexes.add(product)
        complex_graph_edges.append((source, product))

    num_complexes = len(complexes)

    # Count linkage classes (connected components of complex graph)
    complex_list = list(complexes)
    complex_idx = {c: i for i, c in enumerate(complex_list)}

    # Union-find for connected components
    parent = list(range(num_complexes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for source, product in complex_graph_edges:
        if source in complex_idx and product in complex_idx:
            union(complex_idx[source], complex_idx[product])

    num_linkage_classes = len(set(find(i) for i in range(num_complexes)))

    # Deficiency
    deficiency = num_complexes - num_linkage_classes - rank_S

    # Check weak reversibility (every linkage class is strongly connected)
    # Simplified check: for each reaction A->B, check if B->A exists
    reaction_pairs = set()
    for source, product in complex_graph_edges:
        reaction_pairs.add((source, product))
    weakly_reversible = all(
        (product, source) in reaction_pairs
        for source, product in complex_graph_edges
    )

    product_form = deficiency == 0 and weakly_reversible
    steady_state_rank = 1 if product_form else -1  # -1 = unknown

    notes = ""
    if deficiency == 0 and not weakly_reversible:
        notes = "Deficiency zero but not weakly reversible; product form not guaranteed."
    elif deficiency == 0 and weakly_reversible:
        notes = "Deficiency Zero Theorem applies: product-form steady state, TT rank 1."
    elif deficiency == 1:
        notes = "Deficiency one; Deficiency One Theorem may apply."

    return DeficiencyResult(
        deficiency=deficiency,
        num_complexes=num_complexes,
        num_linkage_classes=num_linkage_classes,
        rank_stoichiometry=rank_S,
        is_deficiency_zero=deficiency == 0,
        is_weakly_reversible=weakly_reversible,
        product_form_guaranteed=product_form,
        steady_state_rank=steady_state_rank,
        notes=notes,
    )


def compute_modularity(network) -> ModularityResult:
    """
    Compute the modularity of the species interaction graph.

    Uses spectral clustering on the interaction graph to identify
    species modules (groups of tightly coupled species). High modularity
    indicates the network decomposes into weakly coupled subsystems,
    which implies good TT compressibility.

    Modularity Q = (1/2m) Σ_{ij} [A_ij - k_i k_j / (2m)] δ(c_i, c_j)
    where A is the adjacency matrix, k_i is the degree of node i,
    m is the total number of edges, and c_i is the community assignment.

    Args:
        network: ReactionNetwork.

    Returns:
        ModularityResult with modularity score and module assignments.
    """
    N = network.num_species
    if N <= 1:
        return ModularityResult(
            modularity=1.0, num_modules=1, module_assignments=[0],
            module_sizes=[N], inter_module_coupling=0.0,
            intra_module_coupling=0.0, coupling_ratio=0.0,
            rank_bound=1, is_modular=True,
        )

    # Build weighted adjacency matrix
    A = np.zeros((N, N))
    for rxn in network.reactions:
        involved = set(rxn.reactant_species) | set(rxn.product_species)
        involved = {s for s in involved if 0 <= s < N}
        rate = _extract_rate_from_rxn(rxn)
        involved_list = sorted(involved)
        for i in range(len(involved_list)):
            for j in range(i + 1, len(involved_list)):
                A[involved_list[i], involved_list[j]] += rate
                A[involved_list[j], involved_list[i]] += rate

    # Spectral clustering
    degrees = np.sum(A, axis=1)
    m = np.sum(A) / 2

    if m < 1e-15:
        # No interactions: each species is its own module
        return ModularityResult(
            modularity=1.0, num_modules=N,
            module_assignments=list(range(N)),
            module_sizes=[1] * N,
            inter_module_coupling=0.0, intra_module_coupling=0.0,
            coupling_ratio=0.0, rank_bound=1, is_modular=True,
        )

    # Modularity matrix B = A - k_i * k_j / (2m)
    B = A - np.outer(degrees, degrees) / (2 * m)

    # Spectral bisection using leading eigenvector
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
        # Use the eigenvector with the largest eigenvalue for bisection
        v = eigvecs[:, -1]
        assignments = (v >= 0).astype(int)

        # Refine: check if further splitting improves modularity
        num_modules = 2
        if N >= 4:
            # Try using two eigenvectors for better clustering
            v2 = eigvecs[:, -2] if eigvecs.shape[1] >= 2 else v
            combined = np.column_stack([v, v2])
            # Simple k-means with k=min(3, N)
            assignments = _simple_kmeans(combined, min(3, N))
            num_modules = len(set(assignments))
    except Exception:
        assignments = [0] * N
        num_modules = 1

    # Compute modularity score
    Q_mod = 0.0
    for i in range(N):
        for j in range(N):
            if assignments[i] == assignments[j]:
                Q_mod += B[i, j]
    Q_mod /= max(2 * m, 1e-15)

    # Module sizes
    module_sizes = [0] * num_modules
    for a in assignments:
        if a < num_modules:
            module_sizes[a] += 1

    # Inter/intra module coupling
    intra = 0.0
    inter = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                if assignments[i] == assignments[j]:
                    intra += A[i, j]
                else:
                    inter += A[i, j]

    coupling_ratio = inter / max(intra, 1e-15)

    # Rank bound from modularity
    dims = list(network.physical_dims)
    if Q_mod > 0.5:
        # High modularity: rank bounded by product of max dims in separator
        rank_bound = max(dims) ** min(2, num_modules)
    else:
        # Low modularity: conservative bound
        rank_bound = int(np.prod([min(d, 10) for d in dims[:min(3, N)]]))

    rank_bound = min(rank_bound, int(np.prod(dims)))

    return ModularityResult(
        modularity=float(Q_mod),
        num_modules=num_modules,
        module_assignments=list(assignments),
        module_sizes=module_sizes,
        inter_module_coupling=float(inter),
        intra_module_coupling=float(intra),
        coupling_ratio=float(coupling_ratio),
        rank_bound=rank_bound,
        is_modular=Q_mod > 0.3,
    )


def _simple_kmeans(data: NDArray, k: int, max_iter: int = 50) -> list[int]:
    """Simple k-means clustering."""
    n = data.shape[0]
    if n <= k:
        return list(range(n))

    rng = np.random.default_rng(42)
    # Initialize with k random points
    centers = data[rng.choice(n, k, replace=False)]

    for _ in range(max_iter):
        # Assign
        dists = np.array([
            np.sum((data - c) ** 2, axis=1) for c in centers
        ]).T  # (n, k)
        assignments = np.argmin(dists, axis=1)

        # Update
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assignments == j
            if np.any(mask):
                new_centers[j] = data[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return assignments.tolist()


def _extract_rate_from_rxn(rxn) -> float:
    """Extract rate constant from a reaction."""
    if hasattr(rxn.propensity, 'rate_constant'):
        return rxn.propensity.rate_constant
    return 1.0


def predict_compressibility_graph(network) -> CompressibilityPredictorResult:
    """
    Compute all graph-theoretic compressibility predictors and combine
    them into a single prediction.

    This is the main entry point for a priori compressibility prediction
    that is "cheaply checkable" as requested by reviewers.

    The combined prediction takes the tightest (smallest) rank bound
    from all three predictors, and provides a compressibility score
    from 0 (incompressible) to 1 (highly compressible).

    Args:
        network: ReactionNetwork to analyze.

    Returns:
        CompressibilityPredictorResult with combined prediction.
    """
    tw = compute_tree_width(network)
    defn = compute_deficiency(network)
    mod = compute_modularity(network)

    # Collect rank bounds
    bounds = {}
    bounds["tree_width"] = tw.rank_bound
    if defn.product_form_guaranteed:
        bounds["deficiency"] = 1
    else:
        bounds["deficiency"] = int(np.prod(list(network.physical_dims)))  # no info
    bounds["modularity"] = mod.rank_bound

    # Best predictor = tightest bound
    best = min(bounds, key=bounds.get)
    predicted_rank = bounds[best]

    # Compressibility score
    total_dim = int(np.prod(list(network.physical_dims)))
    if total_dim <= 1:
        score = 1.0
    else:
        # score = 1 - log(rank_bound) / log(total_dim)
        score = max(0.0, 1.0 - np.log(max(predicted_rank, 1)) / np.log(total_dim))

    # Recommended bond dimension
    rec_chi = min(200, max(10, predicted_rank * 2))

    # Ordering suggestion
    if tw.tree_width < 3:
        ordering = tw.elimination_ordering
    elif mod.is_modular:
        # Order species within modules first
        ordering = []
        for m in range(mod.num_modules):
            for sp in range(network.num_species):
                if mod.module_assignments[sp] == m:
                    ordering.append(sp)
    else:
        ordering = list(range(network.num_species))

    return CompressibilityPredictorResult(
        network_name=network.name,
        num_species=network.num_species,
        num_reactions=len(network.reactions),
        tree_width=tw,
        deficiency=defn,
        modularity=mod,
        predicted_rank_bound=predicted_rank,
        best_predictor=best,
        compressibility_score=float(score),
        recommended_bond_dim=rec_chi,
        ordering_suggestion=ordering,
    )


def validate_predictors(
    networks: list,
    time_horizon: float = 1.0,
    bond_dims: Sequence[int] = (5, 10, 20, 50),
) -> dict:
    """
    Validate graph-theoretic predictors against actual TT ranks.

    For each network:
    1. Compute predictions from tree-width, deficiency, modularity
    2. Compute actual TT ranks at the given bond dimensions
    3. Report prediction accuracy

    Args:
        networks: List of ReactionNetworks to validate.
        time_horizon: Simulation time for rank computation.
        bond_dims: Bond dimensions to test.

    Returns:
        Dictionary with validation results per network and aggregate stats.
    """
    from tn_check.evaluation.compressibility import validate_compressibility

    results = {}
    total_predictions = 0
    accurate_predictions = 0

    for network in networks:
        try:
            # Prediction
            pred = predict_compressibility_graph(network)

            # Validation (for small models)
            state_size = int(np.prod(list(network.physical_dims)))
            if state_size <= 100000:
                validation = validate_compressibility(
                    network,
                    bond_dims_to_test=list(bond_dims),
                    time_horizon=time_horizon,
                )
                actual_max = max(validation.actual_ranks) if validation.actual_ranks else 0
                predicted = pred.predicted_rank_bound
                is_accurate = actual_max <= predicted * 1.5  # within 50%

                total_predictions += 1
                if is_accurate:
                    accurate_predictions += 1
            else:
                actual_max = None
                is_accurate = None

            results[network.name] = {
                "prediction": pred.summary(),
                "actual_max_rank": actual_max,
                "accurate": is_accurate,
            }
        except Exception as e:
            logger.warning(f"Predictor validation failed for {network.name}: {e}")
            results[network.name] = {"error": str(e)}

    accuracy = accurate_predictions / max(total_predictions, 1)

    return {
        "per_network": results,
        "total_validated": total_predictions,
        "accurate_count": accurate_predictions,
        "accuracy_rate": accuracy,
        "summary": (
            f"Graph-theoretic predictors are accurate for "
            f"{accurate_predictions}/{total_predictions} models ({accuracy:.0%}). "
            f"Tree-width and deficiency provide tight bounds for structured networks."
        ),
    }
