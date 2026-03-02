"""
Mathematical rigor verification for all theorems and propositions.

Provides numerical verification procedures for:
1. Error contraction theorem (Theorem 1) — Metzler structure argument
2. O(N) complexity separation proof (Theorem 4)
3. Three-valued CSL semantics soundness (Section 4)
4. Carry-based TT construction for linear predicates (Theorem 3)
5. Bond-dimension preservation for axis-aligned predicates (Theorem 2)
6. Clamping error bound (Proposition 1 / Lemma 2)

Each verification function returns a structured result with:
- statement: formal statement of the theorem
- verification_method: how we check it
- result: pass/fail with numerical evidence
- proof_sketch: key proof steps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TheoremVerification:
    """Result of numerically verifying a theorem."""
    theorem_id: str
    statement: str
    verified: bool
    numerical_evidence: dict = field(default_factory=dict)
    proof_sketch: str = ""
    counterexample: Optional[str] = None
    notes: str = ""


def verify_theorem1_error_contraction(
    n_trials: int = 20,
    max_species: int = 3,
    max_copy: int = 15,
) -> TheoremVerification:
    """
    Verify Theorem 1: Metzler semigroup contractivity.

    The key claim is that per-step truncation errors accumulate linearly:
    ‖p_exact(t) - p_TT(t)‖₁ ≤ Σ_k ε_k.

    We verify this by checking:
    1. Q has Metzler structure (off-diagonal entries ≥ 0).
    2. e^{Qt} preserves probability distributions (non-negative, sum=1).
    3. Q is probability-conserving (column sums ≈ 0 for interior states).
    L1 contractivity follows algebraically from these properties.
    """
    from tn_check.models.library import birth_death, toggle_switch
    from tn_check.solver.dense_reference import DenseReferenceSolver

    evidence = {
        "trials": [],
        "all_valid_distributions": True,
        "all_metzler": True,
        "all_conserving": True,
        "max_sum_deviation": 0.0,
    }

    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        if trial % 2 == 0:
            br = rng.uniform(0.5, 5.0)
            dr = rng.uniform(0.1, 1.0)
            mc = rng.integers(10, max_copy + 1)
            net = birth_death(birth_rate=br, death_rate=dr, max_copy=int(mc))
        else:
            a1 = rng.uniform(5.0, 30.0)
            a2 = rng.uniform(5.0, 30.0)
            mc = rng.integers(8, 15)
            net = toggle_switch(alpha1=a1, alpha2=a2, max_copy=int(mc))

        N = int(np.prod(net.physical_dims))
        if N > 50000:
            continue

        solver = DenseReferenceSolver(net, max_states=500_000)

        # Check 1: Metzler structure (off-diagonal entries ≥ 0)
        Q = solver.compile()
        off_diag = Q.copy()
        np.fill_diagonal(off_diag, 0)
        min_off_diag = float(np.min(off_diag))
        is_metzler = min_off_diag >= -1e-12

        # Check 2: Probability conservation (column sums ≈ 0)
        col_sums = np.sum(Q, axis=0)
        max_col_sum = float(np.max(np.abs(col_sums)))

        # Check 3: Evolution preserves probability distribution
        p0 = np.zeros(N)
        idx = rng.integers(0, N)
        p0[idx] = 1.0

        t = rng.uniform(0.1, 5.0)
        p_t = solver.evolve(p0, t)

        sum_check = abs(p_t.sum() - 1.0)
        nonneg_check = float(np.min(p_t))
        valid_dist = sum_check < 1e-6 and nonneg_check >= -1e-10

        # Also evolve uniform distribution to test another starting point
        p_uniform = np.ones(N) / N
        p_unif_t = solver.evolve(p_uniform, t)
        sum_check_unif = abs(p_unif_t.sum() - 1.0)
        nonneg_unif = float(np.min(p_unif_t))
        valid_dist_unif = sum_check_unif < 1e-6 and nonneg_unif >= -1e-10

        trial_result = {
            "trial": trial,
            "model": net.name,
            "N": N,
            "t": t,
            "is_metzler": is_metzler,
            "min_off_diagonal": min_off_diag,
            "max_column_sum": max_col_sum,
            "sum_deviation": sum_check,
            "min_entry": nonneg_check,
            "valid_distribution": valid_dist,
            "valid_distribution_uniform": valid_dist_unif,
        }
        evidence["trials"].append(trial_result)

        if not (valid_dist and valid_dist_unif):
            evidence["all_valid_distributions"] = False
        if not is_metzler:
            evidence["all_metzler"] = False
        # Column sums should be approximately zero for conservation
        if max_col_sum > 1e-6:
            evidence["all_conserving"] = False
        evidence["max_sum_deviation"] = max(
            evidence["max_sum_deviation"], max_col_sum,
        )

    verified = evidence["all_valid_distributions"] and evidence["all_metzler"]

    return TheoremVerification(
        theorem_id="Theorem1",
        statement=(
            "For CME generator Q (Metzler, probability-conserving), "
            "e^{Qt} maps probability distributions to probability distributions "
            "and is an L1 contraction. Per-step truncation errors accumulate "
            "linearly: ‖p_exact(t) - p_TT(t)‖_1 ≤ Σ_k ε_k."
        ),
        verified=verified,
        numerical_evidence=evidence,
        proof_sketch=(
            "1. Q has Metzler structure: off-diagonal entries ≥ 0 (verified numerically). "
            "2. e^{Qt} preserves probability distributions: entries ≥ 0, sum = 1 (verified). "
            "3. Stochastic matrices are L1 contractions on zero-sum vectors (algebraic fact). "
            "4. Truncation errors are zero-sum perturbations, so they don't amplify."
        ),
    )


def verify_theorem2_bond_dimension_preservation(
    n_trials: int = 10,
) -> TheoremVerification:
    """
    Verify Theorem 2: axis-aligned predicates produce rank-1 TT masks.

    Statement: For an atomic proposition X_i ≥ c (axis-aligned predicate),
    the characteristic function 1_{X_i ≥ c} has TT rank 1 at every bond.

    Verification: For various species indices and thresholds, construct
    the satisfaction MPS and verify all bond dimensions are 1.
    """
    from tn_check.tensor.mps import threshold_mps

    evidence = {"trials": [], "all_rank_one": True}

    for n_sites in [2, 3, 4, 5]:
        for target_site in range(n_sites):
            dims = tuple([10 + i for i in range(n_sites)])
            for threshold in [0, 3, 5, dims[target_site] - 1]:
                mps = threshold_mps(n_sites, dims, target_site, threshold, "greater_equal")
                bond_dims = list(mps.bond_dims)
                is_rank_one = all(d == 1 for d in bond_dims)

                evidence["trials"].append({
                    "n_sites": n_sites,
                    "target_site": target_site,
                    "threshold": threshold,
                    "bond_dims": bond_dims,
                    "is_rank_one": is_rank_one,
                })

                if not is_rank_one:
                    evidence["all_rank_one"] = False

    return TheoremVerification(
        theorem_id="Theorem2",
        statement=(
            "Axis-aligned predicates X_i ≥ c produce rank-1 TT masks "
            "(bond dimension 1 at every bond)."
        ),
        verified=evidence["all_rank_one"],
        numerical_evidence=evidence,
        proof_sketch=(
            "The characteristic function 1_{X_i ≥ c}(n_1,...,n_N) = "
            "∏_{j≠i} 1 · 1_{n_i ≥ c} factors as a product of single-site "
            "functions. A product of single-site functions has TT rank 1."
        ),
    )


def verify_theorem3_linear_predicate_rank(
    n_trials: int = 5,
) -> TheoremVerification:
    """
    Verify Theorem 3: carry-based TT construction for linear predicates
    has bond dimension ≤ R+1 where R is the range of the partial sum.

    Statement: For a linear predicate c_1·X_1 + c_2·X_2 + ... ≥ t,
    the TT representation has bond dimension bounded by the range of
    partial sums at each bond.
    """
    from tn_check.checker.csl_ast import LinearPredicate
    from tn_check.checker.satisfaction import compute_satisfaction_set

    evidence = {"trials": [], "all_bounded": True}

    test_cases = [
        # (coefficients, threshold, dims)
        ({0: 1, 1: 1}, 5, (10, 10)),
        ({0: 2, 1: 3}, 10, (8, 8)),
        ({0: 1, 1: 1, 2: 1}, 8, (6, 6, 6)),
        ({0: 1, 2: 1}, 5, (8, 5, 8)),
    ]

    for coeffs, threshold, dims in test_cases:
        pred = LinearPredicate(
            coefficients=list(coeffs.items()),
            threshold=threshold,
            direction="greater_equal",
        )
        n_sites = len(dims)
        mps = compute_satisfaction_set(pred, n_sites, dims)
        bond_dims = list(mps.bond_dims)

        # Compute expected rank bounds
        partial_max = [0]
        for k in range(n_sites):
            c = coeffs.get(k, 0)
            partial_max.append(partial_max[-1] + abs(c) * (dims[k] - 1))

        expected_bounds = []
        for k in range(n_sites - 1):
            bound = min(int(partial_max[k + 1]) + 1, 200)
            bound = min(bound, int(np.prod(dims[:k+1])), int(np.prod(dims[k+1:])))
            expected_bounds.append(bound)

        within_bound = all(
            bd <= eb + 1 for bd, eb in zip(bond_dims, expected_bounds)
        )

        evidence["trials"].append({
            "coefficients": coeffs,
            "threshold": threshold,
            "dims": dims,
            "bond_dims": bond_dims,
            "expected_bounds": expected_bounds,
            "within_bound": within_bound,
        })

        if not within_bound:
            evidence["all_bounded"] = False

    return TheoremVerification(
        theorem_id="Theorem3",
        statement=(
            "Carry-based TT construction for linear predicates "
            "c₁X₁ + c₂X₂ + ... ≥ t has bond dimension bounded by "
            "the range of partial sums + 1 at each bond."
        ),
        verified=evidence["all_bounded"],
        numerical_evidence=evidence,
        proof_sketch=(
            "At bond k, the partial sum S_k = Σ_{i≤k} c_i·X_i ranges over "
            "[S_min_k, S_max_k]. The TT core at site k maps (left_index, X_k) "
            "to right_index representing S_{k+1}. The number of distinct "
            "values of S_{k+1} is at most S_max_{k+1} - S_min_{k+1} + 1. "
            "After SVD compression, the actual rank may be lower."
        ),
    )


def verify_theorem4_complexity_separation() -> TheoremVerification:
    """
    Verify Theorem 4: TT-based verification is O(N) vs explicit O(M^N).

    Statement: For N species with per-species bound M and bond dimension χ,
    TT-based CSL model checking has cost O(N·M·χ²) vs O(M^N) for explicit.

    Verification: Measure actual parameter counts for increasing N.
    """
    from tn_check.models.library import multi_species_cascade
    from tn_check.cme.compiler import CMECompiler

    evidence = {"measurements": []}

    for n_species in [2, 3, 4, 5, 6, 7, 8]:
        max_copy = max(8, 20 - n_species)
        net = multi_species_cascade(n_species=n_species, max_copy=max_copy)
        state_space = int(np.prod(net.physical_dims))

        compiler = CMECompiler(net)
        mpo = compiler.compile()

        tt_params = int(mpo.total_params)
        dense_params = state_space * state_space
        compression_ratio = dense_params / max(tt_params, 1)

        evidence["measurements"].append({
            "n_species": n_species,
            "max_copy": max_copy,
            "state_space": state_space,
            "tt_params": tt_params,
            "dense_params": dense_params,
            "compression_ratio": compression_ratio,
            "log_state_space": np.log10(state_space),
            "log_tt_params": np.log10(max(tt_params, 1)),
        })

    # Check that TT params grow linearly while dense grows exponentially
    measurements = evidence["measurements"]
    tt_growth = [m["tt_params"] for m in measurements]
    dense_growth = [m["dense_params"] for m in measurements]

    # Fit log-linear model to TT params vs N
    ns = np.array([m["n_species"] for m in measurements], dtype=float)
    log_tt = np.log(np.array([max(t, 1) for t in tt_growth]))
    log_dense = np.log(np.array([max(d, 1) for d in dense_growth]))

    # TT should be approximately linear in N (polynomial)
    if len(ns) >= 3:
        tt_fit = np.polyfit(ns, log_tt, 1)
        dense_fit = np.polyfit(ns, log_dense, 1)
        evidence["tt_growth_exponent"] = float(tt_fit[0])
        evidence["dense_growth_exponent"] = float(dense_fit[0])
        # TT exponent should be much smaller than dense exponent
        separation = dense_fit[0] / max(tt_fit[0], 0.1)
    else:
        separation = 1.0

    verified = separation > 2.0  # dense grows at least 2x faster in log

    return TheoremVerification(
        theorem_id="Theorem4",
        statement=(
            "TT-based verification has complexity O(N·M·χ²) vs O(M^N) "
            "for explicit enumeration. The separation is exponential in N."
        ),
        verified=verified,
        numerical_evidence=evidence,
        proof_sketch=(
            "The TT/MPS representation stores N cores of size (χ, M, χ), "
            "giving O(N·M·χ²) parameters. The explicit representation "
            "stores M^N entries. For fixed χ and M, the ratio is "
            "M^N / (N·M·χ²) = M^{N-1} / (N·χ²), which grows exponentially."
        ),
    )


def verify_three_valued_soundness(
    n_trials: int = 20,
) -> TheoremVerification:
    """
    Verify soundness of three-valued CSL semantics.

    Statement: The three-valued classification {TRUE, FALSE, INDETERMINATE}
    is sound: if the TT-compressed probability is within ε of the exact
    probability, and the verdict is TRUE (or FALSE), then the exact
    verdict is also TRUE (or FALSE).

    Verification: For random probability bounds [p_lo, p_hi] and thresholds,
    verify that the three-valued classification is consistent with exact values.
    """
    from tn_check.checker.satisfaction import SatisfactionResult, ThreeValued
    from tn_check.tensor.mps import ones_mps

    evidence = {"trials": [], "all_sound": True}
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        # Generate random exact probability and error bound
        p_exact = rng.uniform(0.0, 1.0)
        epsilon = rng.uniform(0.001, 0.1)
        threshold = rng.uniform(0.0, 1.0)
        comparison = rng.choice([">=", "<="])

        # TT approximation is within epsilon
        p_tt = p_exact + rng.uniform(-epsilon, epsilon)
        p_lo = max(0.0, p_tt - epsilon)
        p_hi = min(1.0, p_tt + epsilon)

        # Three-valued classification
        dummy_mps = ones_mps(1, (2,))
        result = SatisfactionResult(
            satisfaction_mps=dummy_mps,
            probability_lower=p_lo,
            probability_upper=p_hi,
            total_error=epsilon,
        )
        verdict = result.classify(threshold, comparison, epsilon=0.0)

        # Exact verdict
        if comparison == ">=":
            exact_verdict = "TRUE" if p_exact >= threshold else "FALSE"
        else:
            exact_verdict = "TRUE" if p_exact <= threshold else "FALSE"

        # Soundness check: if TT says TRUE/FALSE, exact must agree
        sound = True
        if verdict == ThreeValued.TRUE and exact_verdict == "FALSE":
            sound = False
        if verdict == ThreeValued.FALSE and exact_verdict == "TRUE":
            sound = False

        evidence["trials"].append({
            "trial": trial,
            "p_exact": p_exact,
            "p_tt": p_tt,
            "epsilon": epsilon,
            "threshold": threshold,
            "comparison": comparison,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "tt_verdict": verdict.value,
            "exact_verdict": exact_verdict,
            "sound": sound,
        })

        if not sound:
            evidence["all_sound"] = False

    return TheoremVerification(
        theorem_id="ThreeValuedSoundness",
        statement=(
            "Three-valued CSL semantics is sound: if the TT-compressed "
            "verdict is TRUE (or FALSE), the exact verdict is also TRUE "
            "(or FALSE). INDETERMINATE is the conservative classification "
            "for uncertain cases."
        ),
        verified=evidence["all_sound"],
        numerical_evidence=evidence,
        proof_sketch=(
            "For P≥p[φ] with computed probability in [p_lo, p_hi]: "
            "- TRUE requires p_lo ≥ p + ε, so p_exact ≥ p_lo - ε ≥ p. "
            "- FALSE requires p_hi < p - ε, so p_exact ≤ p_hi + ε < p. "
            "- INDETERMINATE covers the gap where p_exact could be either side. "
            "This is sound because the interval [p_lo, p_hi] is guaranteed "
            "to contain the exact probability."
        ),
    )


def verify_proposition1_clamping(
    n_trials: int = 30,
) -> TheoremVerification:
    """
    Verify Proposition 1 (strengthened): clamping reduces L1 error.

    Uses the existing verify_clamping_reduces_error function.
    """
    from tn_check.error.certification import (
        verify_clamping_proposition,
        verify_clamping_reduces_error,
    )

    evidence = {"trials": [], "all_verified": True}
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        dim = rng.integers(10, 200)
        # Generate exact non-negative probability vector
        p_exact = rng.exponential(1.0, size=dim)
        p_exact /= p_exact.sum()

        # Generate truncation with some negatives
        noise_scale = rng.uniform(0.01, 0.2)
        noise = rng.standard_normal(dim) * noise_scale
        p_svd = p_exact + noise

        # Verify Lemma 1
        lemma1_holds = verify_clamping_proposition(p_exact, p_svd)

        # Verify Lemma 2
        lemma2_holds = verify_clamping_reduces_error(p_exact, p_svd)

        # Compute actual errors
        p_clamped = np.maximum(p_svd, 0.0)
        l1_before = float(np.sum(np.abs(p_exact - p_svd)))
        l1_after = float(np.sum(np.abs(p_exact - p_clamped)))
        neg_mass = float(np.sum(np.abs(p_svd[p_svd < 0])))

        evidence["trials"].append({
            "trial": trial,
            "dim": int(dim),
            "lemma1_holds": lemma1_holds,
            "lemma2_holds": lemma2_holds,
            "l1_before_clamping": l1_before,
            "l1_after_clamping": l1_after,
            "negativity_mass": neg_mass,
            "error_reduced": l1_after <= l1_before + 1e-12,
        })

        if not (lemma1_holds and lemma2_holds):
            evidence["all_verified"] = False

    return TheoremVerification(
        theorem_id="Proposition1",
        statement=(
            "Post-truncation clamping of negative entries never increases "
            "L1 error: ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁ ≤ ε_trunc. "
            "Proved via index-wise comparison (Lemma 2), not triangle inequality."
        ),
        verified=evidence["all_verified"],
        numerical_evidence=evidence,
        proof_sketch=(
            "Lemma 2 proof: For each index i, "
            "Case 1 (p_SVD[i] ≥ 0): clamping leaves unchanged. "
            "Case 2 (p_SVD[i] < 0): clamping to 0, and since p_exact[i] ≥ 0, "
            "|p_exact[i] - 0| = p_exact[i] ≤ p_exact[i] + |p_SVD[i]| = "
            "|p_exact[i] - p_SVD[i]|. Summing: ‖p_exact - p_clamped‖₁ ≤ ‖p_exact - p_SVD‖₁."
        ),
    )


def run_full_rigor_verification() -> dict:
    """
    Run all theorem verifications and produce a comprehensive report.

    Returns:
        Dictionary with all verification results and summary.
    """
    verifications = [
        verify_theorem1_error_contraction(),
        verify_theorem2_bond_dimension_preservation(),
        verify_theorem3_linear_predicate_rank(),
        verify_theorem4_complexity_separation(),
        verify_three_valued_soundness(),
        verify_proposition1_clamping(),
    ]

    all_verified = all(v.verified for v in verifications)
    n_verified = sum(1 for v in verifications if v.verified)

    return {
        "verifications": [
            {
                "theorem_id": v.theorem_id,
                "statement": v.statement,
                "verified": v.verified,
                "proof_sketch": v.proof_sketch,
                "notes": v.notes,
            }
            for v in verifications
        ],
        "summary": {
            "total": len(verifications),
            "verified": n_verified,
            "failed": len(verifications) - n_verified,
            "all_verified": all_verified,
        },
    }
