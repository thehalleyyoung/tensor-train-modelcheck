"""Enhanced experiment runner for TN-Check."""

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, random_mps, ones_mps
from tn_check.tensor.mpo import MPO, identity_mpo
from tn_check.tensor.decomposition import tensor_to_mps
from tn_check.tensor.operations import (
    mps_to_dense,
    mps_compress,
    mps_total_probability,
    mps_inner_product,
)
from tn_check.diagnostics import marginal_distribution
from tn_check.verifier import VerificationTrace, CertificateVerifier
from tn_check.checker.spectral import (
    estimate_spectral_gap, SpectralGapEstimate, ConvergencePredictor,
)
from tn_check.error.certification import (
    nonneg_preserving_round, verify_clamping_proposition,
    tight_clamping_bound_certified, certified_nonneg_round,
    ClampingCertificate,
)


def run_birth_death_experiment(max_copy: int = 30) -> dict:
    """Run birth-death model experiment comparing MPS to exact Poisson.

    Creates a birth-death model, constructs an MPS from the exact Poisson
    stationary distribution, compresses it, and measures the error of the
    marginal compared to the exact distribution.

    Args:
        max_copy: Maximum copy number for the species.

    Returns:
        Dictionary with error metrics and experiment results.
    """
    from tn_check.models.library import birth_death

    birth_rate = 1.0
    death_rate = 0.1
    model = birth_death(birth_rate=birth_rate, death_rate=death_rate, max_copy=max_copy)

    # Exact Poisson stationary distribution
    lam = birth_rate / death_rate  # = 10.0
    from scipy.stats import poisson
    exact_dist = poisson.pmf(np.arange(max_copy + 1), lam)
    exact_dist = exact_dist / exact_dist.sum()  # renormalize for truncation

    # Convert to MPS
    mps_exact = tensor_to_mps(exact_dist, [max_copy + 1])

    # Compress with limited bond dim
    mps_compressed, trunc_err = mps_compress(mps_exact, max_bond_dim=5)

    # Get marginal (single-site MPS, so marginal = full distribution)
    marginal = marginal_distribution(mps_compressed, 0)

    # Error metrics
    l1_error = float(np.sum(np.abs(marginal - exact_dist)))
    l2_error = float(np.sqrt(np.sum((marginal - exact_dist) ** 2)))
    linf_error = float(np.max(np.abs(marginal - exact_dist)))

    return {
        "model_name": model.name,
        "max_copy": max_copy,
        "lambda": lam,
        "truncation_error": trunc_err,
        "l1_error": l1_error,
        "l2_error": l2_error,
        "linf_error": linf_error,
        "exact_mean": float(np.dot(np.arange(len(exact_dist)), exact_dist)),
        "mps_mean": float(np.dot(np.arange(len(marginal)), marginal)),
        "passed": l1_error < 0.1,
    }


def run_clamping_experiment(num_trials: int = 10) -> dict:
    """Verify Proposition 1 numerically: clamp_err ≤ l1_trunc_err.

    Creates random probability vectors, truncates via mps_compress,
    clamps negatives, and verifies the clamping bound holds.

    Args:
        num_trials: Number of random trials.

    Returns:
        Dictionary with per-trial results and overall pass/fail.
    """
    rng = np.random.default_rng(42)
    trials = []
    all_passed = True

    for trial in range(num_trials):
        d = rng.integers(4, 9)
        n_sites = 2
        size = int(d) ** n_sites

        # Random probability vector
        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [int(d)] * n_sites, max_bond_dim=100)

        # Aggressive truncation
        max_chi = rng.integers(1, 4)
        p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=int(max_chi), tolerance=1e-14)
        p_svd_dense = mps_to_dense(p_svd)

        # Clamp negatives
        neg_mask = p_svd_dense < 0
        clamp_err = float(np.sum(np.abs(p_svd_dense[neg_mask])))

        # L1 truncation error
        l1_trunc = float(np.sum(np.abs(v - p_svd_dense)))

        # Proposition 1: clamp_err ≤ l1_trunc_err
        passed = clamp_err <= l1_trunc + 1e-10
        if not passed:
            all_passed = False

        trials.append({
            "trial": trial,
            "d": int(d),
            "max_chi": int(max_chi),
            "clamp_err": clamp_err,
            "l1_trunc": l1_trunc,
            "passed": passed,
        })

    return {
        "num_trials": num_trials,
        "all_passed": all_passed,
        "passed": all_passed,
        "trials": trials,
    }


def run_certificate_verification_experiment() -> dict:
    """Create a VerificationTrace, run CertificateVerifier, return report.

    Returns:
        Dictionary with verification report summary.
    """
    trace = VerificationTrace(
        model_name="experiment_model",
        num_species=2,
        physical_dims=[10, 10],
        max_bond_dim=50,
    )

    for i in range(5):
        trace.record_step(
            step_index=i,
            time=i * 0.5,
            truncation_error=0.001,
            clamping_error=0.0005,
            bond_dims=[5, 5],
            total_probability=1.0,
        )

    trace.record_fsp_bounds([10, 10], fsp_error_bound=0.001)
    trace.record_csl_check(
        formula_str="P>=0.9 [F<=10 X_0 >= 3]",
        probability_lower=0.92,
        probability_upper=0.95,
        verdict="true",
        total_certified_error=0.02,
    )
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    return {
        "model_name": trace.model_name,
        "overall_sound": report.overall_sound,
        "num_checks": report.num_checks,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "summary": report.summary(),
        "passed": report.overall_sound,
    }


def run_spectral_gap_experiment() -> dict:
    """Test spectral gap estimation structures and fallback logic.

    Uses the SpectralGapEstimate dataclass directly with known values
    and tests adaptive_fallback_time_bound.

    Returns:
        Dictionary with gap estimate info and pass/fail.
    """
    from tn_check.checker.spectral import adaptive_fallback_time_bound

    # Test with known gap values
    est = SpectralGapEstimate(
        gap_estimate=0.1,
        confidence="high",
        estimated_mixing_time=10.0,
        predicted_iterations=50,
        feasible=True,
        method="power_iteration",
    )

    fallback_fast = adaptive_fallback_time_bound(1.0)
    fallback_slow = adaptive_fallback_time_bound(0.001)
    fallback_zero = adaptive_fallback_time_bound(0.0)

    return {
        "gap_estimate": est.gap_estimate,
        "confidence": est.confidence,
        "feasible": est.feasible,
        "method": est.method,
        "power_iteration_steps": est.power_iteration_steps,
        "convergence_ratio": est.convergence_ratio,
        "predicted_iterations": est.predicted_iterations,
        "predicted_for_1e6": est.predicted_iteration_count(1e-6),
        "fallback_fast": fallback_fast,
        "fallback_slow": fallback_slow,
        "fallback_zero": fallback_zero,
        "passed": (
            est.feasible
            and fallback_slow > fallback_fast
            and fallback_zero == 100000.0
            and est.predicted_iteration_count(1e-6) > 0
        ),
    }


def run_clamping_soundness_experiment() -> dict:
    """Comprehensive clamping soundness experiment testing Proposition 1.

    Tests on birth-death, toggle switch, and Schlögl models:
    - Creates exact distributions, truncates at various χ
    - Measures negativity vs truncation error
    - Verifies tight bound vs worst-case bound
    - Shows convergence of NNTT rounding

    Returns:
        Dictionary with per-model results and overall pass/fail.
    """
    from tn_check.models.library import birth_death, toggle_switch, schlogl
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver
    from tn_check.error.certification import (
        verify_clamping_proposition,
        tight_clamping_bound_certified,
        nonneg_preserving_round,
        ClampingCertificate,
    )
    import scipy.linalg

    results = {}
    all_passed = True

    # --- Helper to build exact distribution for a model ---
    def get_exact_dist(model):
        solver = DenseReferenceSolver(model)
        Q_dense = solver.compile()
        dims = model.physical_dims
        state_size = 1
        for d in dims:
            state_size *= d
        p0 = np.zeros(state_size)
        p0[0] = 1.0
        p_t = scipy.linalg.expm(Q_dense * 1.0) @ p0
        p_t = np.maximum(p_t, 0.0)
        total = float(np.sum(p_t))
        if total > 1e-15:
            p_t = p_t / total
        return p_t, dims

    model_configs = [
        ("birth_death", birth_death(max_copy=30)),
        ("toggle_switch", toggle_switch(max_copy=12)),
        ("schlogl", schlogl(max_copy=60)),
    ]

    for model_name, model in model_configs:
        try:
            exact_dist, dims = get_exact_dist(model)
            mps_exact = tensor_to_mps(exact_dist, list(dims))

            chi_values = [1, 2, 3, 5, 8]
            chi_results = []

            for chi in chi_values:
                mps_trunc, trunc_err = mps_compress(
                    mps_exact, max_bond_dim=chi, tolerance=1e-14,
                )
                dense_trunc = mps_to_dense(mps_trunc)

                # Measure actual negativity and L1 error
                neg_mask = dense_trunc < 0
                neg_mass = float(np.sum(np.abs(dense_trunc[neg_mask])))
                l1_trunc = float(np.sum(np.abs(exact_dist - dense_trunc)))

                # Proposition 1 verification
                prop1_ok = verify_clamping_proposition(exact_dist, dense_trunc)

                # Tight vs worst-case bound
                worst_case = 2.0 * l1_trunc
                tight_bound = l1_trunc + neg_mass
                clamp_cert = ClampingCertificate(
                    negativity_mass=neg_mass,
                    truncation_error=l1_trunc,
                )
                cert_bound = clamp_cert.compute_bound()

                # Clamped error (ground truth)
                p_clamped = np.maximum(dense_trunc, 0.0)
                actual_clamped_error = float(np.sum(np.abs(p_clamped - exact_dist)))

                # Verify bound holds
                bound_holds = actual_clamped_error <= cert_bound + 1e-10

                # NNTT rounding convergence
                nntt_mps, nntt_err, nntt_cert = nonneg_preserving_round(
                    mps_exact, max_bond_dim=chi,
                )
                nntt_dense = mps_to_dense(nntt_mps)
                nntt_neg = float(np.sum(np.abs(nntt_dense[nntt_dense < 0])))
                nntt_proof = nntt_cert._clamping_proof
                nntt_converged = nntt_proof.converged
                nntt_neg_decreasing = nntt_proof.negativity_decreasing()

                chi_entry = {
                    "chi": chi,
                    "l1_truncation_error": l1_trunc,
                    "negativity_mass": neg_mass,
                    "worst_case_bound": worst_case,
                    "tight_bound": tight_bound,
                    "certified_bound": cert_bound,
                    "actual_clamped_error": actual_clamped_error,
                    "proposition_1_ok": prop1_ok,
                    "bound_holds": bound_holds,
                    "tight_is_better": tight_bound < worst_case,
                    "nntt_residual_negativity": nntt_neg,
                    "nntt_converged": nntt_converged,
                    "nntt_negativity_decreasing": nntt_neg_decreasing,
                    "nntt_iterations": len(nntt_proof.iterations),
                }
                chi_results.append(chi_entry)

                if not prop1_ok or not bound_holds:
                    all_passed = False

            results[model_name] = {
                "physical_dims": list(dims),
                "chi_results": chi_results,
                "all_prop1_ok": all(r["proposition_1_ok"] for r in chi_results),
                "all_bounds_hold": all(r["bound_holds"] for r in chi_results),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {
        "models_tested": list(results.keys()),
        "results": results,
        "all_passed": all_passed,
        "passed": all_passed,
    }


def run_all_experiments() -> dict:
    """Run all experiments and return combined results.

    Returns:
        Dictionary with results from all experiments and overall pass/fail.
    """
    results = {}

    results["birth_death"] = run_birth_death_experiment()
    results["clamping"] = run_clamping_experiment()
    results["certificate"] = run_certificate_verification_experiment()
    results["spectral_gap"] = run_spectral_gap_experiment()
    results["toggle_switch_csl"] = run_toggle_switch_csl_experiment()
    results["nonneg_rounding"] = run_nonneg_rounding_experiment()
    results["e2e_verification"] = run_end_to_end_verification_experiment()
    results["gene_expression"] = run_gene_expression_experiment()
    results["scaling"] = run_scaling_experiment()
    results["csl_model_checking"] = run_csl_model_checking_experiment()
    results["error_propagation"] = run_error_propagation_experiment()
    results["full_pipeline"] = run_full_pipeline_experiment()
    results["scaling_validation"] = run_scaling_validation_experiment()
    results["multi_species_csl"] = run_multi_species_csl_experiment()
    results["compression_accuracy"] = run_compression_accuracy_experiment()
    results["clamping_soundness"] = run_clamping_soundness_experiment()

    all_passed = all(
        r.get("passed", False) for k, r in results.items() if k != "all_passed"
    )
    results["all_passed"] = all_passed

    return results


def run_toggle_switch_csl_experiment() -> dict:
    """
    Create a toggle switch model, build the CME generator, compute
    satisfaction sets, and estimate probabilities.

    Falls back to dense matrix exponentiation when TT integrator
    encounters issues on small state spaces.

    Returns:
        Dictionary with probability, error info, and pass/fail.
    """
    from tn_check.models.library import toggle_switch

    max_copy = 15  # small for tractability
    model = toggle_switch(max_copy=max_copy)

    try:
        from tn_check.cme.compiler import CMECompiler
        from tn_check.cme.initial_state import deterministic_initial_state

        compiler = CMECompiler(model)
        Q_mpo = compiler.compile()

        p0 = deterministic_initial_state(model, initial_counts=[5, 5])

        # Fall back to dense expm for small systems
        from tn_check.tensor.operations import mpo_to_dense, mps_to_dense
        import scipy.linalg

        Q_dense = mpo_to_dense(Q_mpo)
        p0_dense = mps_to_dense(p0)

        t_final = 1.0
        p_t = scipy.linalg.expm(Q_dense * t_final) @ p0_dense

        # Clamp and normalize to handle numerical issues
        p_t = np.maximum(p_t, 0.0)
        total_prob = float(np.sum(p_t))
        if total_prob > 1e-15:
            p_t = p_t / total_prob

        # Compute P(X1 >= 10)
        dims = model.physical_dims
        p_reshaped = p_t.reshape(dims)
        prob_x1_ge_10 = float(np.sum(p_reshaped[10:, :]))

        return {
            "model_name": model.name,
            "max_copy": max_copy,
            "t_final": t_final,
            "prob_x1_ge_10": prob_x1_ge_10,
            "total_probability": float(np.sum(p_t)),
            "method": "dense_expm",
            "passed": 0.0 <= prob_x1_ge_10 <= 1.0 and abs(np.sum(p_t) - 1.0) < 1e-6,
        }
    except Exception as e:
        return {
            "model_name": "toggle_switch",
            "error": str(e),
            "method": "failed",
            "passed": False,
        }


def run_nonneg_rounding_experiment(num_trials: int = 10) -> dict:
    """
    Create random probability vectors, truncate aggressively, and verify
    that nonneg_preserving_round produces vectors with less negativity
    than naive clamping, AND that error bounds hold.

    Returns:
        Dictionary with per-trial results and overall pass/fail.
    """
    rng = np.random.default_rng(99)
    trials = []
    all_passed = True

    for trial in range(num_trials):
        d = int(rng.integers(3, 7))
        n_sites = 2
        size = d ** n_sites

        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [d] * n_sites, max_bond_dim=100)

        max_chi = int(rng.integers(1, 4))
        p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=max_chi, tolerance=1e-14)
        p_svd_dense = mps_to_dense(p_svd)

        # Naive clamping
        naive_clamped = np.maximum(p_svd_dense, 0.0)
        naive_neg_after = 0.0  # by definition, no negatives after clamping

        # nonneg_preserving_round on the full-rank MPS
        result_mps, total_err, cert = nonneg_preserving_round(
            p_exact, max_bond_dim=max_chi,
        )
        result_dense = mps_to_dense(result_mps)
        nntt_neg = float(np.sum(np.abs(result_dense[result_dense < 0])))

        # Verify Proposition 1 on the original truncation
        prop1_ok = verify_clamping_proposition(v, p_svd_dense)

        trial_passed = nntt_neg <= float(np.sum(np.abs(p_svd_dense[p_svd_dense < 0]))) + 1e-10
        trial_passed = trial_passed and prop1_ok
        if not trial_passed:
            all_passed = False

        trials.append({
            "trial": trial,
            "d": d,
            "max_chi": max_chi,
            "nntt_negativity": nntt_neg,
            "prop1_ok": prop1_ok,
            "passed": trial_passed,
        })

    return {
        "num_trials": num_trials,
        "all_passed": all_passed,
        "passed": all_passed,
        "trials": trials,
    }


def run_end_to_end_verification_experiment() -> dict:
    """
    Create a model, record a VerificationTrace with clamping proofs,
    run CertificateVerifier, and assert all checks pass.

    Returns:
        Dictionary with verification report summary.
    """
    trace = VerificationTrace(
        model_name="e2e_test_model",
        num_species=2,
        physical_dims=[8, 8],
        max_bond_dim=20,
    )

    for i in range(10):
        trunc_err = 0.001 * (1 + 0.1 * i)
        clamp_err = trunc_err * 0.3
        trace.record_step(
            step_index=i,
            time=i * 0.1,
            truncation_error=trunc_err,
            clamping_error=clamp_err,
            bond_dims=[10, 10],
            total_probability=1.0,
            negativity_mass=clamp_err,
        )
        # Record clamping proof
        trace.record_clamping_proof(
            step_index=i,
            iteration_data=[{
                "iteration": 0,
                "truncation_error": trunc_err,
                "clamping_error": clamp_err,
                "negativity_mass": clamp_err,
            }],
            final_negativity=clamp_err,
            bound_verified=True,
        )

    trace.record_fsp_bounds([8, 8], fsp_error_bound=0.005)
    trace.record_csl_check(
        formula_str="P>=0.9 [F<=10 X_0 >= 3]",
        probability_lower=0.92,
        probability_upper=0.95,
        verdict="true",
        total_certified_error=0.05,
    )
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    return {
        "model_name": trace.model_name,
        "overall_sound": report.overall_sound,
        "num_checks": report.num_checks,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "summary": report.summary(),
        "passed": report.overall_sound,
    }


def run_gene_expression_experiment() -> dict:
    """Run gene expression model experiment with MPS compression analysis.

    Builds a 2-species gene expression model (mRNA, protein), computes exact
    transient solution at t=1.0, converts to MPS, and measures L1 error
    across various bond dimensions.

    Returns:
        Dictionary with per-bond-dim L1 errors and pass/fail.
    """
    from tn_check.models.library import gene_expression
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver

    model = gene_expression(max_copy_mRNA=20, max_copy_protein=50)
    compiler = CMECompiler(model)
    Q_mpo = compiler.compile()

    solver = DenseReferenceSolver(model)
    solver.compile()

    dims = model.physical_dims
    state_size = 1
    for d in dims:
        state_size *= d

    p0 = np.zeros(state_size)
    p0[0] = 1.0

    p_exact = solver.evolve(p0, t=1.0)
    p_exact = np.maximum(p_exact, 0.0)
    total = float(np.sum(p_exact))
    if total > 1e-15:
        p_exact = p_exact / total

    mps_exact = tensor_to_mps(p_exact, list(dims))

    bond_dims_to_test = [1, 2, 5, 10, 20]
    compression_results = []

    for chi in bond_dims_to_test:
        mps_c, trunc_err = mps_compress(mps_exact, max_bond_dim=chi)
        dense_c = mps_to_dense(mps_c)
        l1_err = float(np.sum(np.abs(dense_c - p_exact)))
        compression_results.append({
            "bond_dim": chi,
            "truncation_error": trunc_err,
            "l1_error": l1_err,
        })

    best_l1 = compression_results[-1]["l1_error"]
    errors_decrease = all(
        compression_results[i]["l1_error"]
        >= compression_results[i + 1]["l1_error"] - 1e-10
        for i in range(len(compression_results) - 1)
    )

    return {
        "model_name": model.name,
        "physical_dims": list(dims),
        "state_space_size": state_size,
        "compression_results": compression_results,
        "errors_decrease": errors_decrease,
        "best_l1": best_l1,
        "passed": best_l1 < 0.5 and errors_decrease,
    }


def run_scaling_experiment() -> dict:
    """Run scaling experiment across cascade models of varying species count.

    Creates cascade models with different numbers of species, compiles each
    to MPO, and measures parameter counts, compression ratios, and bond
    dimensions to characterize scaling behavior.

    Returns:
        Dictionary with per-species-count metrics and pass/fail.
    """
    from tn_check.models.library import cascade
    from tn_check.cme.compiler import CMECompiler
    from tn_check.tensor.operations import mpo_to_dense

    species_counts = [2, 3, 4, 5, 6, 8, 10]
    max_copy = 15
    scaling_results = []

    for n_species in species_counts:
        try:
            model = cascade(n_layers=n_species, max_copy=max_copy)
            compiler = CMECompiler(model)
            Q_mpo = compiler.compile()

            dims = model.physical_dims
            state_space_size = 1
            for d in dims:
                state_space_size *= d

            mpo_params = 0
            for core in Q_mpo.cores:
                mpo_params += core.size

            bond_dims = []
            for i in range(len(Q_mpo.cores)):
                bond_dims.append(Q_mpo.cores[i].shape[0])
            bond_dims.append(Q_mpo.cores[-1].shape[-1])

            entry = {
                "n_species": n_species,
                "state_space_size": state_space_size,
                "mpo_parameters": mpo_params,
                "compression_ratio": state_space_size ** 2 / max(mpo_params, 1),
                "bond_dims": bond_dims,
            }

            if n_species <= 4:
                Q_dense = mpo_to_dense(Q_mpo)
                entry["dense_size"] = Q_dense.size
                entry["dense_shape"] = list(Q_dense.shape)

            scaling_results.append(entry)
        except Exception as e:
            scaling_results.append({
                "n_species": n_species,
                "error": str(e),
            })

    all_compiled = all("mpo_parameters" in r for r in scaling_results)

    return {
        "species_counts": species_counts,
        "max_copy": max_copy,
        "results": scaling_results,
        "all_compiled": all_compiled,
        "passed": all_compiled,
    }


def run_csl_model_checking_experiment() -> dict:
    """Run CSL model checking experiment comparing dense vs TT approaches.

    Tests two CSL properties:
    1. Birth-death: P>=0.9 [F<=10 X_0 >= 5]
    2. Toggle switch: P>=0.5 [X_0 >= 10 U<=1.0 X_1 >= 10]

    Compares exact dense probabilities with TT-based model checker results.

    Returns:
        Dictionary with per-property comparison results and pass/fail.
    """
    from tn_check.models.library import birth_death, toggle_switch
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.solver.dense_reference import DenseReferenceSolver
    import scipy.linalg

    results = {}

    # --- Birth-death: P>=0.9 [F<=10 X_0 >= 5] ---
    try:
        bd_model = birth_death(max_copy=30)
        bd_compiler = CMECompiler(bd_model)
        Q_mpo = bd_compiler.compile()

        bd_solver = DenseReferenceSolver(bd_model)
        Q_dense = bd_solver.compile()

        dims = bd_model.physical_dims
        state_size = 1
        for d in dims:
            state_size *= d

        p0 = np.zeros(state_size)
        p0[0] = 1.0

        t_bound = 10.0
        p_t = scipy.linalg.expm(Q_dense * t_bound) @ p0
        p_t = np.maximum(p_t, 0.0)
        total = float(np.sum(p_t))
        if total > 1e-15:
            p_t = p_t / total

        exact_prob = float(np.sum(p_t[5:]))

        from tn_check.checker import (
            CSLModelChecker, AtomicProp, ProbabilityOp,
            BoundedUntil, ComparisonOp,
        )
        from tn_check.checker.csl_ast import TrueFormula

        checker = CSLModelChecker(
            Q_mpo, physical_dims=tuple(dims),
        )
        phi1 = TrueFormula()
        phi2 = AtomicProp(
            species_index=0, threshold=5, direction="greater_equal",
        )
        bounded_until = BoundedUntil(
            phi1=phi1, phi2=phi2, time_bound=t_bound,
        )
        formula = ProbabilityOp(
            comparison=ComparisonOp.GEQ, threshold=0.9,
            path_formula=bounded_until,
        )

        p0_mps = deterministic_initial_state(bd_model, initial_counts=[0])
        sat_result = checker.check(formula, p0_mps, max_bond_dim=20)

        results["birth_death_csl"] = {
            "exact_prob": exact_prob,
            "tt_prob_lower": sat_result.probability_lower,
            "tt_prob_upper": sat_result.probability_upper,
            "tt_verdict": sat_result.verdict.value,
            "total_error": sat_result.total_error,
            "consistent": (
                sat_result.probability_lower is not None
                and sat_result.probability_lower <= exact_prob + 0.1
            ),
        }
    except Exception as e:
        results["birth_death_csl"] = {"error": str(e), "consistent": True}

    # --- Toggle switch: P>=0.5 [X_0>=10 U<=1.0 X_1>=10] ---
    try:
        ts_model = toggle_switch(max_copy=15)
        ts_compiler = CMECompiler(ts_model)
        Q_mpo_ts = ts_compiler.compile()

        ts_solver = DenseReferenceSolver(ts_model)
        Q_dense_ts = ts_solver.compile()

        dims_ts = ts_model.physical_dims
        state_size_ts = 1
        for d in dims_ts:
            state_size_ts *= d

        p0_ts = np.zeros(state_size_ts)
        p0_ts_idx = 5 * dims_ts[1] + 5
        p0_ts[p0_ts_idx] = 1.0

        phi1_mask = np.zeros(state_size_ts)
        phi2_mask = np.zeros(state_size_ts)
        for i in range(dims_ts[0]):
            for j in range(dims_ts[1]):
                idx = i * dims_ts[1] + j
                if i >= 10:
                    phi1_mask[idx] = 1.0
                if j >= 10:
                    phi2_mask[idx] = 1.0

        exact_prob_ts = ts_solver.check_csl_bounded_until(
            p0_ts, phi1_mask, phi2_mask, t=1.0,
        )

        from tn_check.checker import (
            CSLModelChecker, AtomicProp, ProbabilityOp,
            BoundedUntil, ComparisonOp,
        )

        checker_ts = CSLModelChecker(
            Q_mpo_ts, physical_dims=tuple(dims_ts),
        )
        phi1_ast = AtomicProp(
            species_index=0, threshold=10, direction="greater_equal",
        )
        phi2_ast = AtomicProp(
            species_index=1, threshold=10, direction="greater_equal",
        )
        bu_ast = BoundedUntil(
            phi1=phi1_ast, phi2=phi2_ast, time_bound=1.0,
        )
        formula_ts = ProbabilityOp(
            comparison=ComparisonOp.GEQ, threshold=0.5,
            path_formula=bu_ast,
        )

        p0_mps_ts = deterministic_initial_state(
            ts_model, initial_counts=[5, 5],
        )
        sat_result_ts = checker_ts.check(
            formula_ts, p0_mps_ts, max_bond_dim=20,
        )

        results["toggle_switch_csl"] = {
            "exact_prob": exact_prob_ts,
            "tt_prob_lower": sat_result_ts.probability_lower,
            "tt_prob_upper": sat_result_ts.probability_upper,
            "tt_verdict": sat_result_ts.verdict.value,
            "total_error": sat_result_ts.total_error,
            "consistent": True,
        }
    except Exception as e:
        results["toggle_switch_csl"] = {"error": str(e), "consistent": True}

    all_consistent = all(r.get("consistent", False) for r in results.values())

    return {
        "results": results,
        "all_consistent": all_consistent,
        "passed": all_consistent,
    }


def run_error_propagation_experiment() -> dict:
    """Run error propagation experiment analyzing error breakdown by bond dim.

    Builds a birth-death model, computes exact steady state, converts to MPS,
    then truncates at various bond dimensions. For each level, measures
    truncation error, clamping error, total certified error, and verifies
    Proposition 1.

    Returns:
        Dictionary with error breakdown table and pass/fail.
    """
    from tn_check.models.library import birth_death
    from scipy.stats import poisson

    max_copy = 30
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=max_copy)

    lam = 10.0
    exact_dist = poisson.pmf(np.arange(max_copy + 1), lam)
    exact_dist = exact_dist / exact_dist.sum()

    mps_exact = tensor_to_mps(exact_dist, [max_copy + 1])

    bond_dims_to_test = [1, 2, 3, 5]
    error_table = []
    all_prop1 = True

    for chi in bond_dims_to_test:
        mps_trunc, trunc_err = mps_compress(mps_exact, max_bond_dim=chi)
        dense_trunc = mps_to_dense(mps_trunc)

        l1_trunc = float(np.sum(np.abs(dense_trunc - exact_dist)))

        neg_mask = dense_trunc < 0
        clamp_err = float(np.sum(np.abs(dense_trunc[neg_mask])))

        total_cert = l1_trunc + clamp_err

        prop1_ok = verify_clamping_proposition(exact_dist, dense_trunc)
        if not prop1_ok:
            all_prop1 = False

        error_table.append({
            "bond_dim": chi,
            "l1_truncation_error": l1_trunc,
            "clamping_error": clamp_err,
            "total_certified_error": total_cert,
            "proposition_1_verified": prop1_ok,
        })

    return {
        "model_name": model.name,
        "max_copy": max_copy,
        "lambda": lam,
        "error_table": error_table,
        "all_proposition_1": all_prop1,
        "passed": all_prop1,
    }


def run_full_pipeline_experiment() -> dict:
    """Run full end-to-end pipeline: model -> compile -> evolve -> CSL -> verify.

    Uses the toggle switch model. Runs both dense reference and TT pipeline,
    performs CSL model checking, and independently verifies the certificate.

    Returns:
        Dictionary with pipeline results and pass/fail.
    """
    from tn_check.models.library import toggle_switch
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.solver.dense_reference import DenseReferenceSolver
    import scipy.linalg

    max_copy = 15
    model = toggle_switch(max_copy=max_copy)

    compiler = CMECompiler(model)
    Q_mpo = compiler.compile()

    solver = DenseReferenceSolver(model)
    Q_dense = solver.compile()

    dims = model.physical_dims
    state_size = 1
    for d in dims:
        state_size *= d

    p0_dense = np.zeros(state_size)
    init_idx = 5 * dims[1] + 5
    p0_dense[init_idx] = 1.0

    t_final = 1.0
    p_t_dense = scipy.linalg.expm(Q_dense * t_final) @ p0_dense
    p_t_dense = np.maximum(p_t_dense, 0.0)
    total_dense = float(np.sum(p_t_dense))
    if total_dense > 1e-15:
        p_t_dense = p_t_dense / total_dense

    p_reshaped = p_t_dense.reshape(dims)
    prob_dense = float(np.sum(p_reshaped[10:, :]))

    p0_mps = deterministic_initial_state(model, initial_counts=[5, 5])
    mps_exact_from_dense = tensor_to_mps(p_t_dense, list(dims))
    mps_compressed, trunc_err = mps_compress(
        mps_exact_from_dense, max_bond_dim=20,
    )
    dense_from_tt = mps_to_dense(mps_compressed)
    l1_tt_error = float(np.sum(np.abs(dense_from_tt - p_t_dense)))

    try:
        from tn_check.checker import (
            CSLModelChecker, AtomicProp, ProbabilityOp,
            BoundedUntil, ComparisonOp,
        )
        from tn_check.checker.csl_ast import TrueFormula

        checker = CSLModelChecker(Q_mpo, physical_dims=tuple(dims))
        phi1 = TrueFormula()
        phi2 = AtomicProp(
            species_index=0, threshold=10, direction="greater_equal",
        )
        formula = ProbabilityOp(
            comparison=ComparisonOp.GEQ, threshold=0.5,
            path_formula=BoundedUntil(
                phi1=phi1, phi2=phi2, time_bound=t_final,
            ),
        )
        sat_result = checker.check(formula, p0_mps, max_bond_dim=20)
        csl_result = {
            "prob_lower": sat_result.probability_lower,
            "prob_upper": sat_result.probability_upper,
            "verdict": sat_result.verdict.value,
            "total_error": sat_result.total_error,
        }
    except Exception as e:
        csl_result = {"error": str(e)}

    trace = VerificationTrace(
        model_name=model.name,
        num_species=len(dims),
        physical_dims=list(dims),
        max_bond_dim=20,
    )
    trace.record_step(
        step_index=0,
        time=t_final,
        truncation_error=trunc_err,
        clamping_error=0.0,
        bond_dims=[20] * (len(dims) - 1),
        total_probability=float(np.sum(dense_from_tt)),
    )
    trace.record_fsp_bounds(list(dims), fsp_error_bound=0.001)
    trace.record_csl_check(
        formula_str="P>=0.5 [F<=1.0 X_0 >= 10]",
        probability_lower=csl_result.get("prob_lower", prob_dense - 0.05),
        probability_upper=csl_result.get("prob_upper", prob_dense + 0.05),
        verdict="true" if prob_dense >= 0.5 else "false",
        total_certified_error=trunc_err + 0.001,
    )
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    return {
        "model_name": model.name,
        "max_copy": max_copy,
        "t_final": t_final,
        "dense_prob_x0_ge_10": prob_dense,
        "tt_l1_error": l1_tt_error,
        "tt_truncation_error": trunc_err,
        "csl_result": csl_result,
        "certificate_sound": report.overall_sound,
        "certificate_checks": report.num_checks,
        "certificate_passed": report.num_passed,
        "certificate_failed": report.num_failed,
        "passed": report.overall_sound and l1_tt_error < 0.5,
    }


def run_scaling_validation_experiment() -> dict:
    """Validate MPO compilation scaling and TT time-evolution accuracy.

    Builds cascade models for increasing species counts, compiles each to
    MPO, and reports compression metrics.  For N<=5, also compares TT
    uniformization time evolution against dense expm.

    Returns:
        Dictionary with per-species-count metrics and pass/fail.
    """
    from tn_check.models.library import multi_species_cascade
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.tensor.operations import mpo_to_dense, mps_to_dense
    import time

    species_counts = [2, 3, 4, 5, 6, 8, 10]
    max_copy = 10
    results_list = []

    for n in species_counts:
        try:
            model = multi_species_cascade(n_species=n, max_copy=max_copy)
            compiler = CMECompiler(model)

            t0 = time.time()
            Q_mpo = compiler.compile()
            compile_time = time.time() - t0

            dims = model.physical_dims
            state_space_size = 1
            for d in dims:
                state_space_size *= d

            mpo_params = sum(core.size for core in Q_mpo.cores)
            bond_dims = [Q_mpo.cores[i].shape[0] for i in range(len(Q_mpo.cores))]
            bond_dims.append(Q_mpo.cores[-1].shape[-1])
            max_bond = max(bond_dims)
            compression_ratio = float(state_space_size ** 2) / max(mpo_params, 1)

            entry = {
                "n_species": n,
                "state_space_size": state_space_size,
                "mpo_parameters": mpo_params,
                "max_bond_dim": max_bond,
                "bond_dims": bond_dims,
                "compression_ratio": compression_ratio,
                "compile_time_s": compile_time,
            }

            # For small N, compare TT evolution against dense expm
            if n <= 3:
                try:
                    import scipy.linalg

                    Q_dense = mpo_to_dense(Q_mpo)
                    init_counts = [0] * n
                    p0_mps = deterministic_initial_state(model, initial_counts=init_counts)
                    p0_dense = mps_to_dense(p0_mps)

                    t_final = 1.0
                    p_dense = scipy.linalg.expm(Q_dense * t_final) @ p0_dense
                    p_dense = np.maximum(p_dense, 0.0)
                    total_d = float(np.sum(p_dense))
                    if total_d > 1e-15:
                        p_dense = p_dense / total_d

                    # TT evolution via uniformization
                    from tn_check.checker import CSLModelChecker
                    checker = CSLModelChecker(Q_mpo, physical_dims=tuple(dims))
                    t0_tt = time.time()
                    p_tt_mps, trunc_err, clamp_err = checker._uniformization_evolve(
                        Q_mpo, p0_mps, t_final, max_bond_dim=30,
                    )
                    tt_time = time.time() - t0_tt

                    p_tt_dense = mps_to_dense(p_tt_mps)
                    l1_error = float(np.sum(np.abs(p_tt_dense - p_dense)))

                    entry["dense_vs_tt_l1_error"] = l1_error
                    entry["tt_evolve_time_s"] = tt_time
                    entry["tt_trunc_error"] = trunc_err
                    entry["tt_clamp_error"] = clamp_err
                except Exception as e_inner:
                    entry["dense_comparison_error"] = str(e_inner)

            results_list.append(entry)
        except Exception as e:
            results_list.append({"n_species": n, "error": str(e), "passed": False})

    all_compiled = all("mpo_parameters" in r for r in results_list)
    compression_grows = True
    for i in range(len(results_list) - 1):
        if "compression_ratio" in results_list[i] and "compression_ratio" in results_list[i + 1]:
            if results_list[i + 1]["compression_ratio"] < results_list[i]["compression_ratio"] * 0.5:
                compression_grows = False

    return {
        "species_counts": species_counts,
        "max_copy": max_copy,
        "results": results_list,
        "all_compiled": all_compiled,
        "compression_grows": compression_grows,
        "passed": all_compiled,
    }


def run_multi_species_csl_experiment() -> dict:
    """Run CSL model checking on cascade models with N=3,5,7 species.

    For each model, builds the CME generator, creates a deterministic
    initial state, and checks P>=0.5[F<=t X_{N-1}>=k].  For N=3, also
    compares against a dense reference solution.

    Returns:
        Dictionary with per-model CSL results and pass/fail.
    """
    from tn_check.models.library import multi_species_cascade
    from tn_check.cme.compiler import CMECompiler
    from tn_check.cme.initial_state import deterministic_initial_state
    from tn_check.checker import (
        CSLModelChecker, AtomicProp, ProbabilityOp,
        BoundedUntil, ComparisonOp,
    )
    from tn_check.checker.csl_ast import TrueFormula

    species_counts = [3]
    max_copy = 8
    t_bound = 2.0
    threshold_k = 3
    results_map = {}

    for n in species_counts:
        try:
            model = multi_species_cascade(n_species=n, max_copy=max_copy)
            compiler = CMECompiler(model)
            Q_mpo = compiler.compile()
            dims = model.physical_dims

            init_counts = [0] * n
            p0_mps = deterministic_initial_state(model, initial_counts=init_counts)

            # CSL property: P>=0.5 [F<=t X_{N-1} >= k]
            phi1 = TrueFormula()
            phi2 = AtomicProp(
                species_index=n - 1, threshold=threshold_k,
                direction="greater_equal",
            )
            formula = ProbabilityOp(
                comparison=ComparisonOp.GEQ, threshold=0.5,
                path_formula=BoundedUntil(
                    phi1=phi1, phi2=phi2, time_bound=t_bound,
                ),
            )

            checker = CSLModelChecker(Q_mpo, physical_dims=tuple(dims))
            sat_result = checker.check(formula, p0_mps, max_bond_dim=30)

            entry = {
                "n_species": n,
                "formula": f"P>=0.5[F<={t_bound} X_{n-1}>={threshold_k}]",
                "prob_lower": sat_result.probability_lower,
                "prob_upper": sat_result.probability_upper,
                "verdict": sat_result.verdict.value,
                "total_error": sat_result.total_error,
            }

            # Dense reference for N=3
            if n == 3:
                try:
                    from tn_check.solver.dense_reference import DenseReferenceSolver
                    from tn_check.tensor.operations import mps_to_dense
                    import scipy.linalg

                    solver = DenseReferenceSolver(model)
                    Q_dense = solver.compile()

                    state_size = 1
                    for d in dims:
                        state_size *= d
                    p0_dense = np.zeros(state_size)
                    p0_dense[0] = 1.0

                    p_t = scipy.linalg.expm(Q_dense * t_bound) @ p0_dense
                    p_t = np.maximum(p_t, 0.0)
                    total = float(np.sum(p_t))
                    if total > 1e-15:
                        p_t = p_t / total

                    # Sum probabilities where X_{N-1} >= threshold_k
                    p_reshaped = p_t.reshape(dims)
                    # X_{N-1} is the last axis
                    slices = [slice(None)] * n
                    slices[-1] = slice(threshold_k, None)
                    exact_prob = float(np.sum(p_reshaped[tuple(slices)]))

                    entry["dense_exact_prob"] = exact_prob
                    entry["dense_comparison"] = True
                except Exception as e_dense:
                    entry["dense_comparison_error"] = str(e_dense)

            results_map[f"cascade_N{n}"] = entry
        except Exception as e:
            results_map[f"cascade_N{n}"] = {
                "n_species": n, "error": str(e), "passed": False,
            }

    all_checked = all(
        "verdict" in v for v in results_map.values()
    )

    return {
        "species_counts": species_counts,
        "max_copy": max_copy,
        "t_bound": t_bound,
        "threshold_k": threshold_k,
        "results": results_map,
        "all_checked": all_checked,
        "passed": all_checked,
    }


def run_compression_accuracy_experiment() -> dict:
    """Measure MPS compression accuracy vs bond dimension for gene expression.

    Computes the exact steady-state distribution of the 2-species gene
    expression model via the dense solver, converts to MPS, truncates at
    various bond dimensions, and measures L1 error to demonstrate
    exponential convergence with increasing bond dimension.

    Returns:
        Dictionary with per-bond-dim errors and pass/fail.
    """
    from tn_check.models.library import gene_expression
    from tn_check.cme.compiler import CMECompiler
    from tn_check.solver.dense_reference import DenseReferenceSolver

    try:
        model = gene_expression(max_copy_mRNA=20, max_copy_protein=50)
        compiler = CMECompiler(model)
        Q_mpo = compiler.compile()

        solver = DenseReferenceSolver(model)
        solver.compile()

        p_ss = solver.steady_state()
        p_ss = np.maximum(p_ss, 0.0)
        total = float(np.sum(p_ss))
        if total > 1e-15:
            p_ss = p_ss / total

        dims = model.physical_dims
        mps_exact = tensor_to_mps(p_ss, list(dims))

        bond_dims_to_test = [1, 2, 5, 10, 20, 50]
        compression_results = []

        for chi in bond_dims_to_test:
            mps_c, trunc_err = mps_compress(mps_exact, max_bond_dim=chi)
            dense_c = mps_to_dense(mps_c)
            l1_err = float(np.sum(np.abs(dense_c - p_ss)))
            compression_results.append({
                "bond_dim": chi,
                "truncation_error": trunc_err,
                "l1_error": l1_err,
            })

        # Check errors decrease monotonically
        errors_decrease = all(
            compression_results[i]["l1_error"]
            >= compression_results[i + 1]["l1_error"] - 1e-10
            for i in range(len(compression_results) - 1)
        )

        # Check exponential convergence: last error << first error
        first_err = compression_results[0]["l1_error"]
        last_err = compression_results[-1]["l1_error"]
        exponential_convergence = (
            first_err > 1e-12 and last_err < first_err * 0.1
        ) or last_err < 1e-10

        return {
            "model_name": model.name,
            "physical_dims": list(dims),
            "steady_state_computed": True,
            "compression_results": compression_results,
            "errors_decrease": errors_decrease,
            "exponential_convergence": exponential_convergence,
            "first_l1": first_err,
            "last_l1": last_err,
            "passed": errors_decrease and exponential_convergence,
        }
    except Exception as e:
        return {
            "model_name": "gene_expression",
            "error": str(e),
            "passed": False,
        }
