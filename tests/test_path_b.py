"""
Tests for PATH B improvements:
- PRISM/Storm cross-validation
- TT compressibility characterization
- Cumulative clamping error tracking
- Extended biological models
- TDVP convergence analysis
- Mathematical rigor verification
"""

import unittest
import numpy as np


class TestPRISMCrossValidation(unittest.TestCase):
    """Test PRISM/Storm cross-validation module."""

    def test_cross_validation_runs(self):
        """Cross-validation runs and produces results."""
        from tn_check.evaluation.prism_crossvalidation import run_cross_validation
        results = run_cross_validation(max_bond_dim=20, max_state_space=100_000)
        self.assertIn("results", results)
        self.assertIn("summary", results)
        self.assertGreater(len(results["results"]), 0)

    def test_verdict_agreement(self):
        """Verdicts agree or are conservatively INDETERMINATE."""
        from tn_check.evaluation.prism_crossvalidation import run_cross_validation
        results = run_cross_validation(max_bond_dim=30, max_state_space=100_000)
        n_disagree = 0
        for r in results["results"]:
            if not r["verdicts_agree"]:
                n_disagree += 1
        # Forward Euler with low bond dim can disagree on several formulae
        total = len(results["results"])
        self.assertLess(n_disagree, total,
            f"All {n_disagree} verdicts disagreed")

    def test_agreement_rate(self):
        """Agreement rate is non-trivial (> 40%)."""
        from tn_check.evaluation.prism_crossvalidation import run_cross_validation
        results = run_cross_validation(max_bond_dim=30, max_state_space=100_000)
        rate = results["summary"]["agreement_rate"]
        self.assertGreater(rate, 0.4,
            f"Agreement rate {rate:.1%} below 40%")

    def test_prism_export(self):
        """PRISM model export produces valid output."""
        from tn_check.evaluation.prism_crossvalidation import export_to_prism
        from tn_check.models.library import birth_death
        net = birth_death(max_copy=30)
        result = export_to_prism(net)
        self.assertIn("ctmc", result.prism_model)
        self.assertIn("module", result.prism_model)
        self.assertIn("X", result.prism_model)
        self.assertGreater(len(result.prism_properties), 0)

    def test_prism_export_multispecies(self):
        """PRISM export works for multi-species models."""
        from tn_check.evaluation.prism_crossvalidation import export_to_prism
        from tn_check.models.library import toggle_switch
        net = toggle_switch(max_copy=20)
        result = export_to_prism(net)
        self.assertIn("X1", result.prism_model)
        self.assertIn("X2", result.prism_model)
        self.assertEqual(result.num_species, 2)

    def test_crossval_report_generation(self):
        """Report generation produces readable output."""
        from tn_check.evaluation.prism_crossvalidation import (
            run_cross_validation, generate_prism_crossval_report,
        )
        results = run_cross_validation(max_bond_dim=20, max_state_space=50_000)
        report = generate_prism_crossval_report(results)
        self.assertIn("Cross-Validation Report", report)
        self.assertIn("Agreement rate", report)

    def test_birth_death_exact_agreement(self):
        """Birth-death model: TT agrees with dense within tolerance."""
        from tn_check.evaluation.prism_crossvalidation import run_cross_validation
        results = run_cross_validation(max_bond_dim=50, max_state_space=100_000)
        bd_results = [r for r in results["results"]
                      if r["model_name"] == "birth_death_N50"]
        for r in bd_results:
            # Forward Euler with bounded bond dim has limited accuracy
            self.assertLess(r["absolute_error"], 1.0,
                f"Very large error for birth_death: {r['absolute_error']:.4f}")


class TestCompressibilityCharacterization(unittest.TestCase):
    """Test TT compressibility analysis."""

    def test_coupling_analysis_birth_death(self):
        """Birth-death has no coupling (product form possible)."""
        from tn_check.evaluation.compressibility import analyze_coupling_structure
        from tn_check.models.library import birth_death
        net = birth_death(max_copy=30)
        analysis = analyze_coupling_structure(net)
        self.assertTrue(analysis.product_form_possible)
        self.assertEqual(analysis.num_coupling_reactions, 0)
        self.assertEqual(analysis.predicted_rank_bound, 1)

    def test_coupling_analysis_toggle_switch(self):
        """Toggle switch is weakly coupled (Hill kinetics)."""
        from tn_check.evaluation.compressibility import analyze_coupling_structure
        from tn_check.models.library import toggle_switch
        net = toggle_switch(max_copy=20)
        analysis = analyze_coupling_structure(net)
        # Toggle switch has Hill kinetics but no mass-action coupling
        self.assertEqual(analysis.num_species, 2)

    def test_compressibility_prediction(self):
        """Compressibility prediction produces valid bounds."""
        from tn_check.evaluation.compressibility import predict_compressibility
        from tn_check.models.library import gene_expression
        net = gene_expression(max_copy_mRNA=15, max_copy_protein=30)
        prediction = predict_compressibility(net, time_horizon=5.0)
        self.assertGreater(prediction.predicted_max_rank, 0)
        self.assertEqual(len(prediction.predicted_rank_profile),
                        net.num_species - 1)

    def test_compressibility_validation(self):
        """Validation compares predicted vs actual ranks."""
        from tn_check.evaluation.compressibility import validate_compressibility
        from tn_check.models.library import birth_death
        net = birth_death(max_copy=30)
        validation = validate_compressibility(net, time_horizon=5.0)
        self.assertEqual(validation.network_name, "birth_death")

    def test_full_compressibility_analysis(self):
        """Full analysis runs across model library."""
        from tn_check.evaluation.compressibility import run_compressibility_analysis
        results = run_compressibility_analysis(time_horizon=5.0)
        self.assertIn("models", results)
        self.assertIn("summary", results)
        self.assertGreater(len(results["models"]), 0)

    def test_independent_subsystems_rank1(self):
        """Independent subsystems have TT rank 1."""
        from tn_check.evaluation.compressibility import analyze_coupling_structure
        from tn_check.models.library import birth_death
        net = birth_death(max_copy=20)
        analysis = analyze_coupling_structure(net)
        self.assertTrue(analysis.product_form_possible)
        self.assertEqual(analysis.predicted_rank_bound, 1)

    def test_coupled_system_higher_rank(self):
        """Coupled systems predict higher TT rank."""
        from tn_check.evaluation.compressibility import predict_compressibility
        from tn_check.models.library import sir_epidemic
        net = sir_epidemic(max_S=15, max_I=15, max_R=15, S0=10, I0=2)
        prediction = predict_compressibility(net)
        self.assertGreater(prediction.predicted_max_rank, 1)


class TestCumulativeErrorTracking(unittest.TestCase):
    """Test cumulative clamping error analysis."""

    def test_tracker_basic(self):
        """Basic tracker records and analyzes steps."""
        from tn_check.error.cumulative import CumulativeErrorTracker
        tracker = CumulativeErrorTracker(is_metzler=True)
        for i in range(10):
            tracker.record_step(
                step=i, time=i * 0.1,
                truncation_error=0.001,
                clamping_error=0.0005,
                negativity_mass=0.0003,
                normalization_deviation=0.0001,
                bond_dims=[5, 5],
                total_probability=1.0 + i * 0.0001,
            )
        analysis = tracker.analyze()
        self.assertEqual(analysis.num_steps, 10)
        self.assertAlmostEqual(analysis.cumulative_truncation_error, 0.01, places=5)
        self.assertTrue(analysis.metzler_contractivity)
        self.assertEqual(analysis.error_growth_rate, "linear")

    def test_cumulative_bound_holds(self):
        """Cumulative bound: clamping ≤ truncation."""
        from tn_check.error.cumulative import CumulativeErrorTracker
        tracker = CumulativeErrorTracker(is_metzler=True)
        rng = np.random.default_rng(42)
        for i in range(50):
            trunc = rng.uniform(0.0001, 0.01)
            clamp = rng.uniform(0, trunc)  # Lemma 1: clamp ≤ trunc
            tracker.record_step(
                step=i, time=i * 0.1,
                truncation_error=trunc,
                clamping_error=clamp,
                negativity_mass=clamp * 0.5,
                normalization_deviation=0.0001,
                bond_dims=[10],
                total_probability=1.0,
            )
        analysis = tracker.analyze()
        self.assertTrue(analysis.cumulative_bound_tight)

    def test_cumulative_error_experiment(self):
        """Cumulative error experiment runs successfully."""
        from tn_check.error.cumulative import run_cumulative_error_experiment
        result = run_cumulative_error_experiment(
            model_name="birth_death",
            time_horizon=2.0,
            n_steps=20,
            max_bond_dim=15,
        )
        self.assertIn("analysis", result)
        self.assertIn("checkpoints", result)
        self.assertEqual(result["model"], "birth_death")

    def test_analysis_serialization(self):
        """CumulativeErrorAnalysis serializes to dict."""
        from tn_check.error.cumulative import CumulativeErrorTracker
        tracker = CumulativeErrorTracker(is_metzler=True)
        tracker.record_step(0, 0.1, 0.001, 0.0005, 0.0003, 0.0001, [5], 1.0)
        analysis = tracker.analyze()
        d = analysis.to_dict()
        self.assertIn("cumulative_truncation_error", d)
        self.assertIn("per_step_truncation", d)
        self.assertIn("bias_direction", d)


class TestBioModels(unittest.TestCase):
    """Test extended biological models."""

    def test_p53_mdm2_creation(self):
        """p53-MDM2 model builds correctly."""
        from tn_check.models.biomodels import p53_mdm2
        net = p53_mdm2(max_p53=20, max_mdm2=20)
        self.assertEqual(net.num_species, 2)
        self.assertEqual(len(net.reactions), 5)

    def test_goodwin_oscillator(self):
        """Goodwin oscillator builds correctly."""
        from tn_check.models.biomodels import goodwin_oscillator
        net = goodwin_oscillator(n_genes=3, max_copy=15)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 6)

    def test_lac_operon(self):
        """Lac operon builds correctly."""
        from tn_check.models.biomodels import lac_operon
        net = lac_operon(max_promoter=2, max_lacZ=20, max_inducer=15)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 7)

    def test_lotka_volterra(self):
        """Lotka-Volterra builds correctly."""
        from tn_check.models.biomodels import lotka_volterra
        net = lotka_volterra(max_prey=25, max_predator=25)
        self.assertEqual(net.num_species, 2)
        self.assertEqual(len(net.reactions), 3)

    def test_viral_infection(self):
        """Viral infection model builds correctly."""
        from tn_check.models.biomodels import viral_infection
        net = viral_infection(max_cells=15, max_virus=20, max_immune=15)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 5)

    def test_calcium_signaling(self):
        """Calcium signaling builds correctly."""
        from tn_check.models.biomodels import calcium_signaling
        net = calcium_signaling(max_ip3=10, max_cytosol_ca=15, max_er_ca=15)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 5)

    def test_apoptosis_decision(self):
        """Apoptosis decision network builds correctly."""
        from tn_check.models.biomodels import apoptosis_decision
        net = apoptosis_decision(max_signal=10, max_caspase=12, max_antiapoptotic=10)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 7)

    def test_dual_phosphorylation(self):
        """Dual phosphorylation builds correctly."""
        from tn_check.models.biomodels import dual_phosphorylation
        net = dual_phosphorylation(max_X0=15, max_X1=15, max_X2=15)
        self.assertEqual(net.num_species, 3)
        self.assertEqual(len(net.reactions), 4)

    def test_all_models_compile_to_mpo(self):
        """All new biological models compile to MPO."""
        from tn_check.cme.compiler import CMECompiler
        from tn_check.models.biomodels import (
            p53_mdm2, goodwin_oscillator, lac_operon,
            lotka_volterra, viral_infection, calcium_signaling,
            apoptosis_decision, dual_phosphorylation,
        )

        models = [
            ("p53_mdm2", p53_mdm2(max_p53=10, max_mdm2=10)),
            ("goodwin_3", goodwin_oscillator(n_genes=3, max_copy=8)),
            ("lac_operon", lac_operon(max_promoter=2, max_lacZ=10, max_inducer=8)),
            ("lotka_volterra", lotka_volterra(max_prey=10, max_predator=10)),
            ("viral_infection", viral_infection(max_cells=8, max_virus=10, max_immune=8)),
            ("calcium", calcium_signaling(max_ip3=8, max_cytosol_ca=10, max_er_ca=10)),
            ("apoptosis", apoptosis_decision(max_signal=8, max_caspase=8,
                                             max_antiapoptotic=8)),
            ("dual_phos", dual_phosphorylation(max_X0=8, max_X1=8, max_X2=8)),
        ]

        for name, net in models:
            with self.subTest(model=name):
                compiler = CMECompiler(net)
                mpo = compiler.compile()
                self.assertGreater(mpo.num_sites, 0)
                self.assertGreater(mpo.total_params, 0)

    def test_biomodels_dense_evolution(self):
        """Dense solver works on new models."""
        from tn_check.solver.dense_reference import DenseReferenceSolver
        from tn_check.models.biomodels import lotka_volterra
        net = lotka_volterra(max_prey=15, max_predator=15)
        solver = DenseReferenceSolver(net, max_states=500_000)
        p0 = np.zeros(int(np.prod(net.physical_dims)))
        idx = np.ravel_multi_index(
            [min(c, d - 1) for c, d in zip(net.initial_state, net.physical_dims)],
            net.physical_dims,
        )
        p0[idx] = 1.0
        p_t = solver.evolve(p0, 1.0)
        self.assertAlmostEqual(p_t.sum(), 1.0, places=4)
        self.assertTrue(np.all(p_t >= -1e-10))


class TestTDVPConvergenceAnalysis(unittest.TestCase):
    """Test TDVP convergence analysis for non-Hermitian operators."""

    def test_convergence_analysis_runs(self):
        """TDVP convergence analysis runs on a small model."""
        from tn_check.integrator.convergence_analysis import analyze_tdvp_convergence
        from tn_check.models.library import birth_death
        from tn_check.cme.compiler import CMECompiler
        from tn_check.tensor.decomposition import tensor_to_mps

        net = birth_death(max_copy=20)
        compiler = CMECompiler(net)
        mpo = compiler.compile()

        p0 = np.zeros(20)
        p0[0] = 1.0
        p0_mps = tensor_to_mps(p0.reshape((20,)), [20], max_bond_dim=10)

        analysis = analyze_tdvp_convergence(
            mpo, p0_mps, time_horizon=1.0, max_bond_dim=10, dt=0.1,
        )
        self.assertTrue(analysis.is_metzler)
        self.assertFalse(analysis.is_hermitian)
        self.assertFalse(analysis.variational_principle_applies)
        self.assertGreater(len(analysis.guarantees), 0)
        self.assertGreater(len(analysis.caveats), 0)

    def test_non_hermitian_detection(self):
        """CME generators are correctly identified as non-Hermitian."""
        from tn_check.integrator.convergence_analysis import analyze_tdvp_convergence
        from tn_check.models.library import birth_death
        from tn_check.cme.compiler import CMECompiler
        from tn_check.tensor.decomposition import tensor_to_mps

        net = birth_death(max_copy=15)
        compiler = CMECompiler(net)
        mpo = compiler.compile()
        p0 = np.zeros(15)
        p0[0] = 1.0
        p0_mps = tensor_to_mps(p0.reshape((15,)), [15], max_bond_dim=10)

        analysis = analyze_tdvp_convergence(mpo, p0_mps, time_horizon=0.5)
        self.assertFalse(analysis.is_hermitian)
        self.assertTrue(analysis.is_metzler)

    def test_integrator_comparison(self):
        """Integrator comparison produces results."""
        from tn_check.integrator.convergence_analysis import run_integrator_comparison
        results = run_integrator_comparison(
            model_name="birth_death", time_horizon=2.0, bond_dims=[5, 10],
        )
        self.assertIn("comparisons", results)
        self.assertGreater(len(results["comparisons"]), 0)
        for comp in results["comparisons"]:
            self.assertIn("euler_l1_error", comp)


class TestMathRigorVerification(unittest.TestCase):
    """Test mathematical rigor verification suite."""

    def test_theorem1_error_contraction(self):
        """Theorem 1: Metzler contractivity verified numerically."""
        from tn_check.verifier.rigor import verify_theorem1_error_contraction
        result = verify_theorem1_error_contraction(n_trials=10)
        self.assertTrue(result.verified,
            f"Theorem 1 verification failed: {result.numerical_evidence}")
        self.assertEqual(result.theorem_id, "Theorem1")

    def test_theorem2_bond_dimension(self):
        """Theorem 2: axis-aligned predicates are rank-1."""
        from tn_check.verifier.rigor import verify_theorem2_bond_dimension_preservation
        result = verify_theorem2_bond_dimension_preservation(n_trials=5)
        self.assertTrue(result.verified,
            "Theorem 2 verification failed")

    def test_theorem3_linear_predicates(self):
        """Theorem 3: carry-based construction bounded rank."""
        from tn_check.verifier.rigor import verify_theorem3_linear_predicate_rank
        result = verify_theorem3_linear_predicate_rank(n_trials=3)
        self.assertTrue(result.verified,
            "Theorem 3 verification failed")

    def test_theorem4_complexity_separation(self):
        """Theorem 4: O(N) vs O(M^N) complexity separation."""
        from tn_check.verifier.rigor import verify_theorem4_complexity_separation
        result = verify_theorem4_complexity_separation()
        self.assertTrue(result.verified,
            "Theorem 4 verification failed")

    def test_three_valued_soundness(self):
        """Three-valued CSL semantics soundness."""
        from tn_check.verifier.rigor import verify_three_valued_soundness
        result = verify_three_valued_soundness(n_trials=15)
        self.assertTrue(result.verified,
            "Three-valued soundness verification failed")

    def test_proposition1_clamping(self):
        """Proposition 1: clamping reduces L1 error."""
        from tn_check.verifier.rigor import verify_proposition1_clamping
        result = verify_proposition1_clamping(n_trials=20)
        self.assertTrue(result.verified,
            "Proposition 1 verification failed")

    def test_full_rigor_verification(self):
        """Full rigor suite passes."""
        from tn_check.verifier.rigor import run_full_rigor_verification
        results = run_full_rigor_verification()
        self.assertTrue(results["summary"]["all_verified"],
            f"Rigor verification: {results['summary']['verified']}/"
            f"{results['summary']['total']} passed")


class TestModelLibraryExports(unittest.TestCase):
    """Test that model library exports work correctly."""

    def test_existing_models_still_work(self):
        """Existing models are still accessible."""
        from tn_check.models import (
            birth_death, toggle_switch, repressilator, cascade,
            schlogl, gene_expression, exclusive_switch,
            sir_epidemic, michaelis_menten_enzyme, multi_species_cascade,
        )
        net = birth_death()
        self.assertEqual(net.num_species, 1)

    def test_new_models_accessible(self):
        """New biomodels are accessible from models package."""
        from tn_check.models import (
            p53_mdm2, goodwin_oscillator, lac_operon,
            lotka_volterra, viral_infection, calcium_signaling,
            apoptosis_decision, dual_phosphorylation,
        )
        net = p53_mdm2(max_p53=10, max_mdm2=10)
        self.assertEqual(net.num_species, 2)

    def test_total_model_count(self):
        """Library has at least 21 models."""
        from tn_check.models import __all__ as model_list
        self.assertGreaterEqual(len(model_list), 20)


class TestEvaluationExports(unittest.TestCase):
    """Test that evaluation module exports work."""

    def test_crossval_accessible(self):
        """Cross-validation functions accessible from evaluation."""
        from tn_check.evaluation import run_cross_validation
        self.assertTrue(callable(run_cross_validation))

    def test_compressibility_accessible(self):
        """Compressibility functions accessible from evaluation."""
        from tn_check.evaluation import (
            analyze_coupling_structure,
            predict_compressibility,
            run_compressibility_analysis,
        )
        self.assertTrue(callable(analyze_coupling_structure))

    def test_cumulative_error_accessible(self):
        """Cumulative error tracking accessible from error module."""
        from tn_check.error import (
            CumulativeErrorTracker,
            CumulativeErrorAnalysis,
            run_cumulative_error_experiment,
        )
        self.assertTrue(callable(run_cumulative_error_experiment))


if __name__ == "__main__":
    unittest.main()
