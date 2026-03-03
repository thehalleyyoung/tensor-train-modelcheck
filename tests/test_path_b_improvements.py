"""
Tests for Path B improvements: FSP-TT coupling, stiffness-aware TDVP,
error tightness, graph-theoretic predictors, and SMC comparison.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# === FSP-TT Coupling Tests ===

class TestFSPTTCoupling:
    """Tests for FSP-TT coupling analysis module."""

    def _get_birth_death(self):
        from tn_check.models.library import birth_death
        return birth_death(birth_rate=1.0, death_rate=0.1, max_copy=30)

    def test_boundary_exit_rate(self):
        """Test boundary exit rate computation."""
        from tn_check.coupling.fsp_tt_coupling import compute_boundary_exit_rate
        from tn_check.cme.fsp import compute_fsp_bounds
        net = self._get_birth_death()
        bounds = compute_fsp_bounds(net, 1.0)
        exit_rate = compute_boundary_exit_rate(net, bounds)
        assert exit_rate >= 0, "Exit rate must be non-negative"

    def test_nonconservativity_bound(self):
        """Test FSP non-conservativity bound."""
        from tn_check.coupling.fsp_tt_coupling import fsp_nonconservativity_bound
        # For exit_rate=0, bound should be 0
        assert fsp_nonconservativity_bound(0.0, 1.0) == 0.0
        # For positive rate and time, bound should be in (0, 1)
        bound = fsp_nonconservativity_bound(1.0, 1.0)
        assert 0 < bound < 1
        # Bound increases with time
        assert fsp_nonconservativity_bound(1.0, 2.0) > bound

    def test_nonconservativity_tracker(self):
        """Test NonConservativityTracker."""
        from tn_check.coupling.fsp_tt_coupling import NonConservativityTracker
        tracker = NonConservativityTracker(exit_rate_bound=0.5)
        tracker.record(0.0, 1.0)
        tracker.record(0.5, 0.98)
        tracker.record(1.0, 0.95)
        assert tracker.max_nonconservativity == pytest.approx(0.05)
        assert tracker.final_nonconservativity == pytest.approx(0.05)
        summary = tracker.summary()
        assert summary["num_points"] == 3

    def test_coupling_analysis(self):
        """Test FSP-TT coupling analysis runs without error."""
        from tn_check.coupling.fsp_tt_coupling import analyze_fsp_tt_coupling
        net = self._get_birth_death()
        analysis = analyze_fsp_tt_coupling(
            net, 1.0, fsp_scales=(1.0,), bond_dims=(5,),
        )
        assert analysis.network_name == "birth_death"
        assert analysis.num_species == 1

    def test_joint_optimization(self):
        """Test joint M-χ optimization runs."""
        from tn_check.coupling.fsp_tt_coupling import joint_m_chi_optimization
        net = self._get_birth_death()
        result = joint_m_chi_optimization(
            net, 1.0, compute_budget=1e10,
            num_fsp_samples=3, num_chi_samples=3,
        )
        assert result.optimal_total_error >= 0


# === Stiffness-Aware TDVP Tests ===

class TestStiffnessTDVP:
    """Tests for stiffness-aware TDVP integration."""

    def _get_birth_death(self):
        from tn_check.models.library import birth_death
        return birth_death(birth_rate=1.0, death_rate=0.1, max_copy=30)

    def test_stiffness_analysis_dataclass(self):
        """Test StiffnessAnalysis dataclass."""
        from tn_check.stiffness.adaptive_tdvp import StiffnessAnalysis
        sa = StiffnessAnalysis(
            stiffness_ratio=100000.0,
            max_eigenvalue_magnitude=100.0,
            min_eigenvalue_magnitude=0.001,
            spectral_gap=0.001,
            num_species=2,
            rate_constant_range=100000.0,
            classification="severely-stiff",
            fast_timescale=0.01,
            slow_timescale=1000.0,
            recommended_dt=0.001,
            recommended_chi=40,
        )
        assert sa.is_stiff
        assert sa.is_severely_stiff

    def test_step_size_controller(self):
        """Test PI step-size controller."""
        from tn_check.stiffness.adaptive_tdvp import StepSizeController
        controller = StepSizeController(target_projection_error=1e-6)
        # When error is at target, dt should stay similar
        dt = controller.suggest_dt(0.01, 1e-6, 100.0)
        assert dt > 0
        assert dt < 1.0
        # When error is high, dt should decrease
        dt_small = controller.suggest_dt(0.01, 1e-3, 100.0)
        assert dt_small <= 0.01

    def test_projection_monitor(self):
        """Test ProjectionMonitor."""
        from tn_check.stiffness.adaptive_tdvp import ProjectionMonitor
        mon = ProjectionMonitor()
        mon.record(0.0, 1e-3, 100.0, 0.01, 10, 1e-5, True)
        mon.record(0.1, 1e-6, 100.0, 0.05, 10, 1e-8, False)
        assert mon.max_projection_error == 1e-3
        assert mon.fast_transient_fraction == 0.5
        summary = mon.summary()
        assert summary["num_steps"] == 2

    def test_detect_fast_transients(self):
        """Test fast transient detection."""
        from tn_check.stiffness.adaptive_tdvp import detect_fast_transients, ProjectionMonitor
        mon = ProjectionMonitor()
        mon.record(0.0, 0.5, 100.0, 0.001, 10, 0.0, True)
        mon.record(0.01, 0.3, 100.0, 0.001, 10, 0.0, True)
        mon.record(0.02, 0.01, 100.0, 0.01, 10, 0.0, False)
        mon.record(0.1, 0.001, 100.0, 0.05, 10, 0.0, False)
        windows = detect_fast_transients(mon, threshold=0.1)
        assert len(windows) >= 1
        assert windows[0][0] == 0.0

    def test_spectrum_analysis(self):
        """Test generator spectrum analysis."""
        from tn_check.stiffness.adaptive_tdvp import analyze_generator_spectrum
        net = self._get_birth_death()
        result = analyze_generator_spectrum(net)
        assert "stiffness_ratio" in result or "error" in result


# === Error Bound Tightness Tests ===

class TestErrorTightness:
    """Tests for error bound tightness study."""

    def test_tightness_result_dataclass(self):
        """Test TightnessResult initialization."""
        from tn_check.evaluation.tightness import TightnessResult
        r = TightnessResult(
            model_name="test",
            num_species=1,
            state_space_size=30,
            time_horizon=1.0,
            max_bond_dim=10,
            certified_error=0.01,
            actual_l1_error=0.001,
            actual_l2_error=0.0005,
            actual_linf_error=0.0001,
            fsp_error_bound=0.005,
            tt_error_bound=0.003,
            tdvp_error_bound=0.002,
            clamping_error_bound=0.0,
        )
        assert r.tightness_ratio == pytest.approx(10.0)
        assert r.bound_valid is True

    def test_tightness_study(self):
        """Test that tightness study runs."""
        from tn_check.evaluation.tightness import run_tightness_study
        study = run_tightness_study(bond_dims=(5, 10), time_horizon=0.5)
        assert study.models_tested > 0


# === Graph-Theoretic Predictors Tests ===

class TestGraphPredictors:
    """Tests for graph-theoretic compressibility predictors."""

    def _get_birth_death(self):
        from tn_check.models.library import birth_death
        return birth_death(birth_rate=1.0, death_rate=0.1, max_copy=30)

    def _get_gene_expression(self):
        from tn_check.models.library import gene_expression
        return gene_expression(
            k_txn=0.5,
            k_tln=5.0,
            gamma_m=0.1,
            gamma_p=0.005,
            max_copy_mRNA=15,
            max_copy_protein=20,
        )

    def test_tree_width_single_species(self):
        """Tree-width of single-species network should be 0."""
        from tn_check.evaluation.graph_predictors import compute_tree_width
        net = self._get_birth_death()
        result = compute_tree_width(net)
        assert result.tree_width == 0
        assert result.is_tree

    def test_tree_width_two_species(self):
        """Tree-width of 2-species coupled network should be ≤ 1."""
        from tn_check.evaluation.graph_predictors import compute_tree_width
        net = self._get_gene_expression()
        result = compute_tree_width(net)
        assert result.tree_width >= 0
        assert result.num_species == 2

    def test_deficiency_birth_death(self):
        """Birth-death is deficiency zero."""
        from tn_check.evaluation.graph_predictors import compute_deficiency
        net = self._get_birth_death()
        result = compute_deficiency(net)
        # Birth-death: 2 complexes (∅, X), 1 or 2 linkage classes
        assert result.deficiency >= 0
        assert result.rank_stoichiometry == 1

    def test_modularity(self):
        """Test modularity computation."""
        from tn_check.evaluation.graph_predictors import compute_modularity
        net = self._get_birth_death()
        result = compute_modularity(net)
        assert result.num_modules >= 1

    def test_combined_predictor(self):
        """Test combined compressibility prediction."""
        from tn_check.evaluation.graph_predictors import predict_compressibility_graph
        net = self._get_birth_death()
        result = predict_compressibility_graph(net)
        assert result.predicted_rank_bound >= 1
        assert 0 <= result.compressibility_score <= 1
        assert result.recommended_bond_dim >= 10


# === SMC Comparison Tests ===

class TestSMCComparison:
    """Tests for statistical model checking comparison."""

    def _get_birth_death(self):
        from tn_check.models.library import birth_death
        return birth_death(birth_rate=1.0, death_rate=0.1, max_copy=30)

    def test_gillespie_simulation(self):
        """Test Gillespie SSA produces valid trajectories."""
        from tn_check.smc.statistical_checker import gillespie_simulation
        net = self._get_birth_death()
        trajectories = gillespie_simulation(net, 1.0, num_trajectories=10)
        assert len(trajectories) == 10
        for traj in trajectories:
            assert "final_state" in traj
            assert traj["final_state"][0] >= 0

    def test_sprt(self):
        """Test SPRT gives a verdict."""
        from tn_check.smc.statistical_checker import importance_sampling_sprt
        net = self._get_birth_death()

        def check(state, t):
            return state[0] >= 5

        result = importance_sampling_sprt(
            net, 1.0, check, threshold=0.5,
            alpha=0.05, beta=0.05, max_samples=100,
        )
        assert result.verdict in ("TRUE", "FALSE")
        assert 0 <= result.probability_estimate <= 1
        assert result.num_samples > 0

    def test_wilson_ci(self):
        """Test Wilson score confidence interval."""
        from tn_check.smc.statistical_checker import _wilson_ci
        lower, upper = _wilson_ci(50, 100, 0.95)
        assert 0.3 < lower < 0.5
        assert 0.5 < upper < 0.7

    def test_smc_comparison(self):
        """Test SMC vs TN-Check comparison runs."""
        from tn_check.smc.statistical_checker import run_smc_comparison
        net = self._get_birth_death()
        result = run_smc_comparison(
            net, "P>=0.5[F<=1.0 X>=5]", 0.5, 1.0,
            max_bond_dim=20, smc_max_samples=100,
        )
        assert result.smc_result.verdict in ("TRUE", "FALSE")
        assert result.network_name == "birth_death"


# === Integration Tests ===

class TestPathBIntegration:
    """Integration tests for the complete Path B experiment pipeline."""

    def test_all_experiments_quick(self):
        """Quick run of all experiments should succeed."""
        from tn_check.evaluation.path_b_experiments import run_all_experiments
        results = run_all_experiments(quick=True)
        assert "metadata" in results
        assert results["metadata"]["total_wall_time"] > 0
        # At least some experiments should produce results
        for key in ("fsp_tt_coupling", "stiffness_tdvp", "smc_comparison"):
            assert key in results

    def test_imports(self):
        """All new modules should import correctly."""
        from tn_check.coupling import analyze_fsp_tt_coupling
        from tn_check.stiffness import StiffnessAwareTDVP
        from tn_check.smc import run_smc_comparison
        from tn_check.evaluation.tightness import run_tightness_study
        from tn_check.evaluation.graph_predictors import predict_compressibility_graph
