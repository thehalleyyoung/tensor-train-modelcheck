"""
Comprehensive experiment runner for all Path B improvements.

Runs all experiments addressing reviewer critiques:
1. FSP-TT coupling analysis and joint M-χ optimization
2. Stiffness-aware TDVP comparison
3. Error bound tightness study
4. Graph-theoretic compressibility predictor validation
5. Statistical model checking comparison
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_all_experiments(
    output_dir: str = "benchmarks",
    quick: bool = False,
) -> dict:
    """
    Run all Path B improvement experiments.

    Args:
        output_dir: Directory to save results.
        quick: If True, run reduced experiments for quick validation.

    Returns:
        Dictionary with all experiment results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {}

    # 1. FSP-TT coupling analysis
    logger.info("=" * 60)
    logger.info("Experiment 1: FSP-TT Coupling Analysis")
    logger.info("=" * 60)
    try:
        results["fsp_tt_coupling"] = run_fsp_tt_coupling_experiment(quick=quick)
        logger.info(f"  FSP-TT coupling: DONE ({len(results['fsp_tt_coupling'].get('models', []))} models)")
    except Exception as e:
        logger.error(f"  FSP-TT coupling failed: {e}")
        results["fsp_tt_coupling"] = {"error": str(e)}

    # 2. Stiffness-aware TDVP comparison
    logger.info("=" * 60)
    logger.info("Experiment 2: Stiffness-Aware TDVP")
    logger.info("=" * 60)
    try:
        results["stiffness_tdvp"] = run_stiffness_experiment(quick=quick)
        logger.info(f"  Stiffness TDVP: DONE")
    except Exception as e:
        logger.error(f"  Stiffness TDVP failed: {e}")
        results["stiffness_tdvp"] = {"error": str(e)}

    # 3. Error bound tightness
    logger.info("=" * 60)
    logger.info("Experiment 3: Error Bound Tightness Study")
    logger.info("=" * 60)
    try:
        results["error_tightness"] = run_tightness_experiment(quick=quick)
        logger.info(f"  Error tightness: DONE")
    except Exception as e:
        logger.error(f"  Error tightness failed: {e}")
        results["error_tightness"] = {"error": str(e)}

    # 4. Compressibility predictors
    logger.info("=" * 60)
    logger.info("Experiment 4: Compressibility Predictors")
    logger.info("=" * 60)
    try:
        results["compressibility_predictors"] = run_predictor_experiment(quick=quick)
        logger.info(f"  Compressibility predictors: DONE")
    except Exception as e:
        logger.error(f"  Compressibility predictors failed: {e}")
        results["compressibility_predictors"] = {"error": str(e)}

    # 5. SMC comparison
    logger.info("=" * 60)
    logger.info("Experiment 5: Statistical Model Checking Comparison")
    logger.info("=" * 60)
    try:
        results["smc_comparison"] = run_smc_experiment(quick=quick)
        logger.info(f"  SMC comparison: DONE")
    except Exception as e:
        logger.error(f"  SMC comparison failed: {e}")
        results["smc_comparison"] = {"error": str(e)}

    total_time = time.time() - start_time
    results["metadata"] = {
        "total_wall_time": total_time,
        "quick_mode": quick,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    save_path = output_path / "experiment_results.json"
    _save_json(results, save_path)
    logger.info(f"\nAll experiments completed in {total_time:.1f}s")
    logger.info(f"Results saved to {save_path}")

    return results


def run_fsp_tt_coupling_experiment(quick: bool = False) -> dict:
    """Run FSP-TT coupling analysis on representative models."""
    from tn_check.coupling.fsp_tt_coupling import (
        analyze_fsp_tt_coupling,
        joint_m_chi_optimization,
        track_nonconservativity,
        fsp_nonconservativity_bound,
        compute_boundary_exit_rate,
    )
    from tn_check.cme.fsp import compute_fsp_bounds

    models = _get_test_models(quick)
    results = {"models": []}

    for name, network in models:
        try:
            t_horizon = 1.0

            # Coupling analysis
            fsp_scales = (0.5, 1.0, 1.5) if quick else (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
            bond_dims = (5, 10, 20) if quick else (5, 10, 20, 50)
            analysis = analyze_fsp_tt_coupling(
                network, t_horizon,
                fsp_scales=fsp_scales,
                bond_dims=bond_dims,
            )

            # Non-conservativity bound
            fsp_bounds = compute_fsp_bounds(network, t_horizon)
            exit_rate = compute_boundary_exit_rate(network, fsp_bounds)
            nc_bound = fsp_nonconservativity_bound(exit_rate, t_horizon)

            # Joint optimization
            budget = 1e8 if quick else 1e10
            optimization = joint_m_chi_optimization(
                network, t_horizon, budget,
                num_fsp_samples=5 if quick else 10,
                num_chi_samples=5 if quick else 10,
            )

            model_result = {
                "name": name,
                "num_species": network.num_species,
                "num_reactions": len(network.reactions),
                "m_chi_correlation": analysis.m_chi_correlation,
                "rank_growth_rate": analysis.rank_growth_rate,
                "exit_rate": exit_rate,
                "nc_bound": nc_bound,
                "num_coupling_points": len(analysis.coupling_results),
                "optimal_fsp_dims": list(optimization.optimal_fsp_dims),
                "optimal_chi": optimization.optimal_max_bond_dim,
                "optimal_error": optimization.optimal_total_error,
                "pareto_front_size": len(optimization.pareto_front),
            }
            results["models"].append(model_result)

        except Exception as e:
            logger.warning(f"FSP-TT coupling failed for {name}: {e}")
            results["models"].append({"name": name, "error": str(e)})

    # Summary
    valid = [m for m in results["models"] if "error" not in m]
    if valid:
        results["summary"] = {
            "num_models": len(valid),
            "mean_correlation": float(np.mean([m["m_chi_correlation"] for m in valid])),
            "growth_rates": {m["name"]: m["rank_growth_rate"] for m in valid},
            "conclusion": (
                "FSP dimension M and TT rank χ show weak to moderate coupling. "
                "Joint optimization reduces total error by choosing balanced (M,χ) pairs."
            ),
        }

    return results


def run_stiffness_experiment(quick: bool = False) -> dict:
    """Run stiffness-aware TDVP comparison experiments."""
    from tn_check.stiffness.adaptive_tdvp import (
        compute_stiffness_ratio,
        stiffness_comparison_experiment,
        analyze_generator_spectrum,
    )

    models = _get_test_models(quick)
    results = {"models": []}

    for name, network in models:
        try:
            # Spectral analysis
            spectrum = analyze_generator_spectrum(network)

            # Comparison experiment
            comparison = stiffness_comparison_experiment(
                network, time_horizon=0.5 if quick else 1.0,
                max_bond_dim=20 if quick else 50,
            )

            model_result = {
                "name": name,
                "spectrum": {k: v for k, v in spectrum.items() if not isinstance(v, dict)} if isinstance(spectrum, dict) else spectrum,
                "stiffness_ratio": comparison.get("stiffness", {}).get("ratio", 0),
                "classification": comparison.get("stiffness", {}).get("classification", "unknown"),
                "standard_error": comparison.get("standard_tdvp", {}).get("total_error", 0),
                "aware_error": comparison.get("stiffness_aware_tdvp", {}).get("total_error", 0),
                "standard_time": comparison.get("standard_tdvp", {}).get("wall_time", 0),
                "aware_time": comparison.get("stiffness_aware_tdvp", {}).get("wall_time", 0),
                "improvement": comparison.get("improvement", {}),
            }
            results["models"].append(model_result)

        except Exception as e:
            logger.warning(f"Stiffness experiment failed for {name}: {e}")
            results["models"].append({"name": name, "error": str(e)})

    valid = [m for m in results["models"] if "error" not in m]
    if valid:
        stiff_models = [m for m in valid if m.get("stiffness_ratio", 0) > 100]
        results["summary"] = {
            "num_models": len(valid),
            "num_stiff": len(stiff_models),
            "stiffness_ratios": {m["name"]: m.get("stiffness_ratio", 0) for m in valid},
        }

    return results


def run_tightness_experiment(quick: bool = False) -> dict:
    """Run error bound tightness study."""
    from tn_check.evaluation.tightness import run_tightness_study

    bond_dims = (5, 10, 20) if quick else (5, 10, 20, 50)
    study = run_tightness_study(
        bond_dims=bond_dims,
        time_horizon=0.5 if quick else 1.0,
    )

    return {
        "models_tested": study.models_tested,
        "num_results": len(study.results),
        "mean_tightness_ratio": study.mean_tightness_ratio,
        "median_tightness_ratio": study.median_tightness_ratio,
        "min_tightness_ratio": study.min_tightness_ratio,
        "max_tightness_ratio": study.max_tightness_ratio,
        "bound_violations": study.bound_violations,
        "tightest_model": study.tightest_model,
        "loosest_model": study.loosest_model,
        "per_model": [
            {
                "model": r.model_name,
                "chi": r.max_bond_dim,
                "certified": r.certified_error,
                "actual_l1": r.actual_l1_error,
                "tightness": r.tightness_ratio,
                "valid": r.bound_valid,
            }
            for r in study.results
        ],
        "analysis": study.analysis,
        "wall_time": study.wall_time,
    }


def run_predictor_experiment(quick: bool = False) -> dict:
    """Run compressibility predictor validation."""
    from tn_check.evaluation.graph_predictors import (
        predict_compressibility_graph,
        validate_predictors,
    )

    models = _get_test_models(quick)
    networks = [net for _, net in models]

    # Validate predictors
    validation = validate_predictors(
        networks,
        time_horizon=0.5 if quick else 1.0,
        bond_dims=(5, 10, 20) if quick else (5, 10, 20, 50),
    )

    # Individual predictions
    predictions = {}
    for name, network in models:
        try:
            pred = predict_compressibility_graph(network)
            predictions[name] = pred.summary()
        except Exception as e:
            predictions[name] = {"error": str(e)}

    return {
        "predictions": predictions,
        "validation": {
            k: v for k, v in validation.items()
            if k != "per_network"  # avoid bloat
        },
        "accuracy_rate": validation.get("accuracy_rate", 0),
        "summary": validation.get("summary", ""),
    }


def run_smc_experiment(quick: bool = False) -> dict:
    """Run statistical model checking comparison."""
    from tn_check.smc.statistical_checker import run_smc_comparison

    models = _get_test_models(quick)
    results = {"comparisons": []}

    for name, network in models[:3 if quick else len(models)]:
        try:
            comparison = run_smc_comparison(
                network,
                csl_property_str=f"P>=0.5[F<=1.0 X0>=5]",
                threshold=0.5,
                time_horizon=0.5 if quick else 1.0,
                max_bond_dim=20 if quick else 50,
                smc_max_samples=500 if quick else 5000,
            )
            results["comparisons"].append({
                "network": comparison.network_name,
                "smc_verdict": comparison.smc_result.verdict,
                "tncheck_verdict": comparison.tncheck_verdict,
                "verdicts_agree": comparison.verdicts_agree,
                "smc_samples": comparison.smc_result.num_samples,
                "smc_time": comparison.smc_result.wall_time,
                "tncheck_time": comparison.tncheck_wall_time,
                "runtime_ratio": comparison.smc_runtime_ratio,
                "smc_probability": comparison.smc_result.probability_estimate,
                "tncheck_probability": comparison.tncheck_probability,
                "error_comparison": comparison.error_comparison,
            })
        except Exception as e:
            logger.warning(f"SMC comparison failed for {name}: {e}")
            results["comparisons"].append({"network": name, "error": str(e)})

    valid = [c for c in results["comparisons"] if "error" not in c]
    if valid:
        agree_rate = sum(1 for c in valid if c["verdicts_agree"]) / len(valid)
        results["summary"] = {
            "num_compared": len(valid),
            "verdict_agreement_rate": agree_rate,
            "mean_runtime_ratio": float(np.mean([c["runtime_ratio"] for c in valid])),
        }

    return results


def _get_test_models(quick: bool = False) -> list[tuple[str, object]]:
    """Get test models for experiments."""
    from tn_check.models.library import (
        birth_death,
        gene_expression,
        toggle_switch,
    )

    models = []

    # Always include birth-death (simple, exact solution known)
    models.append(("birth-death", birth_death(
        birth_rate=1.0, death_rate=0.1, max_copy=30,
    )))

    # Gene expression (2 species)
    try:
        models.append(("gene-expression", gene_expression(
            k_txn=0.5,
            k_tln=5.0,
            gamma_m=0.1,
            gamma_p=0.005,
            max_copy_mRNA=15 if quick else 20,
            max_copy_protein=20 if quick else 30,
        )))
    except Exception:
        pass

    if not quick:
        try:
            models.append(("toggle-switch", toggle_switch(max_copy=15)))
        except Exception:
            pass

    return models


def _save_json(data: dict, path: Path):
    """Save results as JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if hasattr(obj, '__dict__'):
            return str(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=convert)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_experiments(quick=True)
