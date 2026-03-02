"""
Evaluation and benchmarking framework.

Provides ground-truth comparison for small systems,
TT-compressibility survey utilities, scaling benchmarks,
accuracy benchmarks, PRISM-style comparison utilities,
cross-validation with dense reference, and compressibility analysis.
"""

from tn_check.evaluation.benchmark import (
    ground_truth_check,
    compressibility_survey,
    BenchmarkResult,
    run_scaling_benchmark,
    run_accuracy_benchmark,
    run_all_benchmarks,
)

from tn_check.evaluation.prism_crossvalidation import (
    run_cross_validation,
    export_to_prism,
    generate_prism_crossval_report,
)

from tn_check.evaluation.compressibility import (
    analyze_coupling_structure,
    predict_compressibility,
    validate_compressibility,
    run_compressibility_analysis,
    compute_deficiency,
    compute_interaction_graph_bandwidth,
    run_deficiency_analysis,
)

__all__ = [
    "ground_truth_check",
    "compressibility_survey",
    "BenchmarkResult",
    "run_scaling_benchmark",
    "run_accuracy_benchmark",
    "run_all_benchmarks",
    "run_cross_validation",
    "export_to_prism",
    "generate_prism_crossval_report",
    "analyze_coupling_structure",
    "predict_compressibility",
    "validate_compressibility",
    "run_compressibility_analysis",
    "compute_deficiency",
    "compute_interaction_graph_bandwidth",
    "run_deficiency_analysis",
]
