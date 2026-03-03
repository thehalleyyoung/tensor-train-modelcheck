"""
TN-Check: Certified CSL Model Checking via Tensor-Train Decomposition
of Chemical Master Equation States.

This package implements CSL model-checking algorithms natively on
tensor-train (TT) / matrix product state (MPS) compressed probability
vectors for Chemical Master Equation (CME) states, enabling certified
probabilistic verification of stochastic biochemical systems at scales
far beyond explicit enumeration.

Key features:
- CSL model checking on MPS-compressed probability vectors
- Certified error bounds via Metzler contractivity (Theorem 1)
- Non-negativity-preserving TT rounding (resolves clamping gap)
- Spectral-gap-informed fixpoint convergence for unbounded-until
- Independent certificate verification via verification traces
- Three-valued semantics for nested probability operators
- FSP-TT coupling analysis with joint M-χ optimization
- Stiffness-aware adaptive TDVP integration
- Graph-theoretic compressibility predictors (tree-width, deficiency, modularity)
- Statistical model checking (SPRT) comparison baseline
- Error bound tightness characterization
"""

__version__ = "0.3.0"
__author__ = "TN-Check Authors"

from tn_check.config import TNCheckConfig
