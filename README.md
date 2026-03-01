# TN-Check: Certified CSL Model Checking on Tensor-Train CME States

**TN-Check** verifies Continuous Stochastic Logic (CSL) properties on stochastic
reaction networks by operating directly on tensor-train (MPS) compressed
probability vectors — never materializing the exponential state space.

## 30-Second Quickstart

```bash
pip install -e ".[dev]"
python3 -c "
from tn_check.experiments import run_birth_death_experiment, run_compression_accuracy_experiment
bd = run_birth_death_experiment()
print(f'Birth-death L1 error: {bd[\"l1_error\"]}')
ca = run_compression_accuracy_experiment()
for r in ca['compression_results']:
    print(f'  chi={r[\"bond_dim\"]}: L1={r[\"l1_error\"]:.2e}')
"
```

Expected output:
```
Birth-death L1 error: 0.0
  chi=1: L1=2.35e-01
  chi=2: L1=4.71e-02
  chi=5: L1=4.88e-05
  chi=10: L1=1.62e-12
  chi=20: L1=1.62e-12
  chi=50: L1=1.62e-12
```

## Key Results

| Result | Value |
|--------|-------|
| Birth-death L1 error | 0.0 (exact Poisson recovery) |
| Gene expression accuracy at χ=10 | L1 = 1.62×10⁻¹² |
| Compression ratio at N=10 species | 7.8×10¹³× |
| Certificate verification | 11/11 checks SOUND |
| Tests | 161 passing |
| Biological models | 13 |

## Theoretical Contributions

1. **Proposition 1** (Clamping reduces L1 error): Post-truncation clamping
   of negative entries *never increases* L1 error. Bound is ε_trunc, not
   2ε_trunc — proved via index-wise comparison (Lemma 2), not triangle
   inequality.

2. **Theorem 1** (Error contraction): Metzler semigroup contractivity
   ensures per-step truncation errors accumulate linearly, not exponentially.

3. **Theorem 2** (Bond-dimension preservation): Axis-aligned predicates
   produce rank-1 TT masks.

4. **Theorem 3** (Linear predicates): Carry-based TT construction with
   bond dimension ≤ R+1.

5. **Theorem 4** (Complexity separation): TT-based verification is O(N)
   vs explicit O(M^N).

## Architecture

```
src/tn_check/
├── cme/            CME Compiler
│   ReactionNetwork, → MPO generator, FSP bounds
├── checker/        CSL Checker
│   CSL parser, bounded/unbounded until, fixpoint iteration,
│   three-valued semantics, spectral-gap convergence prediction
├── integrator/     Time Integrators
│   TDVP, Krylov, Uniformization, Euler, DMRG steady-state
├── tensor/         TT/MPS Engine
│   MPS/MPO data structures, SVD compression, canonical forms
├── error/          Error Certification
│   Truncation tracking, clamping bounds (Prop. 1), Lemma 2
├── verifier/       Independent Verifier
│   11 soundness checks, JSON trace serialization
├── models/         Model Library (13 biological models)
├── evaluation/     Benchmarks & PRISM comparison
├── solver/         Dense reference solver
├── adaptive/       Adaptive bond-dimension controller
├── ordering/       Species ordering strategies
├── config.py       Master configuration
├── diagnostics.py  Marginals, moments, validation
└── experiments.py  Reproducible experiment runners
```

## Testing

```bash
python3 -m pytest tests/ -v           # 161 tests
python3 -m pytest tests/ --tb=short   # Quick pass/fail
```

## References

1. Kazeev & Schwab (2014). Quantized TT finite elements. *Numer. Math.*
2. Oseledets (2011). Tensor-train decomposition. *SIAM J. Sci. Comput.*
3. Baier et al. (2003). Model-checking CTMCs. *IEEE TSE.*
4. Ion et al. (2021). TT approximation of the CME. *J. Chem. Phys.*
