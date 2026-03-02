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
| Cross-validation verdicts | 0 disagreements across 30 CSL properties on 10 models |
| Tests | 205+ passing |
| Biological models | 51 (13 core + 12 extended + 26 broad coverage) |
| Theoretical results | 9 (7 theorems + 2 propositions) |

## Theoretical Contributions

1. **Theorem 1** (Error contraction): Metzler semigroup contractivity
   ensures per-step truncation errors accumulate linearly, not exponentially.

2. **Theorem 2** (Bond-dimension preservation): Axis-aligned predicates
   produce rank-1 TT masks; `χ(1_Φ ⊙ p) ≤ χ(p)`.

3. **Theorem 3** (Linear predicates): Carry-based TT construction with
   bond dimension ≤ R+1.

4. **Theorem 4** (Complexity separation): TT-based verification is O(N·d·χ³)
   vs explicit O(d^N).

5. **Theorem 5** (Compressibility): Three sufficient conditions for bounded
   TT rank — product-form equilibria, weak coupling with ε-perturbation,
   and area-law entanglement entropy S ≤ c·log(χ). Verified via entanglement
   entropy computation across all 20+ models.

6. **Theorem 6** (Deficiency-zero compressibility): Networks with deficiency
   δ=0 and weak reversibility have product-form stationary distributions
   (TT rank 1). Provides a purely graph-theoretic sufficient condition for
   compressibility — computed from reaction structure alone, without solving
   the CME. Based on Feinberg (1972) and Anderson et al. (2010).

7. **Theorem 7** (Unified error propagation): Combines FSP truncation,
   TT-SVD truncation, and TDVP projection into a single certified L1 bound:
   `ε_total = ε_FSP + Σε_TT + √(2K_eff)·Σr_k·Δt`. All terms accumulate
   linearly due to Metzler contractivity. Clamping does not add to the bound.

8. **Proposition 1** (Clamping soundness): Post-truncation clamping of
   negative entries *never increases* L1 error — proved via index-wise
   comparison (Lemma 2), not triangle inequality.

9. **Proposition 2** (Frobenius-to-L1 bound): Converts TDVP Frobenius
   residuals to CSL-relevant L1 error: `‖Δp‖₁ ≤ √(2K_eff) · ∫‖res‖_F dt`.

## Cross-Validation & Verification

- **Dense cross-validation**: 30 CSL properties across 10 models — zero
  verdict disagreements; 22 agree, 8 indeterminate (sound), 0 disagree.
  PRISM-format model export (`.pm` + `.csl`) for external validation with
  PRISM/Storm. Verdict comparison table via `run_verdict_comparison_table()`.
- **Deficiency analysis**: Graph-theoretic compressibility pre-check on all
  51 models; 15+ have deficiency zero, 8 are also weakly reversible.
- **Compressibility theorem**: Entanglement entropy computation (von Neumann
  + Rényi-2) across all bipartitions; three sufficient conditions checked
  per model with formal justification.
- **Verdict-sensitivity analysis**: Asymmetric clamping bias effect on
  P≥p (false TRUE risk) vs P≤p (conservative/sound) verdicts.
- **Cumulative bias tracking**: Per-step clamping bias rates with trend
  detection (growing/stable/decaying) and long-horizon projections.
- **TDVP convergence**: Frobenius residual tracking during evolution with
  L1-from-Frobenius bound; backward error analysis; conservation error
  monitoring.
- **Spectral gap circularity**: Weyl's theorem application to bound the
  spectral gap from truncated operators.
- **Mathematical rigor suite**: Automated numerical verification of all
  9 theoretical claims across randomized instances.

## Architecture

```
src/tn_check/
├── cme/            CME compiler (ReactionNetwork → MPO generator, FSP bounds)
├── checker/        CSL checker (parser, bounded/unbounded until, fixpoint,
│                   three-valued semantics, spectral-gap convergence)
├── integrator/     Time integrators (TDVP, Krylov, Euler, DMRG steady-state)
├── tensor/         TT/MPS engine (MPS/MPO, SVD compression, canonical forms)
├── error/          Error certification (truncation, clamping bounds,
│                   cumulative tracker, Frobenius-to-L1, spectral gap)
├── verifier/       Independent verifier (11 soundness checks, JSON traces)
├── models/         Model library (51 biological models)
├── evaluation/     Benchmarks, PRISM cross-validation, compressibility
│                   theorem, deficiency analysis, math rigor suite
├── solver/         Dense reference solver (expm-based)
├── adaptive/       Adaptive bond-dimension controller
├── ordering/       Species ordering strategies
├── config.py       Master configuration
├── diagnostics.py  Marginals, moments, validation
└── experiments.py  Reproducible experiment runners
```

## Testing

```bash
python3 -m pytest tests/ -q           # 205 tests, quick summary
python3 -m pytest tests/ -v           # Verbose output
python3 -m pytest tests/ --tb=short   # Quick pass/fail
```

## Grounding

Every paper claim is mapped to code, proof, and numerical verification in
[`docs/meta/grounding.json`](docs/meta/grounding.json). The critique triage
and review responses are in [`docs/meta/`](docs/meta/).

## References

1. Kazeev & Schwab (2014). Quantized TT finite elements. *Numer. Math.*
2. Oseledets (2011). Tensor-train decomposition. *SIAM J. Sci. Comput.*
3. Baier et al. (2003). Model-checking CTMCs. *IEEE TSE.*
4. Ion et al. (2021). TT approximation of the CME. *J. Chem. Phys.*
5. Kwiatkowska et al. (2011). PRISM 4.0. *QEST.*
6. Gillespie (1992). Markov processes. *Academic Press.*
7. Feinberg (1972). Complex balancing in general kinetic systems. *Arch. Rational Mech. Anal.*
8. Anderson, Craciun & Kurtz (2010). Product-form stationary distributions for deficiency zero networks. *Bull. Math. Biol.*
