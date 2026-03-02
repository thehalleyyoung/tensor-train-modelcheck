# TN-Check: Certified CSL Model Checking on Tensor-Train CME States

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

**TN-Check** verifies Continuous Stochastic Logic (CSL) properties on stochastic
reaction networks by operating directly on tensor-train (MPS) compressed
probability vectors — never materializing the exponential state space.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Network File Formats](#network-file-formats) (YAML · JSON · SBML)
- [Reaction Kinetics](#reaction-kinetics) (Mass-Action · Hill · Michaelis-Menten)
- [CSL Property Syntax](#csl-property-syntax)
- [Output Formats](#output-formats)
- [Tensor-Train Algorithm](#tensor-train-algorithm)
- [Theoretical Results](#theoretical-results)
- [Examples](#examples)
- [Python API](#python-api)
- [Architecture](#architecture)
- [Testing](#testing)
- [References](#references)

---

## Quick Start

```bash
pip install -e ".[dev]"
tn-check check examples/gene_expression.yaml "P>=0.9 [true U<=10.0 X_0 >= 3]"
tn-check check examples/gene_expression.yaml --steady-state mRNA 3 -f json
tn-check check-network examples/birth_death.sbml
```

## Installation

Requires Python 3.10+, NumPy, SciPy, Click, PyYAML.

```bash
git clone <repository-url>
cd csl-model-checking-on-tensor-train-cme
pip install -e ".[dev]"
tn-check --version
```

---

## CLI Reference

### `tn-check check`

Check a CSL property against a reaction network.

```
tn-check check [OPTIONS] NETWORK_FILE [PROPERTY]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-chi`, `--max-bond-dim` | 200 | Maximum TT bond dimension |
| `-o`, `--output PATH` | — | Write JSON result to file |
| `-f`, `--output-format` | text | Output format: `text`, `json`, `csv` |
| `--steady-state SPECIES THRESH` | — | Shorthand: S>=0.5 [X_i >= THRESH] |
| `--extinction SPECIES` | — | Shorthand: P>=0.5 [true U<=100 X_i <= 0] |
| `-v`, `--verbose` | — | Enable verbose logging |

```bash
tn-check check examples/gene_expression.yaml "P>=0.9 [true U<=10.0 X_0 >= 3]"
tn-check check examples/gene_expression.yaml --steady-state mRNA 3 -f json
tn-check check examples/toggle_switch.yaml --extinction X1
```

### `tn-check verify`

Like `check`, but also produces a formal verification certificate.

```
tn-check verify [OPTIONS] NETWORK_FILE [PROPERTY]
```

Same options as `check`, plus `-c`/`--certificate PATH` (default: `certificate.json`).

```bash
tn-check verify examples/toggle_switch.yaml "S>=0.5 [X_0 >= 10]" -c cert.json
```

### `tn-check info`

Display network statistics (species, reactions, state space size).

```bash
tn-check info examples/enzyme_kinetics.yaml
```

### `tn-check init-network`

Generate a starter YAML template with mass-action, Hill, and Michaelis-Menten examples.

```bash
tn-check init-network my_network.yaml
```

### `tn-check check-network`

Validate a network file (YAML, JSON, or SBML) without running model checking.

```bash
tn-check check-network examples/gene_expression.yaml
tn-check check-network examples/birth_death.sbml
```

---

## Network File Formats

### YAML Format

```yaml
name: my_model
description: "Optional description"
species:
  - name: A
    max_copy_number: 50    # FSP truncation bound (default: 50)
    initial_count: 0
reactions:
  - name: production
    reactants: {}
    products: {A: 1}
    rate_constant: 1.0
    kinetics: mass_action
```

Required: `species[].name`, `reactions[].name`. See `examples/` for complete files.

### JSON Format

Same structure as YAML, encoded as JSON:

```json
{
  "name": "my_model",
  "species": [{"name": "A", "max_copy_number": 50, "initial_count": 0}],
  "reactions": [{
    "name": "production", "reactants": {}, "products": {"A": 1},
    "rate_constant": 1.0, "kinetics": "mass_action"
  }]
}
```

### SBML Format

TN-Check imports [SBML](http://sbml.org/) (Level 2/3) XML files — the
standard format for biological models. Files with `.xml` or `.sbml`
extensions are auto-detected.

```xml
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="birth_death">
    <listOfSpecies>
      <species id="X" initialAmount="10"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="birth" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <listOfLocalParameters>
            <localParameter id="k" value="1.0"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
```

**Import details:**
- Initial amounts are rounded to integers for the discrete CME.
- Mass-action rates are extracted from kinetic law parameters (`k`, `rate`, `c`, `kf`, etc.).
- Hill kinetics are auto-detected when parameters `Vmax`, `K`/`K_half`, and `n`/`hill` are present.
- Reversible reactions generate forward + reverse pairs automatically.
- Use models from [BioModels](https://www.ebi.ac.uk/biomodels/) or export from COPASI, CellDesigner, etc.

---

## Reaction Kinetics

### Mass-Action Kinetics

Propensity = rate constant × falling factorials of reactant copy numbers.

```yaml
- name: binding
  reactants: {E: 1, S: 1}    # a(n) = k · n_E · n_S
  products: {ES: 1}
  rate_constant: 0.01
  kinetics: mass_action
```

| Order | Reactants | Propensity |
|-------|-----------|------------|
| 0th | `{}` | k |
| 1st | `{A: 1}` | k · n_A |
| 2nd | `{A: 1, B: 1}` | k · n_A · n_B |
| 2nd (homodimer) | `{A: 2}` | k · n_A · (n_A-1) / 2 |

### Hill Kinetics

Cooperative binding: `a(n) = V_max · n^h / (K^h + n^h)` (activation) or
`V_max · K^h / (K^h + n^h)` (repression).

```yaml
- name: regulated
  products: {B: 1}
  kinetics: hill
  hill:
    regulator: A
    v_max: 10.0
    k_half: 5.0
    hill_coefficient: 2.0
    activation: true    # false for repression
```

### Michaelis-Menten Kinetics

Enzymatic: `a(n_S, n_E) = k_cat · n_E · n_S / (K_M + n_S)`.

```yaml
- name: enzymatic
  kinetics: michaelis_menten
  michaelis_menten:
    substrate: S
    product: P
    enzyme: E
    k_cat: 1.0
    k_m: 10.0
```

---

## CSL Property Syntax

Species are referenced by index (`X_0`, `X_1`, …) in network definition order.

**Atomic propositions:** `X_0 >= 5`, `X_1 < 10`, `X_2 == 0`

**Boolean:** `true`, `!phi` (negation), `phi1 & phi2` (conjunction)

**Probability operator:** `P~p [path]` where `~` ∈ {`>=`, `>`, `<=`, `<`}

```
P>=0.9 [true U<=10.0 X_0 >= 3]     # prob ≥ 0.9 that X_0 reaches 3 within t=10
P>=0.5 [X_0 >= 1 U<=5.0 X_1 >= 2]  # bounded until with precondition
P<0.01 [true U X_0 >= 100]          # unbounded until
P>=0.9 [F<=10.0 X_0 >= 3]           # eventually (sugar for true U<=t)
```

**Steady-state operator:** `S~p [phi]`

```
S>=0.5 [X_0 >= 10]    # long-run prob of X_0 ≥ 10 is at least 0.5
```

**CLI shorthands** (instead of full CSL):

```bash
--steady-state mRNA 5    # → S>=0.5 [X_i >= 5]   (species by name)
--extinction Protein     # → P>=0.5 [true U<=100.0 X_i <= 0]
```

---

## Output Formats

Use `--output-format` / `-f` with `check` or `verify`:

- **text** (default): human-readable table
- **json**: machine-readable JSON object
- **csv**: single-row CSV with header

```bash
tn-check check network.yaml "S>=0.5 [X_0 >= 3]" -f json
tn-check check network.yaml "S>=0.5 [X_0 >= 3]" -f csv
```

---

## Tensor-Train Algorithm

### Overview

The CME describes time evolution of probability over discrete molecular
counts: `dp/dt = Q · p`. For N species with up to d counts each, the
state space is d^N. TN-Check avoids this exponential blow-up using
**tensor-train decomposition** (MPS):

```
p(n₁, …, nₙ) = G₁[n₁] · G₂[n₂] · … · Gₙ[nₙ]
```

Each `Gₖ` is a (χₖ₋₁ × χₖ) matrix. Bond dimension χ gives O(N·d·χ²)
parameters instead of d^N.

### Pipeline

1. **CME Compilation**: Each reaction → MPO. Mass-action propensities factor
   into rank-1 per-species tensors. Full generator Q = sum of per-reaction MPOs.

2. **TT Compression**: Truncated SVD sweeps. Eckart-Young guarantees error
   equals sum of discarded singular values. Non-negative clamping preserves
   probability semantics (Proposition 1).

3. **TDVP Integration**: 1-site (fixed rank) or 2-site (adaptive rank) TDVP
   projects dynamics onto the TT manifold. Stiffness-aware step-size adaptation.

4. **Error Certification**: Theorem 7 combines FSP truncation (ε_FSP),
   TT-SVD truncation (ε_TT), and TDVP projection (ε_TDVP) into:
   `ε_total = ε_FSP + Σε_TT + √(2K_eff) · Σrₖ · Δt`

   Three-valued semantics: if `[p-ε, p+ε]` straddles the threshold, verdict
   is INDETERMINATE rather than a potentially incorrect TRUE/FALSE.

---

## Theoretical Results

| # | Result | Statement |
|---|--------|-----------|
| T1 | Error contraction | Metzler contractivity → linear (not exponential) error growth |
| T2 | Bond preservation | Axis-aligned predicates: rank-1 TT masks, χ preserved |
| T3 | Linear predicates | Carry-based TT with bond dim ≤ R+1 |
| T4 | Complexity | O(N·d·χ³) vs explicit O(d^N) |
| T5 | Compressibility | Product-form, weak coupling, area-law → bounded rank |
| T6 | Deficiency-zero | δ=0 + weak reversibility → TT rank 1 |
| T7 | Unified error | ε_FSP + Σε_TT + √(2K_eff)·Σrₖ·Δt |
| — | FSP bound | 1 - ‖e^{Q·t}p₀‖₁ ≤ 1 - e^{-λ_exit·t} |
| — | M-χ coupling | TT rank grows ≤ log(M) with FSP dimension |
| P1 | Clamping | Clamping negatives never increases L1 error |
| P2 | Frob→L1 | Converts TDVP residuals to L1 error |

---

## Examples

### Gene Expression (2 species)

```bash
tn-check info examples/gene_expression.yaml
tn-check check examples/gene_expression.yaml "P>=0.9 [true U<=10.0 X_0 >= 3]"
tn-check check examples/gene_expression.yaml --steady-state mRNA 3 -f json
```

### Toggle Switch (Hill kinetics, bistable)

```bash
tn-check verify examples/toggle_switch.yaml "S>=0.3 [X_0 >= 30]" -c cert.json
```

### Enzyme Kinetics (4 species)

```bash
tn-check check examples/enzyme_kinetics.yaml "P>=0.9 [true U<=20.0 X_3 >= 5]"
```

### SBML Models

```bash
tn-check check-network model.sbml
tn-check check model.sbml "P>=0.9 [true U<=10.0 X_0 >= 5]"
tn-check info model.xml
```

---

## Python API

```python
from tn_check.network_loader import load_network
from tn_check.cme import compile_network_to_mpo, deterministic_initial_state
from tn_check.checker.csl_ast import parse_csl
from tn_check.checker.model_checker import CSLModelChecker

net = load_network("examples/gene_expression.yaml")  # or .sbml
mpo = compile_network_to_mpo(net, max_bond_dim=200)
init = deterministic_initial_state(net)
formula = parse_csl("P>=0.9 [true U<=10.0 X_0 >= 3]")
checker = CSLModelChecker(mpo, physical_dims=tuple(net.physical_dims))
result = checker.check(formula, init, max_bond_dim=200)
print(f"Verdict: {result.verdict.value}, Error: {result.total_error:.2e}")
```

Built-in models and experiments:

```python
from tn_check.experiments import run_birth_death_experiment
from tn_check.evaluation.graph_predictors import predict_compressibility_graph
from tn_check.models.library import gene_expression

result = run_birth_death_experiment()
pred = predict_compressibility_graph(gene_expression())
print(f"Predicted rank: {pred.predicted_rank_bound}, score: {pred.compressibility_score:.2f}")
```

See [API.md](API.md) for the full reference.

---

## Architecture

```
src/tn_check/
├── cme/            CME compiler (ReactionNetwork → MPO, FSP bounds)
├── checker/        CSL checker (parser, three-valued semantics, spectral gap)
├── integrator/     TDVP 1-site/2-site, Krylov integrators
├── tensor/         TT/MPS engine (SVD compression, canonical forms)
├── error/          Error certification (truncation, clamping, propagation)
├── verifier/       Independent verifier (11 soundness checks, JSON traces)
├── models/         51 biological models
├── evaluation/     Benchmarks, PRISM cross-validation, graph predictors
├── coupling/       FSP-TT coupling analysis
├── stiffness/      Stiffness-aware adaptive TDVP
├── smc/            Statistical model checking (SPRT) baseline
├── cli.py          Command-line interface
└── network_loader.py  YAML/JSON/SBML network loader
```

---

## Testing

```bash
python3 -m pytest tests/ -q                         # All tests
python3 -m pytest tests/ -q --timeout=30             # With timeout
python3 -m pytest tests/test_checker.py -q           # CSL checker only
python3 -m pytest tests/test_path_b_improvements.py  # Path B only
```

Key results: 11/11 certificate checks SOUND, 0 cross-validation disagreements
across 30 CSL properties on 10 models, birth-death L1 error = 0.0.

## Grounding

Every paper claim is mapped to code, proof, and numerical verification in
[`docs/meta/grounding.json`](docs/meta/grounding.json).

---

## References

1. Kazeev & Schwab (2014). Quantized TT finite elements. *Numer. Math.*
2. Oseledets (2011). Tensor-train decomposition. *SIAM J. Sci. Comput.*
3. Baier et al. (2003). Model-checking CTMCs. *IEEE TSE.*
4. Ion et al. (2021). TT approximation of the CME. *J. Chem. Phys.*
5. Kwiatkowska et al. (2011). PRISM 4.0. *QEST.*
6. Feinberg (1972). Complex balancing. *Arch. Rational Mech. Anal.*
7. Anderson, Craciun & Kurtz (2010). Product-form distributions. *Bull. Math. Biol.*
8. Munsky & Khammash (2006). FSP method for CME. *J. Chem. Phys.*
9. Wald (1945). Sequential tests. *Ann. Math. Statist.*
10. Haegeman et al. (2016). TDVP for MPS. *Phys. Rev. B.*
11. Hucka et al. (2003). SBML. *Bioinformatics.*
