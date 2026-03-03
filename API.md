# TN-Check API Reference

---

## Table of Contents

- [Configuration (`tn_check.config`)](#configuration)
- [Tensor: MPS (`tn_check.tensor.mps`)](#mps--matrix-product-states)
- [Tensor: MPO (`tn_check.tensor.mpo`)](#mpo--matrix-product-operators)
- [Tensor: Operations (`tn_check.tensor.operations`)](#operations)
- [Tensor: Canonical (`tn_check.tensor.canonical`)](#canonical-forms)
- [Tensor: Decomposition (`tn_check.tensor.decomposition`)](#decomposition)
- [Tensor: Algebra (`tn_check.tensor.algebra`)](#algebra)
- [CME: Reaction Network (`tn_check.cme.reaction_network`)](#reaction-network)
- [CME: Compiler (`tn_check.cme.compiler`)](#compiler)
- [CME: Initial State (`tn_check.cme.initial_state`)](#initial-state)
- [Checker: CSL AST (`tn_check.checker.csl_ast`)](#csl-ast)
- [Checker: Model Checker (`tn_check.checker.model_checker`)](#model-checker)
- [Checker: Satisfaction (`tn_check.checker.satisfaction`)](#satisfaction)
- [Checker: Spectral (`tn_check.checker.spectral`)](#spectral-analysis)
- [Error: Certification (`tn_check.error.certification`)](#certification)
- [Error: Propagation (`tn_check.error.propagation`)](#propagation)
- [Verifier: Trace (`tn_check.verifier.trace`)](#trace)
- [Verifier: Checker (`tn_check.verifier.checker`)](#verifier-checker)
- [Diagnostics (`tn_check.diagnostics`)](#diagnostics)
- [Model Library (`tn_check.models.library`)](#model-library)
- [Experiments (`tn_check.experiments`)](#experiments)
- [Coupling: FSP-TT Analysis (`tn_check.coupling`)](#fsp-tt-coupling-analysis)
- [Stiffness: Adaptive TDVP (`tn_check.stiffness`)](#stiffness-aware-tdvp)
- [SMC: Statistical Model Checking (`tn_check.smc`)](#statistical-model-checking)
- [Evaluation: Tightness (`tn_check.evaluation.tightness`)](#error-bound-tightness)
- [Evaluation: Graph Predictors (`tn_check.evaluation.graph_predictors`)](#graph-theoretic-compressibility)
- [Evaluation: Path B Experiments (`tn_check.evaluation.path_b_experiments`)](#path-b-experiments)
- [CLI Reference (`tn-check`)](#cli-reference)
- [Network Loader (`tn_check.network_loader`)](#network-loader)

---

## Configuration

**Module:** `tn_check.config`

**Enums:** `IntegratorType`, `CanonicalForm`, `OrderingMethod`, `CSLSemantics`.

**Sub-configs (dataclasses):** `TTConfig`, `CMEConfig`, `IntegratorConfig`, `CheckerConfig`, `ErrorConfig`, `OrderingConfig`, `AdaptiveConfig`, `EvaluationConfig`.

### `TNCheckConfig`
Master configuration aggregating all sub-configs.

| Method | Description |
|--------|-------------|
| `to_dict() -> dict` | Serialize to dictionary |
| `from_dict(cls, d) -> TNCheckConfig` | Deserialize from dictionary |
| `save(path)` | Save to JSON |
| `load(cls, path) -> TNCheckConfig` | Load from JSON |
| `from_env(cls) -> TNCheckConfig` | Create from environment variables |

```python
def default_config() -> TNCheckConfig          # Default configuration
def quick_config(max_bond_dim=50, max_copy_number=30, integrator="tdvp_2site") -> TNCheckConfig
```

---

## MPS — Matrix Product States

**Module:** `tn_check.tensor.mps`

### `MPS`

```python
MPS(cores: list[NDArray], canonical_form=CanonicalForm.NONE, orthogonality_center=None, copy_cores=True)
```

**Properties:** `num_sites`, `physical_dims`, `bond_dims`, `max_bond_dim`, `total_params`, `full_size`, `compression_ratio`, `dtype`.

**Methods:** `copy()`, `get_core(site)`, `set_core(site, core)`, `scale(factor)`, `negate()`, `bond_spectrum(bond)`, `entanglement_entropy(bond)`, `renyi_entropy(bond, alpha=2.0)`, `invalidate_cache()`, `info()`, `validate()`.

### Factory Functions

```python
random_mps(num_sites, physical_dims, bond_dim, normalize=True, seed=None, dtype=np.float64) -> MPS
zero_mps(num_sites, physical_dims) -> MPS
ones_mps(num_sites, physical_dims) -> MPS
unit_mps(num_sites, physical_dims, indices) -> MPS           # Rank-1 basis vector
product_mps(vectors: list[NDArray]) -> MPS                   # Rank-1 from per-site vectors
uniform_mps(num_sites, physical_dims) -> MPS                 # Normalized uniform
characteristic_mps(num_sites, physical_dims, site, predicate_mask) -> MPS  # Single-site indicator
multi_site_characteristic_mps(num_sites, physical_dims, site_masks) -> MPS # Multi-site indicator
threshold_mps(num_sites, physical_dims, site, threshold, direction="greater") -> MPS
interval_mps(num_sites, physical_dims, site, low, high) -> MPS
```

---

## MPO — Matrix Product Operators

**Module:** `tn_check.tensor.mpo`

### `MPO`

```python
MPO(cores: list[NDArray], copy_cores=True)
```

**Properties:** `num_sites`, `physical_dims_in`, `physical_dims_out`, `bond_dims`, `max_bond_dim`, `total_params`, `is_square`, `dtype`.

**Methods:** `copy()`, `get_core(site)`, `set_core(site, core)`, `scale(factor)`, `compress(max_bond_dim=None, tolerance=1e-12)`, `check_column_sum_zero(tol)`, `check_metzler(tol)`, `check_hermitian(tol)`, `validate()`.

### Factory Functions

```python
identity_mpo(num_sites, physical_dims) -> MPO
zero_mpo(num_sites, physical_dims) -> MPO
scalar_mpo(num_sites, physical_dims, scalar) -> MPO         # scalar * I
random_mpo(num_sites, physical_dims, bond_dim, seed=None, hermitian=False) -> MPO
diagonal_mpo(diagonal_mps: MPS) -> MPO
creation_mpo(num_sites, physical_dims, site) -> MPO          # Bosonic creation at site
annihilation_mpo(num_sites, physical_dims, site) -> MPO      # Bosonic annihilation at site
number_mpo(num_sites, physical_dims, site) -> MPO            # Number operator at site
shift_mpo(num_sites, physical_dims, site, shift=1) -> MPO
propensity_diagonal_mpo(num_sites, physical_dims, site, propensity_values) -> MPO
```

---

## Operations

**Module:** `tn_check.tensor.operations`

### Core Arithmetic

```python
mps_inner_product(mps_a, mps_b) -> float        # ⟨a|b⟩
mps_norm(mps) -> float                           # ‖mps‖₂
mps_addition(mps_a, mps_b) -> MPS
mps_scalar_multiply(mps, scalar) -> MPS
mps_hadamard_product(mps_a, mps_b) -> MPS        # Element-wise product
mps_weighted_sum(mps_list, weights, max_bond_dim=None) -> MPS
mps_distance(mps_a, mps_b) -> float              # Euclidean distance
mps_total_variation_distance(mps_a, mps_b) -> float
```

### MPO–MPS Contraction

```python
mpo_mps_contraction(mpo, mps) -> MPS             # Apply MPO to MPS
mpo_mpo_contraction(mpo_a, mpo_b) -> MPO         # Compose two MPOs
mps_zip_up(mpo, mps, max_bond_dim=None, tolerance=1e-12) -> tuple[MPS, float]  # Memory-efficient
mps_expectation_value(mpo, mps) -> float          # ⟨mps|mpo|mps⟩
mpo_mps_expectation(mpo, mps_bra, mps_ket) -> float  # ⟨bra|mpo|ket⟩
```

### Compression & Normalization

```python
mps_compress(mps, max_bond_dim=None, tolerance=1e-10, relative=True) -> tuple[MPS, float]
mps_clamp_nonnegative(mps, in_place=False) -> tuple[MPS, float]
mps_normalize_probability(mps, in_place=False) -> tuple[MPS, float]
```

### Query & Conversion

```python
mps_probability_at_index(mps, indices) -> float
mps_marginalize(mps, keep_sites) -> MPS
mps_total_probability(mps) -> float
mps_entanglement_entropy(mps) -> list[float]
mps_bond_dimensions(mps) -> list[int]
mps_to_dense(mps) -> NDArray
mpo_to_dense(mpo) -> NDArray
```

### MPO Arithmetic

```python
mpo_addition(mpo_a, mpo_b) -> MPO
mpo_scalar_multiply(mpo, scalar) -> MPO
```

### Environment Tensors

```python
compute_transfer_matrix(mpo, mps, site) -> NDArray
left_environment(mps_bra, mps_ket, up_to_site) -> NDArray
right_environment(mps_bra, mps_ket, from_site) -> NDArray
mpo_left_environment(mpo, mps_bra, mps_ket, up_to_site) -> NDArray
mpo_right_environment(mpo, mps_bra, mps_ket, from_site) -> NDArray
build_all_environments(mpo, mps_bra, mps_ket) -> tuple[list[NDArray], list[NDArray]]
```

---

## Canonical Forms

**Module:** `tn_check.tensor.canonical`

```python
left_canonicalize(mps) -> MPS                    # All cores left-orthogonal
right_canonicalize(mps) -> MPS                   # All cores right-orthogonal
mixed_canonicalize(mps, center) -> MPS           # Mixed-canonical with center
move_orthogonality_center(mps, new_center) -> MPS
normalize_mps(mps) -> tuple[MPS, float]
svd_compress(mps, max_bond_dim=None, tolerance=1e-10) -> tuple[MPS, float]
svd_truncate_bond(mps, bond, max_bond_dim=None, tolerance=1e-10) -> float
qr_left_sweep(mps, start=0, end=None) -> MPS
qr_right_sweep(mps, start=None, end=0) -> MPS
gauge_transform(mps, gauge_ops: list[NDArray]) -> MPS
schmidt_decomposition(mps, bond) -> tuple[NDArray, NDArray, NDArray]
two_site_tensor(mps, bond) -> NDArray            # Contract two adjacent cores
split_two_site_tensor(tensor, chi_l, chi_r, max_bond_dim=None, tolerance=1e-10) -> tuple[NDArray, NDArray, float]
```

---

## Decomposition

**Module:** `tn_check.tensor.decomposition`

```python
svd_truncate(matrix, max_rank=None, tolerance=1e-10, relative=True) -> tuple[NDArray, NDArray, NDArray, float]
adaptive_svd_truncate(matrix, target_error=1e-10, max_rank=None) -> tuple[NDArray, NDArray, NDArray, float, int]
tensor_to_mps(tensor, physical_dims, max_bond_dim=None, tolerance=1e-10) -> MPS  # TT-SVD
matrix_to_mpo(matrix, physical_dims_in, physical_dims_out, max_bond_dim=None, tolerance=1e-10) -> MPO
randomized_svd(matrix, rank, oversampling=10, power_iterations=2) -> tuple[NDArray, NDArray, NDArray]
incremental_svd_update(U, S, Vt, new_column, new_row) -> tuple[NDArray, NDArray, NDArray]
```

---

## Algebra

**Module:** `tn_check.tensor.algebra`

```python
kronecker_product_mpo(mpo_list) -> MPO           # Tensor product of MPOs
sum_mpo(mpo_list, weights=None, compress=False, max_bond_dim=None, tolerance=1e-12) -> MPO
mpo_transpose(mpo) -> MPO
mpo_hermitian_conjugate(mpo) -> MPO
mpo_trace(mpo) -> float
mps_outer_product(mps_a, mps_b) -> MPO           # |a⟩⟨b|
mpo_hadamard_product(mpo_a, mpo_b) -> MPO
mpo_power(mpo, power, max_bond_dim=None) -> MPO
apply_diagonal_mask_to_mpo(mpo, mask_mps) -> MPO
apply_column_mask_to_mpo(mpo, mask_mps, column_index) -> MPO
project_mpo_to_subspace(mpo, projection_mps) -> MPO
```

---

## Reaction Network

**Module:** `tn_check.cme.reaction_network`

**Enum:** `KineticsType` — `MASS_ACTION`, `HILL`, `MICHAELIS_MENTEN`, `CUSTOM`.

### `Species` (dataclass)
Fields: `name`, `index`, `max_copy_number`, `initial_count`, `description`, `is_conserved`, `compartment`.

### Propensity Functions

```python
class PropensityFunction:           # Base class
    evaluate(copy_numbers) -> float
    per_species_factors(species_indices, max_copy_numbers) -> list[NDArray]
    involved_species() -> list[int]

class MassActionPropensity(PropensityFunction):
    __init__(rate_constant, reactant_species, reactant_stoichiometry)

class HillPropensity(PropensityFunction):
    __init__(v_max, k_half, hill_coefficient, species_index, activation=True)

class MichaelisMentenPropensity(PropensityFunction):
    __init__(k_cat, k_m, enzyme_index, substrate_index)

class CustomPropensity(PropensityFunction):
    __init__(func, ...)
```

### `Reaction`
Chemical reaction with stoichiometry and propensity function.

### `ReactionNetwork`
Container managing species and reactions, computing stoichiometry matrices, and detecting conservation laws.

---

## Compiler

**Module:** `tn_check.cme.compiler`

### `CMECompiler`

```python
CMECompiler(network, max_bond_dim=None, compression_tolerance=1e-12, use_conservation_laws=True)
```

**Properties:** `physical_dims`, `num_sites`.  
**Methods:** `compile() -> MPO` — compile full CME generator.

### `CMEGeneratorAnalyzer`
Analysis utilities for compiled generator MPO.

### Functions

```python
compile_reaction_to_mpo(reaction, network) -> MPO
compile_network_to_mpo(network, ...) -> MPO
compile_propensity_to_diagonal(propensity, network) -> MPO
build_uniformization_mpo(generator, uniformization_rate) -> MPO
```

---

## Initial State

**Module:** `tn_check.cme.initial_state`

```python
deterministic_initial_state(network, initial_counts=None) -> MPS   # Delta distribution
poisson_initial_state(network, means=None) -> MPS                  # Product of Poisson marginals
binomial_initial_state(network, total_counts, probabilities) -> MPS
thermal_initial_state(network, temperature=1.0, energy_function=None) -> MPS
mixture_initial_state(network, distributions, weights) -> MPS
uniform_initial_state(network) -> MPS
```

---

## CSL AST

**Module:** `tn_check.checker.csl_ast`

**Enum:** `ComparisonOp` — `GEQ`, `GT`, `LEQ`, `LT`.

**Formula types (frozen dataclasses):** `CSLFormula` (base), `TrueFormula`, `AtomicProp`, `LinearPredicate`, `Negation`, `Conjunction`, `BoundedUntil`, `UnboundedUntil`, `Next`, `ProbabilityOp`, `SteadyStateOp`.

```python
parse_csl(formula_str: str) -> CSLFormula        # Parse CSL formula from string
```

---

## Model Checker

**Module:** `tn_check.checker.model_checker`

### `ConvergenceDiagnostics` (dataclass)
Methods: `convergence_rate()`, `geometric_convergence_ratio()`.

### `CSLModelChecker`

```python
CSLModelChecker(generator: MPO, config=None, physical_dims=None)
```

**Property:** `trace`.  
**Method:** `check(formula, initial_state, max_bond_dim=None) -> SatisfactionResult`.

---

## Satisfaction

**Module:** `tn_check.checker.satisfaction`

**Enum:** `ThreeValued` — `TRUE`, `FALSE`, `INDETERMINATE`.

### `SatisfactionResult` (dataclass)
Method: `classify(threshold, comparison, epsilon=1e-6) -> ThreeValued`.

### Functions

```python
compute_satisfaction_set(formula, num_sites, physical_dims, species_names=None) -> MPS
project_rate_matrix(generator, sat_set, excluded_set=None) -> MPO
```

---

## Spectral Analysis

**Module:** `tn_check.checker.spectral`

### `SpectralGapEstimate` (dataclass)
Method: `predicted_iteration_count(tolerance) -> int`.

### Functions

```python
rayleigh_quotient_refinement(Q: MPO, v: MPS) -> float        # Eigenvalue refinement
estimate_spectral_gap(Q_proj, num_sites, physical_dims, max_power_steps=50, max_bond_dim=100, tolerance=1e-8) -> SpectralGapEstimate
adaptive_fallback_time_bound(gap_estimate, safety_factor=5.0, min_time=100.0, max_time=100000.0) -> float
```

### `ConvergencePredictor`

```python
ConvergencePredictor(estimate: SpectralGapEstimate)
```

| Method | Description |
|--------|-------------|
| `will_converge(tolerance, max_iter)` | Predict if fixpoint converges |
| `recommended_bond_dim(tolerance, base_dim=10)` | Suggest bond dimension |
| `convergence_certificate()` | Dict with prediction details |

---

## Certification

**Module:** `tn_check.error.certification`

### Dataclasses

- **`ClampingProofIteration`** — per-iteration clamping proof data.
- **`ClampingProof`** — methods: `record()`, `verify()`.
- **`ErrorCertificate`** — methods: `compute_total()`, `is_within_budget()`, `to_dict()`.

### `ErrorTracker`

| Method | Description |
|--------|-------------|
| `record_step(...)` | Record a time-step error |
| `record_clamping(...)` | Record a clamping error |
| `accumulated_truncation_error()` | Total truncation error |
| `accumulated_clamping_error()` | Total clamping error |
| `certify() -> ErrorCertificate` | Produce certified error bound |
| `richardson_extrapolation()` | Richardson extrapolation estimate |

### Functions

```python
nonneg_preserving_round(mps, max_bond_dim, tolerance=1e-10, max_iterations=20) -> tuple[MPS, float, ErrorCertificate]
certified_nonneg_round(mps, max_bond_dim, tolerance=1e-10, max_iterations=20) -> tuple[MPS, float, ErrorCertificate, ClampingCertificate]
clamping_error_bound(mps, truncation_epsilon) -> float    # Returns ε_trunc (Lemma 2: clamping reduces error)
estimate_negativity(mps, n_samples=10000) -> float
tight_clamping_bound(mps, truncation_epsilon, n_samples=50000, safety_factor=1.5) -> float
tight_clamping_bound_certified(mps, truncation_epsilon, n_samples=50000) -> ClampingCertificate
verify_clamping_proposition(p_exact_dense, p_svd_dense) -> bool     # Verifies Lemma 1 (negativity bound)
verify_clamping_reduces_error(p_exact_dense, p_svd_dense) -> bool   # Verifies Lemma 2 (clamping reduces L1 error)
```

---

## Propagation

**Module:** `tn_check.error.propagation`

### `PropagationAnalysis` (dataclass)

### Functions

```python
semigroup_error_bound(per_step_errors, time_horizon, is_metzler=True) -> PropagationAnalysis
csl_error_propagation(inner_error, threshold, comparison, num_states_near_threshold=0, total_states=1) -> dict
compose_error_certificates(*certificates) -> ErrorCertificate
compose_fsp_tt_errors(fsp_error, per_step_truncation_errors, per_step_clamping_errors=None, threshold=None, is_metzler=True) -> dict
frobenius_to_l1_bound(frobenius_residuals, effective_support_size, dt) -> dict
spectral_gap_circularity_analysis(generator_dense, truncated_generator_dense=None) -> dict
unified_error_theorem(fsp_error, per_step_truncation_errors, frobenius_residuals, dt, effective_support, threshold=0.5) -> dict
```

#### `unified_error_theorem(fsp_error, per_step_truncation_errors, frobenius_residuals, dt, effective_support, threshold=0.5)`

Implements Theorem 7: computes the unified L1 error bound combining FSP truncation, TT-SVD truncation, and TDVP projection errors. All terms accumulate linearly due to Metzler contractivity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fsp_error` | `float` | FSP truncation error ε_FSP |
| `per_step_truncation_errors` | `list[float]` | Per-step TT-SVD truncation errors |
| `frobenius_residuals` | `list[float]` | Per-step TDVP Frobenius residuals |
| `dt` | `float` | Time step size |
| `effective_support` | `int` | Effective support size K_eff |
| `threshold` | `float` | CSL threshold for verdict certification (default 0.5) |

**Returns:** `dict` with keys `total_certified_error`, `error_sources` (dict with `fsp_truncation`, `tt_svd_truncation`, `tdvp_projection`, `clamping`), `error_fractions`, `verdict_certified`, `verdict_margin`, `proof_sketch`.

#### `frobenius_to_l1_bound(frobenius_residuals, effective_support_size, dt)`

Converts TDVP Frobenius residuals to L1 probability error bound.

| Parameter | Type | Description |
|-----------|------|-------------|
| `frobenius_residuals` | `list[float]` | Per-step Frobenius norms of integration residuals |
| `effective_support_size` | `int` | Number of states with non-negligible probability (K_eff) |
| `dt` | `float` | Time step size |

**Returns:** `dict` with keys `l1_bound`, `frobenius_integral`, `conversion_factor`, `effective_support_size`.

#### `spectral_gap_circularity_analysis(generator_dense, truncated_generator_dense=None)`

Analyzes circularity in spectral-gap-based convergence bounds by applying Weyl's theorem.

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator_dense` | `np.ndarray` | Full generator matrix |
| `truncated_generator_dense` | `np.ndarray` or `None` | Truncated generator (optional) |

**Returns:** `dict` with keys `full_gap`, `truncated_gap`, `weyl_bound`, `gap_relative_error`, `is_gap_reliable`.

---

## Cumulative Error Tracking

**Module:** `tn_check.error.cumulative`

### `CumulativeErrorTracker`

| Method | Description |
|--------|-------------|
| `record_step(truncation_error, clamping_error)` | Record errors from one integration step |
| `get_certificate() -> dict` | Return cumulative error certificate |
| `long_horizon_bias_analysis(project_to_time=None) -> dict` | Analyze long-horizon clamping bias trends |

#### `long_horizon_bias_analysis(project_to_time=None)`

Analyzes cumulative clamping bias for trend detection and long-horizon projections.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_to_time` | `float` or `None` | Target time for bias projection (default: 10× current) |

**Returns:** `dict` with keys `total_clamping_bias`, `bias_rate_per_step`, `trend` (growing/stable/decaying), `projected_bias`, `safety_margin`.

#### `verdict_sensitivity_analysis(csl_threshold, csl_comparison='geq')`

Analyzes whether accumulated clamping bias could flip a CSL verdict. Clamping is always mass-adding (positive bias), so its effect on verdicts is asymmetric: P≥p verdicts may have false TRUE risk, while P≤p verdicts are conservative/sound.

| Parameter | Type | Description |
|-----------|------|-------------|
| `csl_threshold` | `float` | CSL probability threshold p |
| `csl_comparison` | `str` | Comparison operator: `'geq'` for P≥p or `'leq'` for P≤p |

**Returns:** `dict` with keys `corrected_probability`, `verdict_safe`, `bias_direction`, `margin`, `recommendation`.

---

## Trace

**Module:** `tn_check.verifier.trace`

**Dataclasses:** `StepRecord`, `FSPBoundRecord`, `CSLCheckRecord`, `ClampingProofRecord` — all have `to_dict()`.

### `VerificationTrace`

| Method | Description |
|--------|-------------|
| `record_step(...)` | Record integration step |
| `record_fsp_bounds(...)` | Record FSP bounds |
| `record_clamping_proof(...)` | Record clamping proof |
| `record_csl_check(...)` | Record CSL check |
| `finalize()` | Finalize trace |
| `to_dict()` | Serialize |
| `to_json(path)` | Save to JSON file |
| `from_json(cls, path)` | Load from JSON file |

---

## Verifier Checker

**Module:** `tn_check.verifier.checker`

**Dataclasses:** `CheckResult`, `VerificationReport` (methods: `add_check()`, `summary()`, `to_dict()`).

### `CertificateVerifier`

```python
CertificateVerifier(probability_tolerance=1e-6, clamping_factor=2.0, max_allowed_error=1.0)
```

Method: `verify(trace: VerificationTrace) -> VerificationReport`.

---

## Diagnostics

**Module:** `tn_check.diagnostics`

```python
marginal_distribution(mps, site) -> NDArray      # Marginal at single site
compute_moments(mps, site, max_order=4) -> dict   # Mean, variance, skewness, kurtosis
kl_divergence_marginal(mps_a, mps_b, site) -> float
total_negative_mass(mps) -> float
validate_probability_vector(mps, tolerance=1e-6) -> dict
```

---

## Model Library

**Module:** `tn_check.models.library`

```python
birth_death(birth_rate=1.0, death_rate=0.1, max_copy=50) -> ReactionNetwork
toggle_switch(alpha1=50.0, alpha2=50.0, beta=2.5, gamma=1.0, max_copy=100) -> ReactionNetwork
repressilator(alpha=50.0, alpha0=0.5, beta=2.0, gamma=1.0, n_genes=3, max_copy=80) -> ReactionNetwork
cascade(n_layers=3, k_activation=1.0, k_deactivation=0.1, max_copy=50) -> ReactionNetwork
schlogl(c1=3e-7, c2=1e-4, c3=1e-3, c4=3.5, max_copy=200) -> ReactionNetwork
gene_expression(k_txn=0.5, k_tln=5.0, gamma_m=0.1, gamma_p=0.005, max_copy_mRNA=40, max_copy_protein=200) -> ReactionNetwork
exclusive_switch(...) -> ReactionNetwork
sir_epidemic(beta=0.005, gamma=0.1, max_S=50, max_I=50, max_R=50, S0=40, I0=5) -> ReactionNetwork
michaelis_menten_enzyme(k_f=0.01, k_r=0.1, k_cat=0.5, ...) -> ReactionNetwork
multi_species_cascade(n_species=5, k_activation=1.0, k_degradation=0.1, hill_coeff=2.0, k_half=10.0, max_copy=40) -> ReactionNetwork
mapk_cascade(n_layers=3, k_phos=0.1, k_dephos=0.05, max_copy=30) -> ReactionNetwork
gene_regulatory_network(n_genes=3, alpha=10.0, beta=1.0, hill_coeff=2.0, k_half=5.0, max_copy=25) -> ReactionNetwork
multi_gene_expression(n_genes=4, k_txn=0.5, k_tln=5.0, gamma_m=0.1, gamma_p=0.005, max_copy_mrna=15, max_copy_prot=30) -> ReactionNetwork
```

**Module:** `tn_check.models.biomodels` (extended library)

```python
p53_mdm2_oscillator(max_copy=25) -> ReactionNetwork
goodwin_oscillator(max_copy=20) -> ReactionNetwork
lac_operon(max_copy=15) -> ReactionNetwork
lotka_volterra(max_copy=30) -> ReactionNetwork
viral_infection(max_copy=20) -> ReactionNetwork
calcium_signaling(max_copy=20) -> ReactionNetwork
apoptosis_network(max_copy=15) -> ReactionNetwork
dual_phosphorylation(max_copy=20) -> ReactionNetwork
genetic_toggle_with_inducer(max_copy=20) -> ReactionNetwork
two_gene_oscillator(max_copy=25) -> ReactionNetwork
branching_pathway(max_copy=20) -> ReactionNetwork
cooperative_binding(max_copy=20) -> ReactionNetwork
```

**Module:** `tn_check.models.biomodels_extended` (broad coverage library, 26 models)

Categories: metabolic networks (glycolysis, pentose phosphate, TCA cycle, fatty acid oxidation), gene regulation (autoregulatory feedback, repressilator variant, feed-forward loop, CRISPR interference, epigenetic toggle), signaling pathways (MAPK cascade, NF-κB, JAK-STAT, Wnt/β-catenin, Notch-Delta), cell cycle (cell cycle checkpoint, growth factor, stem cell niche, circadian clock), population dynamics (predator-prey-scavenger, competition, mutualism, epidemic SEIR), stochastic phenomena (stochastic focusing, bistable switch, noise-induced oscillation, bursting gene expression).

```python
glycolysis_oscillator(max_copy=25) -> ReactionNetwork
purine_metabolism(max_copy=20) -> ReactionNetwork
pentose_phosphate(max_copy=20) -> ReactionNetwork
tca_cycle_minimal(max_copy=20) -> ReactionNetwork
autoregulatory_gene(max_copy=25) -> ReactionNetwork
feedforward_loop(max_copy=20) -> ReactionNetwork
mutual_inhibition(max_copy=20) -> ReactionNetwork
positive_feedback_gene(max_copy=20) -> ReactionNetwork
incoherent_ffl(max_copy=20) -> ReactionNetwork
egfr_signaling(max_copy=15) -> ReactionNetwork
nfkb_signaling(max_copy=20) -> ReactionNetwork
wnt_signaling(max_copy=20) -> ReactionNetwork
jak_stat_signaling(max_copy=20) -> ReactionNetwork
insulin_signaling(max_copy=20) -> ReactionNetwork
cell_cycle_checkpoint(max_copy=20) -> ReactionNetwork
growth_factor_response(max_copy=25) -> ReactionNetwork
quorum_sensing(max_copy=20) -> ReactionNetwork
stem_cell_differentiation(max_copy=20) -> ReactionNetwork
predator_prey_3sp(max_copy=20) -> ReactionNetwork
sir_vaccination(max_copy=20) -> ReactionNetwork
competition_model(max_copy=25) -> ReactionNetwork
mutualism_model(max_copy=25) -> ReactionNetwork
stochastic_switch(max_copy=30) -> ReactionNetwork
noise_induced_oscillation(max_copy=20) -> ReactionNetwork
excitable_system(max_copy=25) -> ReactionNetwork
stochastic_bursting(max_copy=25) -> ReactionNetwork
```

---

## Dense Reference Solver

**Module:** `tn_check.solver.dense_reference`

### `DenseReferenceSolver`

```python
DenseReferenceSolver(network: ReactionNetwork)
```

| Method | Description |
|--------|-------------|
| `solve(t, initial_counts=None)` | Solve CME exactly via matrix exponentiation |
| `steady_state()` | Compute steady-state via null space |
| `check_probability(state, formula, t)` | Evaluate CSL formula at state |

---

## Evaluation

**Module:** `tn_check.evaluation.benchmark`

```python
run_scaling_benchmark(species_counts=[2,3,4,5], max_copy=28, bond_dim=20) -> dict
run_accuracy_benchmark(models=None, bond_dims=[5,10,20,50]) -> dict
run_all_benchmarks() -> dict
```

**Module:** `tn_check.evaluation.prism_comparison`

```python
run_prism_style_comparison(model_name="birth_death") -> dict
compare_all_models() -> dict
```

**Module:** `tn_check.evaluation.compressibility`

### Dataclasses

- `EntanglementProfile`: `bipartition_index`, `von_neumann_entropy`, `renyi2_entropy`, `singular_values`, `bond_dim_for_eps`
- `CompressibilityTheorem`: `model_name`, `num_species`, `is_product_form`, `is_weakly_coupled`, `coupling_strength`, `entanglement_profile`, `max_entanglement`, `is_area_law`, `compressibility_verdict`, `sufficient_conditions_met`, `justification`

### Functions

```python
compute_entanglement_profile(state_vector, dims) -> list[EntanglementProfile]
compressibility_theorem(model_name, network, dims, state_vector=None) -> CompressibilityTheorem
run_compressibility_theorem_suite() -> dict
compute_deficiency(network) -> dict
compute_interaction_graph_bandwidth(network) -> dict
run_deficiency_analysis() -> list[dict]
```

#### `compute_entanglement_profile(state_vector, dims)`

Computes von Neumann and Rényi-2 entanglement entropies across all bipartitions of a multi-species state via SVD.

#### `compressibility_theorem(model_name, network, dims, state_vector=None)`

Applies sufficient conditions for bounded TT rank: deficiency-zero (strongest), product-form, weak coupling, area-law entanglement. Returns a `CompressibilityTheorem` with verdict and justification.

#### `run_compressibility_theorem_suite()`

Runs the compressibility theorem analysis across all 51 models in the library.

#### `compute_deficiency(network)`

Computes network deficiency δ = |C| - ℓ - s using Feinberg's CRNT theory. Extracts complexes, linkage classes, and stoichiometric subspace rank from the reaction graph.

**Returns:** `dict` with keys `deficiency`, `num_complexes`, `num_linkage_classes`, `stoich_rank`, `is_weakly_reversible`, `is_product_form_eligible`.

#### `compute_interaction_graph_bandwidth(network)`

Computes interaction graph structure for TT rank bounds. Analyzes which species interact through shared reactions.

**Returns:** `dict` with keys `bandwidth`, `interaction_edges`, `max_interaction_range`.

#### `run_deficiency_analysis()`

Runs deficiency analysis across all 51 models. Reports deficiency, weak reversibility, and product-form eligibility for each model.

**Module:** `tn_check.evaluation.prism_crossvalidation`

### `PrismCrossValidator`

| Method | Description |
|--------|-------------|
| `run_full_crossvalidation() -> dict` | Run verdict-by-verdict comparison on all 10 suite models |
| `run_single_model(model_name) -> dict` | Cross-validate a single model |

### Additional Functions

```python
export_storm_benchmark(output_dir="storm_benchmarks") -> dict
run_verdict_comparison_table() -> list[dict]
```

#### `export_storm_benchmark(output_dir)`

Generates PRISM-format `.pm` model files and `.csl` property files for all benchmark models, enabling independent external validation with PRISM or Storm.

#### `run_verdict_comparison_table()`

Runs cross-validation and produces structured verdict-by-verdict comparison output suitable for inclusion in papers or reports.

**Module:** `tn_check.integrator.convergence_analysis`

### `TDVPConvergenceAnalysis` (dataclass)

Key fields: `tdvp_errors`, `euler_errors`, `error_ratios`, `backward_error_constant`, `conservation_error`, `frobenius_residual`, `effective_support_size`, `l1_from_frobenius_bound`, `guarantees`.

### Functions

```python
analyze_tdvp_convergence(network, max_copy=20, bond_dim=15, t_final=1.0, dt_values=None) -> TDVPConvergenceAnalysis
```

---

## Experiments

**Module:** `tn_check.experiments`

```python
run_all_experiments() -> dict
run_birth_death_experiment(max_copy=30) -> dict
run_clamping_experiment(num_trials=10) -> dict
run_certificate_verification_experiment() -> dict
run_spectral_gap_experiment() -> dict
run_toggle_switch_csl_experiment() -> dict
run_nonneg_rounding_experiment(num_trials=10) -> dict
run_end_to_end_verification_experiment() -> dict
run_gene_expression_experiment() -> dict
run_scaling_experiment(species_counts=[2,3,4,5,6,8,10,15]) -> dict
run_csl_model_checking_experiment() -> dict
run_error_propagation_experiment() -> dict
run_full_pipeline_experiment() -> dict
run_scaling_validation_experiment() -> dict
run_multi_species_csl_experiment() -> dict
run_compression_accuracy_experiment() -> dict
```

---

## FSP-TT Coupling Analysis

**Module:** `tn_check.coupling`

Analyzes the interplay between FSP truncation dimension *M* and TT bond dimension *χ*, providing joint optimization and non-conservativity tracking.

### Data Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `CouplingResult` | `fsp_dims, tt_ranks, errors, non_conservativity, boundary_exit_rates` | Single coupling experiment result |
| `JointOptimizationResult` | `optimal_fsp_dims, optimal_tt_ranks, pareto_front, total_cost` | Pareto-optimal (M, χ) configurations |
| `NonConservativityTracker` | `times, mass_loss, exit_rates, cumulative_loss` | Time series of probability mass leakage |

### `FSPTTCouplingAnalysis`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(network, config=None)` | Initialize with a `ReactionNetwork` |
| `analyze_coupling` | `(fsp_dims, chi_values, time_horizon, ...) -> CouplingResult` | Sweep (M, χ) grid, measure errors and non-conservativity |
| `joint_m_chi_optimization` | `(target_error, time_horizon, ...) -> JointOptimizationResult` | Find Pareto-optimal allocations meeting error target |
| `track_nonconservativity` | `(fsp_dims, time_horizon, n_snapshots) -> NonConservativityTracker` | Record probability mass loss over time |

### Module-Level Functions

```python
fsp_nonconservativity_bound(exit_rate: float, time_horizon: float) -> float
compute_boundary_exit_rate(network, fsp_dims) -> float
coupling_correlation_study(network, fsp_dims_list, chi_list, time_horizon) -> dict
```

```python
from tn_check.coupling import FSPTTCouplingAnalysis, fsp_nonconservativity_bound

coupling = FSPTTCouplingAnalysis(network)
result = coupling.analyze_coupling(
    fsp_dims=[20, 40, 60],
    chi_values=[4, 8, 16],
    time_horizon=1.0
)
bound = fsp_nonconservativity_bound(exit_rate=0.1, time_horizon=1.0)
```

---

## Stiffness-Aware TDVP

**Module:** `tn_check.stiffness`

Extends the TDVP integrator with stiffness detection and adaptive step-size control via a PI controller with CFL-like stability constraints.

### Data Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `StiffnessAnalysis` | `stiffness_ratio, max_eigenvalue, min_eigenvalue, condition_number, is_stiff` | Spectral analysis of the CME generator |
| `ProjectionMonitor` | `projection_errors, norm_violations, step_sizes, accepted_steps, rejected_steps` | Tracks TDVP projection fidelity per step |

### `StepSizeController`

PI controller for adaptive time-stepping.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(dt_init, dt_min, dt_max, safety, k_p, k_i)` | Initialize with PI gains and bounds |
| `propose_step` | `(error_estimate, tolerance) -> float` | Compute next step size from local error |
| `reject_step` | `() -> float` | Halve current step on rejection |

### `StiffnessAwareTDVP`

Extends `IntegratorBase`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config=None, stiffness_threshold=1e4, ...)` | Initialize with stiffness parameters |
| `analyze_stiffness` | `(generator_mpo, dims) -> StiffnessAnalysis` | Compute stiffness ratio of CME generator |
| `evolve_adaptive` | `(state, generator, time_horizon, ...) -> tuple[MPS, ProjectionMonitor]` | Integrate with adaptive step control |
| `detect_fast_transients` | `(state, generator) -> dict` | Identify fast-decaying modes via power iteration |

### Module-Level Functions

```python
compute_stiffness_ratio(generator_mpo, dims) -> StiffnessAnalysis
analyze_generator_spectrum(generator_mpo, dims, k=10) -> dict
stiffness_comparison_experiment(models_dict, time_horizon) -> dict
```

```python
from tn_check.stiffness import StiffnessAwareTDVP, compute_stiffness_ratio

tdvp = StiffnessAwareTDVP(stiffness_threshold=1e4)
analysis = tdvp.analyze_stiffness(generator_mpo, dims)
state, monitor = tdvp.evolve_adaptive(
    state=initial_mps,
    generator=generator_mpo,
    time_horizon=1.0,
    tolerance=1e-6
)
```

---

## Statistical Model Checking

**Module:** `tn_check.smc`

Provides Gillespie SSA simulation and Sequential Probability Ratio Test (SPRT) for CSL model checking, enabling direct comparison with TN-Check's tensor-based approach.

### Data Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `SMCResult` | `verdict, p_estimate, confidence_interval, n_samples, elapsed_time, trajectory_stats` | Result of an SPRT or Monte Carlo CSL check |
| `SMCComparison` | `smc_result, tn_result, agreement, speedup, smc_time, tn_time` | Head-to-head SMC vs TN-Check comparison |

### Module-Level Functions

```python
gillespie_simulation(network, initial_state, t_final, max_steps=1_000_000) -> dict
importance_sampling_sprt(
    network, initial_state, property_fn, t_final,
    alpha=0.01, beta=0.01, theta=0.5, indifference=0.01, ...
) -> SMCResult
run_smc_comparison(network, config, property_fn, t_final, ...) -> SMCComparison
smc_scaling_study(models_dict, property_fns, t_final) -> dict
```

```python
from tn_check.smc import gillespie_simulation, run_smc_comparison

traj = gillespie_simulation(network, initial_state=[10], t_final=1.0)
comparison = run_smc_comparison(
    network, config,
    property_fn=lambda counts: counts[0] > 5,
    t_final=1.0
)
print(f"Agreement: {comparison.agreement}, Speedup: {comparison.speedup:.1f}x")
```

---

## Error Bound Tightness

**Module:** `tn_check.evaluation.tightness`

Studies how tight the certified error bounds (Theorem 7) are compared to actual L1 errors, computing tightness ratios across models and parameter regimes.

### Data Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `TightnessResult` | `model_name, certified_bound, actual_error, tightness_ratio, ...` | Per-model tightness measurement |
| `TightnessStudy` | `results, summary_stats, parameter_sensitivity` | Aggregated study across models |

### Module-Level Functions

```python
run_tightness_study(
    models: dict[str, tuple[ReactionNetwork, TNCheckConfig]],
    time_horizons: list[float],
    chi_values: list[int],
    fsp_dims_list: list[list[int]]
) -> TightnessStudy
```

```python
from tn_check.evaluation.tightness import run_tightness_study

study = run_tightness_study(
    models={"birth_death": (network, config)},
    time_horizons=[0.5, 1.0],
    chi_values=[4, 8],
    fsp_dims_list=[[30], [60]]
)
print(f"Mean tightness ratio: {study.summary_stats['mean_ratio']:.1f}×")
```

---

## Graph-Theoretic Compressibility

**Module:** `tn_check.evaluation.graph_predictors`

Predicts TT-rank compressibility from the CRN interaction graph using three complementary methods: tree-width, deficiency, and modularity.

### Module-Level Functions

```python
compute_treewidth_upper_bound(network) -> dict
    # Returns: {'treewidth': int, 'elimination_order': list, 'rank_bound': int, ...}

compute_crn_deficiency(network) -> dict
    # Returns: {'deficiency': int, 'n_complexes': int, 'rank': int, 'n_linkage_classes': int, ...}

compute_modularity(network) -> dict
    # Returns: {'modularity': float, 'communities': list, 'separator_dims': list, 'rank_bound': int}

predict_compressibility_graph(network) -> dict
    # Returns combined prediction with treewidth, deficiency, modularity sub-dicts,
    # plus 'combined_rank_bound' (minimum of all) and 'compressibility_rating'
```

```python
from tn_check.evaluation.graph_predictors import predict_compressibility_graph

prediction = predict_compressibility_graph(network)
print(f"Rating: {prediction['compressibility_rating']}")
print(f"Combined rank bound: {prediction['combined_rank_bound']}")
```

---

## Path B Experiments

**Module:** `tn_check.evaluation.path_b_experiments`

Orchestrates all five Category B improvement experiments and saves results to JSON.

### Module-Level Functions

```python
run_all_experiments(output_dir="benchmarks", quick=False) -> dict
    # Runs: fsp_tt_coupling, stiffness_analysis, error_tightness,
    #       compressibility_predictors, smc_comparison
    # Returns dict with per-experiment results and metadata
```

```python
from tn_check.evaluation.path_b_experiments import run_all_experiments

results = run_all_experiments(output_dir="benchmarks", quick=True)
for name, data in results["experiments"].items():
    print(f"{name}: {data.get('status', 'ok')}")
```

---

## CLI Reference

**Command:** `tn-check`

The CLI is installed as a console script via `pip install -e .` and provides
subcommands for working with reaction networks and CSL properties.

### `tn-check check`

```
tn-check check NETWORK_FILE PROPERTY [OPTIONS]
```

Check a CSL property against a reaction network.

| Argument / Option | Description |
|---|---|
| `NETWORK_FILE` | Path to YAML/JSON network definition |
| `PROPERTY` | CSL formula string (e.g., `"P>=0.9 [true U<=10.0 X_0 >= 3]"`) |
| `--max-bond-dim` / `-chi` | Maximum TT bond dimension (default: 200) |
| `--output` / `-o` | Write result JSON to file |
| `--verbose` / `-v` | Enable verbose logging |

### `tn-check verify`

```
tn-check verify NETWORK_FILE PROPERTY [OPTIONS]
```

Like `check`, but also produces a formal verification certificate via
the independent `CertificateVerifier`.

| Option | Description |
|---|---|
| `--certificate` / `-c` | Output path for certificate JSON (default: `certificate.json`) |
| `--max-bond-dim` / `-chi` | Maximum TT bond dimension (default: 200) |

### `tn-check info`

```
tn-check info NETWORK_FILE
```

Display network statistics: species count, reaction count, state space
size estimate, and per-species/per-reaction details.

### `tn-check init-network`

```
tn-check init-network [OUTPUT_FILE]
```

Generate a starter YAML template with commented examples of all three
kinetics types. Default output: `network.yaml`.

### `tn-check check-network`

```
tn-check check-network NETWORK_FILE
```

Validate a network definition file without running model checking. Checks
for structural errors: missing fields, unknown species, invalid parameters.

---

## Network Loader

**Module:** `tn_check.network_loader`

Reads YAML/JSON network definitions and constructs `ReactionNetwork` objects.

### `load_network(path) -> ReactionNetwork`

Load a `ReactionNetwork` from a YAML or JSON file. Supports mass-action,
Hill, and Michaelis-Menten kinetics.

### `validate_network_dict(data) -> list[str]`

Validate a parsed network dictionary without building the network.
Returns a list of error messages (empty if valid).

### Network File Format

```yaml
name: my_model            # optional
description: "..."        # optional

species:
  - name: A               # required
    max_copy_number: 50    # optional, default 50
    initial_count: 0       # optional, default 0
    description: "..."     # optional

reactions:
  # Mass-action kinetics
  - name: r1              # required
    reactants: {A: 1}     # species_name: stoichiometry
    products: {B: 1}
    rate_constant: 0.5    # required for mass_action
    kinetics: mass_action  # optional, default mass_action

  # Hill kinetics
  - name: r2
    products: {A: 1}
    kinetics: hill
    hill:
      regulator: B          # species name
      v_max: 10.0
      k_half: 5.0
      hill_coefficient: 2.0
      activation: true      # true = activation, false = repression

  # Michaelis-Menten kinetics
  - name: r3
    kinetics: michaelis_menten
    michaelis_menten:
      substrate: S
      product: P
      enzyme: E
      k_cat: 1.0
      k_m: 10.0
```
