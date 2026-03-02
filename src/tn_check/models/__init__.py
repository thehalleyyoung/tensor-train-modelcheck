"""
Standard stochastic reaction network models for benchmarking.

Provides parameterized model builders for:
- Toggle switch (bistable, 2+ species)
- Repressilator (oscillatory, 3+ species)
- MAPK cascade (modular, multi-species)
- Birth-death (simple, 1 species)
- Schlögl model (bistable, 1 species)
- p53-MDM2, Goodwin oscillator, Lac operon, Lotka-Volterra (biomodels)
- Viral infection, Calcium signaling, Apoptosis, Dual phosphorylation
"""

from tn_check.models.library import (
    birth_death,
    toggle_switch,
    repressilator,
    cascade,
    schlogl,
    gene_expression,
    exclusive_switch,
    sir_epidemic,
    michaelis_menten_enzyme,
    multi_species_cascade,
    mapk_cascade,
    gene_regulatory_network,
    multi_gene_expression,
)

from tn_check.models.biomodels import (
    p53_mdm2,
    goodwin_oscillator,
    lac_operon,
    lotka_volterra,
    viral_infection,
    calcium_signaling,
    apoptosis_decision,
    dual_phosphorylation,
)

from tn_check.models.biomodels_extended import (
    glycolysis_oscillator,
    purine_metabolism,
    pentose_phosphate,
    tca_cycle_minimal,
    autoregulatory_gene,
    feedforward_loop,
    mutual_inhibition,
    positive_feedback_gene,
    incoherent_ffl,
    egfr_signaling,
    nfkb_signaling,
    wnt_signaling,
    jak_stat_signaling,
    insulin_signaling,
    cell_cycle_checkpoint,
    growth_factor_response,
    quorum_sensing,
    stem_cell_differentiation,
    predator_prey_3sp,
    sir_vaccination,
    competition_model,
    mutualism_model,
    stochastic_switch,
    noise_induced_oscillation,
    excitable_system,
    stochastic_bursting,
)

__all__ = [
    "birth_death",
    "toggle_switch",
    "repressilator",
    "cascade",
    "schlogl",
    "gene_expression",
    "exclusive_switch",
    "sir_epidemic",
    "michaelis_menten_enzyme",
    "multi_species_cascade",
    "mapk_cascade",
    "gene_regulatory_network",
    "multi_gene_expression",
    "p53_mdm2",
    "goodwin_oscillator",
    "lac_operon",
    "lotka_volterra",
    "viral_infection",
    "calcium_signaling",
    "apoptosis_decision",
    "dual_phosphorylation",
    "glycolysis_oscillator",
    "purine_metabolism",
    "pentose_phosphate",
    "tca_cycle_minimal",
    "autoregulatory_gene",
    "feedforward_loop",
    "mutual_inhibition",
    "positive_feedback_gene",
    "incoherent_ffl",
    "egfr_signaling",
    "nfkb_signaling",
    "wnt_signaling",
    "jak_stat_signaling",
    "insulin_signaling",
    "cell_cycle_checkpoint",
    "growth_factor_response",
    "quorum_sensing",
    "stem_cell_differentiation",
    "predator_prey_3sp",
    "sir_vaccination",
    "competition_model",
    "mutualism_model",
    "stochastic_switch",
    "noise_induced_oscillation",
    "excitable_system",
    "stochastic_bursting",
]
