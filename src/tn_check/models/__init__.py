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
]
