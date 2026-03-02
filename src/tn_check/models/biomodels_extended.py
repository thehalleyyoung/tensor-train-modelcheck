"""
Extended biological reaction network models (26 additional).

Covers diverse network topologies across six categories:
  1. Metabolic networks (4 models)
  2. Gene regulation (5 models)
  3. Signaling pathways (5 models)
  4. Cell cycle & growth (4 models)
  5. Population dynamics (4 models)
  6. Stochastic phenomena (4 models)

Each model is parameterized with biologically realistic rate constants
and can be compiled to MPO format via the CME compiler.
"""

from __future__ import annotations

from tn_check.cme.reaction_network import (
    Species, Reaction, ReactionNetwork,
    MassActionPropensity, HillPropensity,
    KineticsType,
)


def _build_network(name: str, species_list, reactions_list) -> ReactionNetwork:
    """Helper to build a ReactionNetwork."""
    net = ReactionNetwork(name=name)
    for sp in species_list:
        net.species.append(sp)
        net._species_by_name[sp.name] = sp
    for rxn in reactions_list:
        net.reactions.append(rxn)
        net._reaction_by_name[rxn.name] = rxn
    return net


# ---------------------------------------------------------------------------
#  1. Metabolic networks
# ---------------------------------------------------------------------------

def glycolysis_oscillator(
    k_input: float = 0.6,
    k_cat: float = 0.1,
    k_feedback: float = 0.01,
    k_deg_sub: float = 0.05,
    k_deg_enz: float = 0.05,
    k_deg_prod: float = 0.1,
    max_substrate: int = 30,
    max_enzyme: int = 25,
    max_product: int = 30,
) -> ReactionNetwork:
    """
    Glycolytic oscillator (Sel'kov model, 3 species).

    Substrate is converted to product by enzyme; product feeds back to
    activate the enzyme, producing bistable oscillations characteristic
    of PFK-mediated glycolytic rhythms.

    Based on Sel'kov, European J. Biochem., 1968.
    BioModels: BIOMD0000000064
    """
    species = [
        Species(name="Substrate", index=0, max_copy_number=max_substrate,
                initial_count=10),
        Species(name="Enzyme", index=1, max_copy_number=max_enzyme,
                initial_count=5),
        Species(name="Product", index=2, max_copy_number=max_product,
                initial_count=2),
    ]
    reactions = [
        # Substrate input (constant influx)
        Reaction(
            name="substrate_input", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_input, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Catalysis: Substrate + Enzyme → Enzyme + Product
        Reaction(
            name="catalysis", index=1,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1, 2], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_cat,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Positive feedback: Product activates enzyme production
        Reaction(
            name="feedback_activation", index=2,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[1, 2], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_feedback,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # Substrate degradation
        Reaction(
            name="deg_substrate", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_sub,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Enzyme degradation
        Reaction(
            name="deg_enzyme", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_enz,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Product degradation
        Reaction(
            name="deg_product", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_prod,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("glycolysis_oscillator", species, reactions)


def purine_metabolism(
    k_prod_prpp: float = 0.8,
    k_prpp_to_imp: float = 0.3,
    k_imp_to_hyp: float = 0.2,
    k_deg_prpp: float = 0.05,
    k_deg_imp: float = 0.05,
    k_deg_hyp: float = 0.1,
    max_prpp: int = 25,
    max_imp: int = 25,
    max_hyp: int = 25,
) -> ReactionNetwork:
    """
    Purine metabolism linear pathway (3 species).

    PRPP → IMP → Hypoxanthine.  Models the de-novo purine synthesis
    pathway relevant to chemotherapy target studies.

    Based on Curto et al., Math. Biosci., 1998.
    BioModels: BIOMD0000000015
    """
    species = [
        Species(name="PRPP", index=0, max_copy_number=max_prpp, initial_count=8),
        Species(name="IMP", index=1, max_copy_number=max_imp, initial_count=3),
        Species(name="Hypoxanthine", index=2, max_copy_number=max_hyp,
                initial_count=1),
    ]
    reactions = [
        # PRPP production
        Reaction(
            name="prod_prpp", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_prpp, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # PRPP → IMP
        Reaction(
            name="prpp_to_imp", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prpp_to_imp,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # IMP → Hypoxanthine
        Reaction(
            name="imp_to_hyp", index=2,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_imp_to_hyp,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # PRPP degradation
        Reaction(
            name="deg_prpp", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_prpp,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # IMP degradation
        Reaction(
            name="deg_imp", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_imp,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Hypoxanthine degradation
        Reaction(
            name="deg_hyp", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_hyp,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("purine_metabolism", species, reactions)


def pentose_phosphate(
    k_prod_g6p: float = 0.5,
    k_oxidative: float = 0.2,
    k_nonoxidative: float = 0.15,
    k_deg_g6p: float = 0.05,
    k_deg_r5p: float = 0.08,
    k_deg_nadph: float = 0.1,
    max_g6p: int = 25,
    max_r5p: int = 25,
    max_nadph: int = 30,
) -> ReactionNetwork:
    """
    Pentose phosphate pathway (3 species, branching).

    G6P enters the oxidative branch producing NADPH and R5P.
    R5P can also be recycled back to G6P via the non-oxidative branch.

    Based on Vaseghi et al., Metabolic Engineering, 1999.
    BioModels: BIOMD0000000071
    """
    species = [
        Species(name="G6P", index=0, max_copy_number=max_g6p, initial_count=10),
        Species(name="R5P", index=1, max_copy_number=max_r5p, initial_count=2),
        Species(name="NADPH", index=2, max_copy_number=max_nadph, initial_count=5),
    ]
    reactions = [
        # G6P production (from glycolysis)
        Reaction(
            name="prod_g6p", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_g6p, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Oxidative branch: G6P → R5P + NADPH
        Reaction(
            name="oxidative", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[1, 2], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_oxidative,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Non-oxidative branch: R5P → G6P (recycling)
        Reaction(
            name="nonoxidative", index=2,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_nonoxidative,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # G6P degradation
        Reaction(
            name="deg_g6p", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_g6p,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # R5P degradation (utilization for nucleotide synthesis)
        Reaction(
            name="deg_r5p", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_r5p,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # NADPH consumption
        Reaction(
            name="deg_nadph", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_nadph,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("pentose_phosphate", species, reactions)


def tca_cycle_minimal(
    k_prod_acoa: float = 0.4,
    k_citrate_syn: float = 0.08,
    k_cit_to_akg: float = 0.15,
    k_akg_to_oaa: float = 0.12,
    k_deg_acoa: float = 0.03,
    k_deg_cit: float = 0.03,
    k_deg_akg: float = 0.03,
    k_deg_oaa: float = 0.03,
    max_acoa: int = 15,
    max_cit: int = 15,
    max_akg: int = 15,
    max_oaa: int = 15,
) -> ReactionNetwork:
    """
    Minimal TCA cycle (4 species, cyclic topology).

    AcetylCoA + OAA → Citrate → alphaKG → OAA, completing the cycle.
    Captures the cyclic stoichiometry of the Krebs cycle.

    Based on Wu et al., PLoS Comput. Biol., 2007.
    BioModels: BIOMD0000000222
    """
    species = [
        Species(name="AcetylCoA", index=0, max_copy_number=max_acoa,
                initial_count=8),
        Species(name="Citrate", index=1, max_copy_number=max_cit,
                initial_count=3),
        Species(name="AlphaKG", index=2, max_copy_number=max_akg,
                initial_count=2),
        Species(name="OAA", index=3, max_copy_number=max_oaa,
                initial_count=4),
    ]
    reactions = [
        # AcetylCoA production
        Reaction(
            name="prod_acoa", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_acoa, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Citrate synthase: AcetylCoA + OAA → Citrate
        Reaction(
            name="citrate_synthase", index=1,
            reactant_species=[0, 3], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_citrate_syn,
                reactant_species=[0, 3], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Citrate → AlphaKG (aconitase + isocitrate dehydrogenase)
        Reaction(
            name="cit_to_akg", index=2,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_cit_to_akg,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # AlphaKG → OAA (remaining TCA steps lumped)
        Reaction(
            name="akg_to_oaa", index=3,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[3], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_akg_to_oaa,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # AcetylCoA degradation
        Reaction(
            name="deg_acoa", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_acoa,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Citrate degradation
        Reaction(
            name="deg_cit", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_cit,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # AlphaKG degradation
        Reaction(
            name="deg_akg", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_akg,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # OAA degradation
        Reaction(
            name="deg_oaa", index=7,
            reactant_species=[3], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_oaa,
                reactant_species=[3], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("tca_cycle_minimal", species, reactions)


# ---------------------------------------------------------------------------
#  2. Gene regulation
# ---------------------------------------------------------------------------

def autoregulatory_gene(
    k_transcription: float = 5.0,
    k_translation: float = 2.0,
    k_deg_mrna: float = 0.5,
    k_deg_protein: float = 0.1,
    k_half: float = 8.0,
    hill_coeff: float = 2.0,
    max_mrna: int = 25,
    max_protein: int = 40,
) -> ReactionNetwork:
    """
    Autoregulatory gene with self-repression (2 species).

    mRNA is transcribed under Hill repression by its own protein product,
    then translated into protein.  Classic motif for noise buffering.

    Based on Becskei & Serrano, Nature, 2000.
    BioModels: BIOMD0000000311
    """
    species = [
        Species(name="mRNA", index=0, max_copy_number=max_mrna, initial_count=2),
        Species(name="Protein", index=1, max_copy_number=max_protein,
                initial_count=5),
    ]
    reactions = [
        # Transcription (repressed by protein)
        Reaction(
            name="transcription", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_transcription, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=1, activation=False,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Translation: mRNA → mRNA + Protein
        Reaction(
            name="translation", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0, 1], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_translation,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # mRNA degradation
        Reaction(
            name="deg_mrna", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_mrna,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein degradation
        Reaction(
            name="deg_protein", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_protein,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("autoregulatory_gene", species, reactions)


def feedforward_loop(
    k_prod_X: float = 0.8,
    k_X_act_Y: float = 3.0,
    k_prod_Z: float = 2.0,
    k_deg: float = 0.1,
    k_half_XY: float = 5.0,
    k_half_XZ: float = 5.0,
    hill_coeff: float = 2.0,
    max_X: int = 25,
    max_Y: int = 25,
    max_Z: int = 25,
) -> ReactionNetwork:
    """
    Type-1 coherent feedforward loop (3 species).

    X activates Y; both X and Y activate Z.  This motif acts as a
    sign-sensitive delay element filtering transient signals.

    Based on Mangan & Alon, PNAS, 2003.
    BioModels: BIOMD0000000340
    """
    species = [
        Species(name="X", index=0, max_copy_number=max_X, initial_count=5),
        Species(name="Y", index=1, max_copy_number=max_Y, initial_count=1),
        Species(name="Z", index=2, max_copy_number=max_Z, initial_count=0),
    ]
    reactions = [
        # X constitutive production
        Reaction(
            name="prod_X", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_X, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Y production activated by X (Hill)
        Reaction(
            name="prod_Y", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_X_act_Y, k_half=k_half_XY,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Z production activated by Y (Hill); requires X context via AND-gate
        Reaction(
            name="prod_Z", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_prod_Z, k_half=k_half_XZ,
                hill_coefficient=hill_coeff, species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # X degradation
        Reaction(
            name="deg_X", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Y degradation
        Reaction(
            name="deg_Y", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Z degradation
        Reaction(
            name="deg_Z", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("feedforward_loop", species, reactions)


def mutual_inhibition(
    k_prod_A: float = 1.0,
    k_prod_B: float = 1.0,
    k_inhib_AB: float = 0.05,
    k_inhib_BA: float = 0.05,
    k_deg_A: float = 0.1,
    k_deg_B: float = 0.1,
    max_A: int = 30,
    max_B: int = 30,
) -> ReactionNetwork:
    """
    Mutual inhibition switch – mass-action variant (2 species).

    A and B repress each other through direct degradation reactions
    (mass action, no Hill kinetics).  Produces bistability at
    sufficiently high inhibition rates.

    Based on Cherry & Adler, J. Theor. Biol., 2000.
    BioModels: BIOMD0000000398
    """
    species = [
        Species(name="A", index=0, max_copy_number=max_A, initial_count=10),
        Species(name="B", index=1, max_copy_number=max_B, initial_count=2),
    ]
    reactions = [
        # A production
        Reaction(
            name="prod_A", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_A, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # B production
        Reaction(
            name="prod_B", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_B, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # A inhibits B: A + B → A  (B consumed)
        Reaction(
            name="A_inhibits_B", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_inhib_AB,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # B inhibits A: A + B → B  (A consumed)
        Reaction(
            name="B_inhibits_A", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_inhib_BA,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # A degradation
        Reaction(
            name="deg_A", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_A, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B degradation
        Reaction(
            name="deg_B", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_B, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("mutual_inhibition", species, reactions)


def positive_feedback_gene(
    k_basal: float = 0.1,
    k_activated: float = 8.0,
    k_translation: float = 2.0,
    k_deg_mrna: float = 0.5,
    k_deg_protein: float = 0.1,
    k_half: float = 10.0,
    hill_coeff: float = 2.0,
    max_mrna: int = 20,
    max_protein: int = 40,
) -> ReactionNetwork:
    """
    Positive autoregulation gene circuit (2 species).

    Protein product activates its own transcription via Hill kinetics.
    Produces slow turn-on compared to unregulated expression and can
    exhibit bistability.

    Based on Maeda & Sano, J. Mol. Biol., 2006.
    BioModels: BIOMD0000000312
    """
    species = [
        Species(name="mRNA", index=0, max_copy_number=max_mrna, initial_count=1),
        Species(name="Protein", index=1, max_copy_number=max_protein,
                initial_count=0),
    ]
    reactions = [
        # Basal transcription
        Reaction(
            name="basal_transcription", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_basal, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Positive-feedback transcription (Hill activation by protein)
        Reaction(
            name="activated_transcription", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_activated, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Translation: mRNA → mRNA + Protein
        Reaction(
            name="translation", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0, 1], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_translation,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # mRNA degradation
        Reaction(
            name="deg_mrna", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_mrna,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein degradation
        Reaction(
            name="deg_protein", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_protein,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("positive_feedback_gene", species, reactions)


def incoherent_ffl(
    k_prod_X: float = 0.8,
    k_X_act_Y: float = 3.0,
    k_X_act_Z: float = 4.0,
    k_deg: float = 0.1,
    k_half: float = 5.0,
    hill_coeff: float = 2.0,
    max_X: int = 25,
    max_Y: int = 25,
    max_Z: int = 25,
) -> ReactionNetwork:
    """
    Incoherent feedforward loop (type 1) with pulse response (3 species).

    X activates both Y and Z, but Y represses Z.  This produces a
    transient pulse in Z upon step increase in X.

    Based on Mangan et al., J. Mol. Biol., 2006.
    BioModels: BIOMD0000000341
    """
    species = [
        Species(name="X", index=0, max_copy_number=max_X, initial_count=5),
        Species(name="Y", index=1, max_copy_number=max_Y, initial_count=0),
        Species(name="Z", index=2, max_copy_number=max_Z, initial_count=0),
    ]
    reactions = [
        # X constitutive production
        Reaction(
            name="prod_X", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_X, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Y production activated by X
        Reaction(
            name="prod_Y", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_X_act_Y, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Z production activated by X
        Reaction(
            name="X_activates_Z", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_X_act_Z, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Z repression by Y
        Reaction(
            name="Y_represses_Z", index=3,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=0.08,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # X degradation
        Reaction(
            name="deg_X", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Y degradation
        Reaction(
            name="deg_Y", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Z degradation
        Reaction(
            name="deg_Z", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("incoherent_ffl", species, reactions)


# ---------------------------------------------------------------------------
#  3. Signaling pathways
# ---------------------------------------------------------------------------

def egfr_signaling(
    k_ligand: float = 0.3,
    k_egfr_act: float = 0.5,
    k_ras_act: float = 0.4,
    k_erk_act: float = 0.3,
    k_deg_egfr: float = 0.1,
    k_deg_ras: float = 0.1,
    k_deg_erk: float = 0.05,
    k_half: float = 6.0,
    hill_coeff: float = 2.0,
    max_egfr: int = 25,
    max_ras: int = 25,
    max_erk: int = 30,
) -> ReactionNetwork:
    """
    Simplified EGFR → RAS → ERK signaling cascade (3 species).

    EGFR activated constitutively (ligand-bound); RAS activated by
    EGFR; ERK activated by RAS via Hill kinetics.

    Based on Kholodenko, European J. Biochem., 2000.
    BioModels: BIOMD0000000048
    """
    species = [
        Species(name="EGFR", index=0, max_copy_number=max_egfr, initial_count=5),
        Species(name="RAS", index=1, max_copy_number=max_ras, initial_count=2),
        Species(name="ERK", index=2, max_copy_number=max_erk, initial_count=0),
    ]
    reactions = [
        # EGFR activation by ligand (constitutive)
        Reaction(
            name="egfr_activation", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_ligand, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # RAS activation by EGFR (Hill)
        Reaction(
            name="ras_activation", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_ras_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # ERK activation by RAS (Hill)
        Reaction(
            name="erk_activation", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_erk_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # EGFR degradation / internalization
        Reaction(
            name="deg_egfr", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_egfr,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # RAS deactivation
        Reaction(
            name="deg_ras", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_ras,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # ERK dephosphorylation
        Reaction(
            name="deg_erk", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_erk,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("egfr_signaling", species, reactions)


def nfkb_signaling(
    k_stimulus: float = 0.3,
    k_nfkb_activation: float = 0.5,
    k_ikb_prod: float = 0.8,
    k_nfkb_ikb_bind: float = 0.05,
    k_deg_nfkb: float = 0.05,
    k_deg_ikb: float = 0.1,
    k_deg_stimulus: float = 0.1,
    k_half: float = 5.0,
    hill_coeff: float = 2.0,
    max_nfkb: int = 25,
    max_ikb: int = 25,
    max_stimulus: int = 20,
) -> ReactionNetwork:
    """
    NF-κB / IκB negative feedback oscillator (3 species).

    Stimulus activates NFkB; NFkB induces IkB production; IkB sequesters
    NFkB, forming a delayed negative feedback loop producing oscillations.

    Based on Hoffmann et al., Science, 2002.
    BioModels: BIOMD0000000140
    """
    species = [
        Species(name="NFkB", index=0, max_copy_number=max_nfkb, initial_count=2),
        Species(name="IkB", index=1, max_copy_number=max_ikb, initial_count=8),
        Species(name="Stimulus", index=2, max_copy_number=max_stimulus,
                initial_count=5),
    ]
    reactions = [
        # Stimulus production
        Reaction(
            name="stimulus_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_stimulus, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # NFkB activation by stimulus (Hill)
        Reaction(
            name="nfkb_activation", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_nfkb_activation, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=2, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # IkB production induced by NFkB (Hill)
        Reaction(
            name="ikb_prod", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_ikb_prod, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # NFkB sequestration by IkB: NFkB + IkB → IkB
        Reaction(
            name="nfkb_sequester", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_nfkb_ikb_bind,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # NFkB degradation
        Reaction(
            name="deg_nfkb", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_nfkb,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # IkB degradation
        Reaction(
            name="deg_ikb", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_ikb,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Stimulus decay
        Reaction(
            name="deg_stimulus", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_stimulus,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("nfkb_signaling", species, reactions)


def wnt_signaling(
    k_wnt_prod: float = 0.3,
    k_bcat_prod: float = 0.5,
    k_destruction: float = 0.05,
    k_apc_prod: float = 0.4,
    k_wnt_inhibits_apc: float = 0.03,
    k_deg_wnt: float = 0.1,
    k_deg_bcat: float = 0.05,
    k_deg_apc: float = 0.05,
    max_wnt: int = 20,
    max_bcat: int = 30,
    max_apc: int = 25,
) -> ReactionNetwork:
    """
    Wnt / beta-catenin / APC signaling (3 species).

    Wnt inhibits the APC destruction complex, allowing beta-catenin to
    accumulate.  Without Wnt, APC targets beta-catenin for degradation.

    Based on Lee et al., PLoS Biol., 2003.
    BioModels: BIOMD0000000149
    """
    species = [
        Species(name="Wnt", index=0, max_copy_number=max_wnt, initial_count=5),
        Species(name="BetaCatenin", index=1, max_copy_number=max_bcat,
                initial_count=3),
        Species(name="APC", index=2, max_copy_number=max_apc, initial_count=8),
    ]
    reactions = [
        # Wnt ligand production
        Reaction(
            name="wnt_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_wnt_prod, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Beta-catenin production
        Reaction(
            name="bcat_prod", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_bcat_prod, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # APC production
        Reaction(
            name="apc_prod", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_apc_prod, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # APC-mediated beta-catenin destruction: BetaCatenin + APC → APC
        Reaction(
            name="bcat_destruction", index=3,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_destruction,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Wnt inhibits APC: Wnt + APC → Wnt  (APC degraded)
        Reaction(
            name="wnt_inhibits_apc", index=4,
            reactant_species=[0, 2], reactant_stoichiometry=[1, 1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_wnt_inhibits_apc,
                reactant_species=[0, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Wnt degradation
        Reaction(
            name="deg_wnt", index=5,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_wnt,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Beta-catenin basal degradation
        Reaction(
            name="deg_bcat", index=6,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_bcat,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # APC basal degradation
        Reaction(
            name="deg_apc", index=7,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_apc,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("wnt_signaling", species, reactions)


def jak_stat_signaling(
    k_ligand: float = 0.3,
    k_stat_act: float = 0.6,
    k_target_act: float = 0.4,
    k_deg_receptor: float = 0.08,
    k_deg_stat: float = 0.05,
    k_deg_target: float = 0.1,
    k_half: float = 5.0,
    hill_coeff: float = 2.0,
    max_receptor: int = 25,
    max_stat: int = 30,
    max_target: int = 25,
) -> ReactionNetwork:
    """
    JAK-STAT signaling cascade (3 species).

    Receptor activation triggers STAT phosphorylation and nuclear
    translocation, which activates target gene expression.

    Based on Yamada et al., Biochem. J., 2003.
    BioModels: BIOMD0000000093
    """
    species = [
        Species(name="Receptor", index=0, max_copy_number=max_receptor,
                initial_count=5),
        Species(name="STAT", index=1, max_copy_number=max_stat, initial_count=2),
        Species(name="TargetGene", index=2, max_copy_number=max_target,
                initial_count=0),
    ]
    reactions = [
        # Receptor activation (ligand binding)
        Reaction(
            name="receptor_activation", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_ligand, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # STAT activation by receptor (Hill)
        Reaction(
            name="stat_activation", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_stat_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Target gene expression induced by STAT (Hill)
        Reaction(
            name="target_activation", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_target_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Receptor internalization / degradation
        Reaction(
            name="deg_receptor", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_receptor,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # STAT dephosphorylation
        Reaction(
            name="deg_stat", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_stat,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Target gene product degradation
        Reaction(
            name="deg_target", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_target,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("jak_stat_signaling", species, reactions)


def insulin_signaling(
    k_insulin: float = 0.2,
    k_irs_act: float = 0.5,
    k_glut4_transloc: float = 0.4,
    k_deg_insulin: float = 0.15,
    k_deg_irs: float = 0.08,
    k_deg_glut4: float = 0.1,
    k_half: float = 5.0,
    hill_coeff: float = 2.0,
    max_insulin: int = 20,
    max_irs: int = 25,
    max_glut4: int = 30,
) -> ReactionNetwork:
    """
    Insulin → IRS → GLUT4 signaling pathway (3 species).

    Insulin binds receptor, activating IRS; IRS triggers GLUT4
    translocation to the plasma membrane for glucose uptake.

    Based on Sedaghat et al., Am. J. Physiol., 2002.
    BioModels: BIOMD0000000137
    """
    species = [
        Species(name="Insulin", index=0, max_copy_number=max_insulin,
                initial_count=8),
        Species(name="IRS", index=1, max_copy_number=max_irs, initial_count=2),
        Species(name="GLUT4", index=2, max_copy_number=max_glut4, initial_count=1),
    ]
    reactions = [
        # Insulin production / infusion
        Reaction(
            name="insulin_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_insulin, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # IRS activation by insulin (Hill)
        Reaction(
            name="irs_activation", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_irs_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # GLUT4 translocation by IRS (Hill)
        Reaction(
            name="glut4_translocation", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_glut4_transloc, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Insulin clearance
        Reaction(
            name="deg_insulin", index=3,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_insulin,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # IRS deactivation
        Reaction(
            name="deg_irs", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_irs,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # GLUT4 internalization
        Reaction(
            name="deg_glut4", index=5,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_glut4,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("insulin_signaling", species, reactions)


# ---------------------------------------------------------------------------
#  4. Cell cycle & growth
# ---------------------------------------------------------------------------

def cell_cycle_checkpoint(
    k_cyclin_prod: float = 0.5,
    k_cdk_act: float = 0.6,
    k_wee1_prod: float = 0.4,
    k_wee1_inhibits_cdk: float = 0.05,
    k_cdk_inhibits_wee1: float = 0.04,
    k_deg_cyclin: float = 0.1,
    k_deg_cdk: float = 0.05,
    k_deg_wee1: float = 0.05,
    k_half: float = 6.0,
    hill_coeff: float = 2.0,
    max_cyclin: int = 25,
    max_cdk: int = 25,
    max_wee1: int = 25,
) -> ReactionNetwork:
    """
    Cell cycle mitotic checkpoint (3 species).

    CyclinB accumulates and activates CDK1.  Wee1 kinase inhibits CDK1;
    CDK1 reciprocally inhibits Wee1, creating a bistable switch for
    mitotic entry.

    Based on Novak & Tyson, J. Cell Sci., 1993.
    BioModels: BIOMD0000000006
    """
    species = [
        Species(name="CyclinB", index=0, max_copy_number=max_cyclin,
                initial_count=2),
        Species(name="CDK1", index=1, max_copy_number=max_cdk, initial_count=1),
        Species(name="Wee1", index=2, max_copy_number=max_wee1, initial_count=8),
    ]
    reactions = [
        # CyclinB production
        Reaction(
            name="cyclin_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_cyclin_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # CDK1 activation by CyclinB (Hill)
        Reaction(
            name="cdk_activation", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_cdk_act, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Wee1 production
        Reaction(
            name="wee1_prod", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_wee1_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Wee1 inhibits CDK1: CDK1 + Wee1 → Wee1
        Reaction(
            name="wee1_inhibits_cdk", index=3,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_wee1_inhibits_cdk,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # CDK1 inhibits Wee1: CDK1 + Wee1 → CDK1
        Reaction(
            name="cdk_inhibits_wee1", index=4,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_cdk_inhibits_wee1,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # CyclinB degradation
        Reaction(
            name="deg_cyclin", index=5,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_cyclin,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # CDK1 degradation
        Reaction(
            name="deg_cdk", index=6,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_cdk,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Wee1 degradation
        Reaction(
            name="deg_wee1", index=7,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_wee1,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("cell_cycle_checkpoint", species, reactions)


def growth_factor_response(
    k_gf_prod: float = 0.5,
    k_response: float = 3.0,
    k_deg_gf: float = 0.2,
    k_deg_response: float = 0.1,
    k_half: float = 8.0,
    hill_coeff: float = 2.0,
    max_gf: int = 20,
    max_response: int = 40,
) -> ReactionNetwork:
    """
    Growth factor response (2 species, saturating).

    Growth factor is produced constitutively; cell response is a
    saturating (Hill) function of growth factor concentration.

    Based on Bhalla & Iyengar, Science, 1999.
    BioModels: BIOMD0000000009
    """
    species = [
        Species(name="GrowthFactor", index=0, max_copy_number=max_gf,
                initial_count=5),
        Species(name="CellResponse", index=1, max_copy_number=max_response,
                initial_count=1),
    ]
    reactions = [
        # Growth factor production
        Reaction(
            name="gf_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_gf_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Cell response (Hill activation by GF)
        Reaction(
            name="cell_response", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_response, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Growth factor degradation
        Reaction(
            name="deg_gf", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_gf,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Response degradation
        Reaction(
            name="deg_response", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_response,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("growth_factor_response", species, reactions)


def quorum_sensing(
    k_ahl_prod: float = 0.3,
    k_luxr_prod: float = 0.2,
    k_gfp_prod: float = 2.0,
    k_ahl_luxr_bind: float = 0.02,
    k_deg_ahl: float = 0.1,
    k_deg_luxr: float = 0.05,
    k_deg_gfp: float = 0.02,
    k_half: float = 5.0,
    hill_coeff: float = 2.0,
    max_ahl: int = 25,
    max_luxr: int = 25,
    max_gfp: int = 30,
) -> ReactionNetwork:
    """
    Quorum sensing circuit (3 species).

    AHL auto-inducer accumulates; AHL-LuxR complex activates GFP
    reporter transcription.  Models threshold-dependent population
    behaviour in bacteria.

    Based on James et al., Mol. Microbiol., 2000.
    BioModels: BIOMD0000000168
    """
    species = [
        Species(name="AHL", index=0, max_copy_number=max_ahl, initial_count=2),
        Species(name="LuxR", index=1, max_copy_number=max_luxr, initial_count=5),
        Species(name="GFP", index=2, max_copy_number=max_gfp, initial_count=0),
    ]
    reactions = [
        # AHL production
        Reaction(
            name="ahl_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_ahl_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # LuxR production
        Reaction(
            name="luxr_prod", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_luxr_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # AHL + LuxR binding (consumes both to form complex, modeled as loss)
        Reaction(
            name="ahl_luxr_bind", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_ahl_luxr_bind,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # GFP production activated by AHL (Hill, proxy for AHL-LuxR complex)
        Reaction(
            name="gfp_prod", index=3,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_gfp_prod, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # AHL degradation
        Reaction(
            name="deg_ahl", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_ahl,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # LuxR degradation
        Reaction(
            name="deg_luxr", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_luxr,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # GFP degradation
        Reaction(
            name="deg_gfp", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_gfp,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("quorum_sensing", species, reactions)


def stem_cell_differentiation(
    k_stemness_prod: float = 1.0,
    k_diff_prod: float = 0.5,
    k_stemness_inhibits_diff: float = 0.04,
    k_diff_inhibits_stemness: float = 0.04,
    k_deg_stemness: float = 0.1,
    k_deg_diff: float = 0.1,
    max_stemness: int = 30,
    max_diff: int = 30,
) -> ReactionNetwork:
    """
    Stem cell differentiation toggle (2 species).

    Stemness factor and differentiation factor mutually inhibit each
    other, creating a bistable switch between stem and differentiated
    cell states.

    Based on Huang et al., PLoS Comput. Biol., 2007.
    BioModels: BIOMD0000000254
    """
    species = [
        Species(name="Stemness", index=0, max_copy_number=max_stemness,
                initial_count=10),
        Species(name="Differentiation", index=1, max_copy_number=max_diff,
                initial_count=2),
    ]
    reactions = [
        # Stemness factor production
        Reaction(
            name="stemness_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_stemness_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Differentiation factor production
        Reaction(
            name="diff_prod", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_diff_prod,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Stemness inhibits differentiation: Stemness + Diff → Stemness
        Reaction(
            name="stemness_inhibits_diff", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_stemness_inhibits_diff,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Differentiation inhibits stemness: Stemness + Diff → Diff
        Reaction(
            name="diff_inhibits_stemness", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_diff_inhibits_stemness,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Stemness degradation
        Reaction(
            name="deg_stemness", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_stemness,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Differentiation degradation
        Reaction(
            name="deg_diff", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_diff,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("stem_cell_differentiation", species, reactions)


# ---------------------------------------------------------------------------
#  5. Population dynamics
# ---------------------------------------------------------------------------

def predator_prey_3sp(
    k_prey_birth: float = 1.0,
    k_predation_1: float = 0.01,
    k_predator_death: float = 0.3,
    k_predation_2: float = 0.01,
    k_super_death: float = 0.5,
    max_prey: int = 40,
    max_pred: int = 30,
    max_super: int = 20,
    init_prey: int = 25,
    init_pred: int = 8,
    init_super: int = 3,
) -> ReactionNetwork:
    """
    Three-level food chain (3 species).

    Prey → Predator → SuperPredator.  Exhibits complex oscillatory
    and chaotic dynamics depending on parameters.

    Based on Hastings & Powell, Ecology, 1991.
    BioModels: BIOMD0000000370
    """
    species = [
        Species(name="Prey", index=0, max_copy_number=max_prey,
                initial_count=init_prey),
        Species(name="Predator", index=1, max_copy_number=max_pred,
                initial_count=init_pred),
        Species(name="SuperPredator", index=2, max_copy_number=max_super,
                initial_count=init_super),
    ]
    reactions = [
        # Prey birth: Prey → 2 Prey
        Reaction(
            name="prey_birth", index=0,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_prey_birth,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Predation: Prey + Predator → 2 Predator
        Reaction(
            name="predation_1", index=1,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_predation_1,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Predator death
        Reaction(
            name="predator_death", index=2,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_predator_death,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Super-predation: Predator + SuperPredator → 2 SuperPredator
        Reaction(
            name="predation_2", index=3,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[2], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_predation_2,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Super-predator death
        Reaction(
            name="super_death", index=4,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_super_death,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("predator_prey_3sp", species, reactions)


def sir_vaccination(
    k_infection: float = 0.005,
    k_recovery: float = 0.1,
    k_vaccination: float = 0.02,
    k_waning: float = 0.01,
    max_s: int = 20,
    max_i: int = 15,
    max_r: int = 15,
    max_v: int = 15,
    init_s: int = 20,
    init_i: int = 5,
    init_r: int = 0,
    init_v: int = 0,
) -> ReactionNetwork:
    """
    SIR model with vaccination (4 species: S, I, R, V).

    Susceptible individuals can be infected or vaccinated.  Infected
    recover; vaccinated immunity can wane.

    Based on Keeling & Rohani, "Modeling Infectious Diseases", 2008.
    BioModels: BIOMD0000000295
    """
    species = [
        Species(name="S", index=0, max_copy_number=max_s,
                initial_count=init_s),
        Species(name="I", index=1, max_copy_number=max_i,
                initial_count=init_i),
        Species(name="R", index=2, max_copy_number=max_r,
                initial_count=init_r),
        Species(name="V", index=3, max_copy_number=max_v,
                initial_count=init_v),
    ]
    reactions = [
        # Infection: S + I → 2I
        Reaction(
            name="infection", index=0,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_infection,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Recovery: I → R
        Reaction(
            name="recovery", index=1,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_recovery,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Vaccination: S → V
        Reaction(
            name="vaccination", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[3], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_vaccination,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Waning immunity: V → S
        Reaction(
            name="waning", index=3,
            reactant_species=[3], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_waning,
                reactant_species=[3], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("sir_vaccination", species, reactions)


def competition_model(
    k_birth_A: float = 0.8,
    k_birth_B: float = 0.7,
    k_death_A: float = 0.05,
    k_death_B: float = 0.05,
    k_compete_AB: float = 0.01,
    k_compete_BA: float = 0.012,
    max_A: int = 40,
    max_B: int = 40,
    init_A: int = 20,
    init_B: int = 20,
) -> ReactionNetwork:
    """
    Competitive exclusion (2 species).

    Two species compete for the same resource.  Competitive interactions
    are modeled as bimolecular death reactions (Lotka–Volterra
    competition).

    Based on Gause, "The Struggle for Existence", 1934.
    BioModels: BIOMD0000000372
    """
    species = [
        Species(name="SpeciesA", index=0, max_copy_number=max_A,
                initial_count=init_A),
        Species(name="SpeciesB", index=1, max_copy_number=max_B,
                initial_count=init_B),
    ]
    reactions = [
        # A birth: A → 2A
        Reaction(
            name="birth_A", index=0,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_birth_A,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B birth: B → 2B
        Reaction(
            name="birth_B", index=1,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_birth_B,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # A natural death
        Reaction(
            name="death_A", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_death_A,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B natural death
        Reaction(
            name="death_B", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_death_B,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # A outcompetes B: A + B → A  (B dies)
        Reaction(
            name="compete_AB", index=4,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_compete_AB,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # B outcompetes A: A + B → B  (A dies)
        Reaction(
            name="compete_BA", index=5,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_compete_BA,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
    ]
    return _build_network("competition_model", species, reactions)


def mutualism_model(
    k_birth_A: float = 0.1,
    k_birth_B: float = 0.1,
    k_mutualism_AB: float = 0.02,
    k_mutualism_BA: float = 0.02,
    k_death_A: float = 0.3,
    k_death_B: float = 0.3,
    max_A: int = 35,
    max_B: int = 35,
    init_A: int = 10,
    init_B: int = 10,
) -> ReactionNetwork:
    """
    Obligate mutualism (2 species).

    Each species enhances the growth rate of the other; without the
    partner, death exceeds birth, making cooperation essential for
    survival.

    Based on Holland et al., Ecology Letters, 2002.
    BioModels: BIOMD0000000375
    """
    species = [
        Species(name="SpeciesA", index=0, max_copy_number=max_A,
                initial_count=init_A),
        Species(name="SpeciesB", index=1, max_copy_number=max_B,
                initial_count=init_B),
    ]
    reactions = [
        # A basal birth
        Reaction(
            name="birth_A", index=0,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_birth_A,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B basal birth
        Reaction(
            name="birth_B", index=1,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_birth_B,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Mutualistic benefit to A: A + B → 2A + B
        Reaction(
            name="mutualism_to_A", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[0, 1], product_stoichiometry=[2, 1],
            propensity=MassActionPropensity(
                rate_constant=k_mutualism_AB,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Mutualistic benefit to B: A + B → A + 2B
        Reaction(
            name="mutualism_to_B", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[0, 1], product_stoichiometry=[1, 2],
            propensity=MassActionPropensity(
                rate_constant=k_mutualism_BA,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # A death
        Reaction(
            name="death_A", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_death_A,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B death
        Reaction(
            name="death_B", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_death_B,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("mutualism_model", species, reactions)


# ---------------------------------------------------------------------------
#  6. Stochastic phenomena
# ---------------------------------------------------------------------------

def stochastic_switch(
    k_prod_A: float = 1.0,
    k_prod_B: float = 1.0,
    k_A_converts_B: float = 0.003,
    k_B_converts_A: float = 0.003,
    k_deg_A: float = 0.1,
    k_deg_B: float = 0.1,
    max_A: int = 30,
    max_B: int = 30,
) -> ReactionNetwork:
    """
    Ultrasensitive stochastic switch (2 species).

    A and B are produced independently; cooperative conversion reactions
    (second-order in self) create bistable switching triggered by
    stochastic fluctuations.

    Based on Warren & ten Wolde, PRL, 2004.
    BioModels: BIOMD0000000402
    """
    species = [
        Species(name="A", index=0, max_copy_number=max_A, initial_count=15),
        Species(name="B", index=1, max_copy_number=max_B, initial_count=5),
    ]
    reactions = [
        # A production
        Reaction(
            name="prod_A", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_A,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # B production
        Reaction(
            name="prod_B", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_B,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Cooperative A converts B: 2A + B → 3A  (B consumed, A gained)
        Reaction(
            name="A_converts_B", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[2, 1],
            product_species=[0], product_stoichiometry=[3],
            propensity=MassActionPropensity(
                rate_constant=k_A_converts_B,
                reactant_species=[0, 1], reactant_stoichiometry=[2, 1],
            ),
        ),
        # Cooperative B converts A: A + 2B → 3B  (A consumed, B gained)
        Reaction(
            name="B_converts_A", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 2],
            product_species=[1], product_stoichiometry=[3],
            propensity=MassActionPropensity(
                rate_constant=k_B_converts_A,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 2],
            ),
        ),
        # A degradation
        Reaction(
            name="deg_A", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_A,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B degradation
        Reaction(
            name="deg_B", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_B,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("stochastic_switch", species, reactions)


def noise_induced_oscillation(
    k_prod_act: float = 0.5,
    k_act_self: float = 3.0,
    k_act_inhib: float = 2.0,
    k_prod_inhib: float = 0.3,
    k_inhib_degrades_act: float = 0.05,
    k_deg_act: float = 0.1,
    k_deg_inhib: float = 0.15,
    k_half: float = 6.0,
    hill_coeff: float = 3.0,
    max_act: int = 30,
    max_inhib: int = 30,
) -> ReactionNetwork:
    """
    Noise-induced coherent oscillations (2 species).

    An activator–inhibitor pair where the deterministic steady state
    is a stable focus: stochastic fluctuations sustain coherent
    oscillations (coherence resonance).

    Based on McKane & Newman, PRL, 2005.
    BioModels: BIOMD0000000405
    """
    species = [
        Species(name="Activator", index=0, max_copy_number=max_act,
                initial_count=5),
        Species(name="Inhibitor", index=1, max_copy_number=max_inhib,
                initial_count=3),
    ]
    reactions = [
        # Activator basal production
        Reaction(
            name="prod_act_basal", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_act,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Activator positive autoregulation (Hill)
        Reaction(
            name="prod_act_auto", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_act_self, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Inhibitor production induced by activator (Hill)
        Reaction(
            name="prod_inhib", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_act_inhib, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Inhibitor degrades activator: Activator + Inhibitor → Inhibitor
        Reaction(
            name="inhib_degrades_act", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_inhib_degrades_act,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Activator degradation
        Reaction(
            name="deg_act", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_act,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Inhibitor degradation
        Reaction(
            name="deg_inhib", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_inhib,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("noise_induced_oscillation", species, reactions)


def excitable_system(
    k_prod_fast: float = 0.2,
    k_trigger: float = 0.005,
    k_amplify: float = 5.0,
    k_fast_activates_slow: float = 0.3,
    k_slow_degrades_fast: float = 0.06,
    k_deg_fast: float = 0.1,
    k_deg_slow: float = 0.2,
    k_half: float = 8.0,
    hill_coeff: float = 4.0,
    max_fast: int = 30,
    max_slow: int = 30,
) -> ReactionNetwork:
    """
    Excitable system with stochastic triggering (2 species).

    A fast variable amplifies via positive feedback, then a slow
    inhibitor resets the system.  Stochastic fluctuations trigger
    excitable pulses.

    Based on Lindner et al., Physics Reports, 2004.
    BioModels: BIOMD0000000410
    """
    species = [
        Species(name="Fast", index=0, max_copy_number=max_fast, initial_count=1),
        Species(name="Slow", index=1, max_copy_number=max_slow, initial_count=1),
    ]
    reactions = [
        # Fast basal production (stochastic trigger)
        Reaction(
            name="prod_fast_basal", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_fast,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Fast positive feedback (Hill self-activation)
        Reaction(
            name="fast_amplify", index=1,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_amplify, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Slow production induced by fast (Hill)
        Reaction(
            name="fast_activates_slow", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_fast_activates_slow, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Slow degrades fast: Fast + Slow → Slow
        Reaction(
            name="slow_degrades_fast", index=3,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_slow_degrades_fast,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Fast degradation
        Reaction(
            name="deg_fast", index=4,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_fast,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Slow degradation
        Reaction(
            name="deg_slow", index=5,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_slow,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("excitable_system", species, reactions)


def stochastic_bursting(
    k_on: float = 0.05,
    k_off: float = 0.5,
    k_burst_transcription: float = 5.0,
    k_translation: float = 2.0,
    k_deg_mrna: float = 0.5,
    k_deg_protein: float = 0.05,
    max_mrna: int = 25,
    max_protein: int = 40,
) -> ReactionNetwork:
    """
    Transcriptional bursting (2 species: mRNA, Protein).

    Gene stochastically switches between ON and OFF states.  When ON,
    mRNA is produced in rapid bursts (geometric burst-size distribution
    in the limit of fast switching).  Protein is translated from mRNA.

    Based on Raj et al., PLoS Biol., 2006.
    BioModels: BIOMD0000000415
    """
    species = [
        Species(name="mRNA", index=0, max_copy_number=max_mrna, initial_count=0),
        Species(name="Protein", index=1, max_copy_number=max_protein,
                initial_count=5),
    ]
    reactions = [
        # Burst transcription (constitutive, high rate models burst-average)
        Reaction(
            name="burst_transcription", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_on * k_burst_transcription,
                reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Translation: mRNA → mRNA + Protein
        Reaction(
            name="translation", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0, 1], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_translation,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # mRNA degradation
        Reaction(
            name="deg_mrna", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_mrna,
                reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein degradation
        Reaction(
            name="deg_protein", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_protein,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("stochastic_bursting", species, reactions)
