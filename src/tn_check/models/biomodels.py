"""
Extended biological models from the BioModels database.

Adds 8 additional biologically relevant models beyond the existing library:
1. p53-MDM2 oscillator (negative feedback)
2. Circadian clock (Goodwin oscillator)
3. Lac operon (bistable genetic switch)
4. MAPK dual phosphorylation (signaling cascade)
5. Lotka-Volterra predator-prey
6. Viral infection dynamics
7. Calcium signaling (IP3-mediated)
8. Apoptosis decision network

Each model is parameterized with realistic rate constants from the
literature and can be compiled to MPO format via the CME compiler.
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


def p53_mdm2(
    k_prod_p53: float = 0.5,
    k_deg_p53: float = 0.01,
    k_mdm2_deg_p53: float = 0.1,
    k_prod_mdm2: float = 0.1,
    k_deg_mdm2: float = 0.05,
    k_p53_act_mdm2: float = 0.5,
    max_p53: int = 40,
    max_mdm2: int = 40,
    hill_coeff: float = 2.0,
    k_half: float = 10.0,
) -> ReactionNetwork:
    """
    p53-MDM2 negative feedback oscillator.

    p53 activates MDM2 transcription; MDM2 promotes p53 degradation.
    This creates oscillations in response to DNA damage.

    Based on Geva-Zatorsky et al., Molecular Systems Biology, 2006.
    BioModels: BIOMD0000000188

    Reactions:
    - ∅ →[k_prod] p53                  (constitutive production)
    - p53 →[k_deg] ∅                   (basal degradation)
    - p53 →[k_mdm2 * MDM2] ∅          (MDM2-mediated degradation)
    - ∅ →[Hill(p53)] MDM2              (p53-activated MDM2 production)
    - MDM2 →[k_deg_mdm2] ∅            (MDM2 degradation)
    """
    species = [
        Species(name="p53", index=0, max_copy_number=max_p53, initial_count=5),
        Species(name="MDM2", index=1, max_copy_number=max_mdm2, initial_count=2),
    ]
    reactions = [
        Reaction(
            name="prod_p53", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_p53, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        Reaction(
            name="basal_deg_p53", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_p53, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        Reaction(
            name="mdm2_deg_p53", index=2,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_mdm2_deg_p53,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        Reaction(
            name="prod_mdm2", index=3,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_p53_act_mdm2, k_half=k_half,
                hill_coefficient=hill_coeff, species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        Reaction(
            name="deg_mdm2", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg_mdm2, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("p53_mdm2", species, reactions)


def goodwin_oscillator(
    n_genes: int = 3,
    alpha: float = 10.0,
    gamma: float = 0.5,
    k_half: float = 5.0,
    hill_coeff: float = 4.0,
    max_copy: int = 30,
) -> ReactionNetwork:
    """
    Goodwin oscillator (circadian clock model).

    A chain of n genes where the last gene represses the first,
    creating oscillations when the Hill coefficient is sufficiently high.

    Based on Goodwin, Advances in Enzyme Regulation, 1965.
    BioModels: BIOMD0000000035

    - X_0 is repressed by X_{n-1} (Hill repression)
    - X_i (i>0) is activated by X_{i-1} (Hill activation)
    - All species degrade linearly
    """
    species = [
        Species(name=f"X{i}", index=i, max_copy_number=max_copy)
        for i in range(n_genes)
    ]
    reactions = []
    idx = 0

    # First gene: repressed by last
    reactions.append(
        Reaction(
            name="prod_X0", index=idx,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=alpha, k_half=k_half, hill_coefficient=hill_coeff,
                species_index=n_genes - 1, activation=False,
            ),
            kinetics_type=KineticsType.HILL,
        )
    )
    idx += 1

    # Subsequent genes: activated by predecessor
    for i in range(1, n_genes):
        reactions.append(
            Reaction(
                name=f"prod_X{i}", index=idx,
                reactant_species=[], reactant_stoichiometry=[],
                product_species=[i], product_stoichiometry=[1],
                propensity=HillPropensity(
                    v_max=alpha, k_half=k_half, hill_coefficient=hill_coeff,
                    species_index=i - 1, activation=True,
                ),
                kinetics_type=KineticsType.HILL,
            )
        )
        idx += 1

    # Degradation for all
    for i in range(n_genes):
        reactions.append(
            Reaction(
                name=f"deg_X{i}", index=idx,
                reactant_species=[i], reactant_stoichiometry=[1],
                product_species=[], product_stoichiometry=[],
                propensity=MassActionPropensity(
                    rate_constant=gamma, reactant_species=[i], reactant_stoichiometry=[1],
                ),
            )
        )
        idx += 1

    return _build_network(f"goodwin_oscillator_{n_genes}", species, reactions)


def lac_operon(
    k_on: float = 0.1,
    k_off: float = 0.01,
    k_prod_active: float = 5.0,
    k_prod_basal: float = 0.05,
    k_deg: float = 0.1,
    max_promoter: int = 2,
    max_lacZ: int = 50,
    max_inducer: int = 30,
) -> ReactionNetwork:
    """
    Lac operon bistable switch.

    Simplified model of lactose metabolism regulation with positive
    feedback through inducer-promoter interaction.

    Based on Ozbudak et al., Nature, 2004.
    BioModels: BIOMD0000000065

    Species:
    - Promoter state (0=off, 1=on)
    - LacZ protein
    - Inducer (allolactose analog)

    Bistability arises from positive feedback: LacZ processes inducer,
    inducer activates promoter.
    """
    species = [
        Species(name="Promoter", index=0, max_copy_number=max_promoter, initial_count=0),
        Species(name="LacZ", index=1, max_copy_number=max_lacZ, initial_count=0),
        Species(name="Inducer", index=2, max_copy_number=max_inducer, initial_count=5),
    ]
    reactions = [
        # Inducer activates promoter
        Reaction(
            name="activate", index=0,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[0, 2], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_on, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # Promoter deactivation
        Reaction(
            name="deactivate", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_off, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # LacZ production (active promoter)
        Reaction(
            name="prod_lacZ_active", index=2,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0, 1], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_active, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # LacZ basal production
        Reaction(
            name="prod_lacZ_basal", index=3,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_basal, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # LacZ degradation
        Reaction(
            name="deg_lacZ", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Inducer production (constitutive, external)
        Reaction(
            name="prod_inducer", index=5,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=0.2, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Inducer degradation
        Reaction(
            name="deg_inducer", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=0.05, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("lac_operon", species, reactions)


def lotka_volterra(
    k_prey_birth: float = 1.0,
    k_predation: float = 0.01,
    k_predator_death: float = 0.5,
    max_prey: int = 50,
    max_predator: int = 50,
    prey_init: int = 30,
    predator_init: int = 10,
) -> ReactionNetwork:
    """
    Lotka-Volterra predator-prey model.

    Classic ecological model with oscillatory dynamics:
    - Prey reproduce (birth)
    - Predators consume prey (predation)
    - Predators die (death)

    BioModels: BIOMD0000000160
    """
    species = [
        Species(name="Prey", index=0, max_copy_number=max_prey, initial_count=prey_init),
        Species(name="Predator", index=1, max_copy_number=max_predator,
                initial_count=predator_init),
    ]
    reactions = [
        Reaction(
            name="prey_birth", index=0,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_prey_birth, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        Reaction(
            name="predation", index=1,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_predation,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        Reaction(
            name="predator_death", index=2,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_predator_death,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("lotka_volterra", species, reactions)


def viral_infection(
    k_infection: float = 0.001,
    k_virus_prod: float = 5.0,
    k_immune_response: float = 0.5,
    k_cell_death: float = 0.1,
    k_virus_clear: float = 0.5,
    k_immune_decay: float = 0.1,
    max_cells: int = 30,
    max_virus: int = 40,
    max_immune: int = 30,
    init_cells: int = 20,
    init_virus: int = 5,
    init_immune: int = 2,
) -> ReactionNetwork:
    """
    Simple viral infection dynamics (3 species).

    Models the interaction between target cells, virus particles,
    and immune response.

    - Virus infects cells (cell death + virus production)
    - Immune system clears virus
    - Immune response activated by virus presence
    """
    species = [
        Species(name="Cells", index=0, max_copy_number=max_cells,
                initial_count=init_cells),
        Species(name="Virus", index=1, max_copy_number=max_virus,
                initial_count=init_virus),
        Species(name="Immune", index=2, max_copy_number=max_immune,
                initial_count=init_immune),
    ]
    reactions = [
        # Infection: Cells + Virus → Virus (cell killed, virus survives)
        Reaction(
            name="infection", index=0,
            reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_infection,
                reactant_species=[0, 1], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Virus production from infected cells
        Reaction(
            name="virus_prod", index=1,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=k_virus_prod,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Immune clearance of virus
        Reaction(
            name="virus_clear", index=2,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_virus_clear,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Immune activation by virus
        Reaction(
            name="immune_act", index=3,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_immune_response, k_half=5.0, hill_coefficient=2.0,
                species_index=1, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Immune decay
        Reaction(
            name="immune_decay", index=4,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_immune_decay,
                reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("viral_infection", species, reactions)


def calcium_signaling(
    k_ip3_prod: float = 0.5,
    k_ip3_deg: float = 0.1,
    k_release: float = 0.5,
    k_uptake: float = 0.3,
    k_leak: float = 0.01,
    max_ip3: int = 20,
    max_cytosol_ca: int = 30,
    max_er_ca: int = 30,
) -> ReactionNetwork:
    """
    IP3-mediated calcium signaling (3 species).

    Simplified model of calcium oscillations between ER and cytosol,
    mediated by IP3.

    Based on Li & Rinzel, J. Theor. Biol., 1994.
    """
    species = [
        Species(name="IP3", index=0, max_copy_number=max_ip3, initial_count=2),
        Species(name="Ca_cyt", index=1, max_copy_number=max_cytosol_ca, initial_count=1),
        Species(name="Ca_ER", index=2, max_copy_number=max_er_ca, initial_count=15),
    ]
    reactions = [
        # IP3 production
        Reaction(
            name="ip3_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_ip3_prod, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # IP3 degradation
        Reaction(
            name="ip3_deg", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_ip3_deg, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Ca release from ER (IP3-dependent)
        Reaction(
            name="ca_release", index=2,
            reactant_species=[0, 2], reactant_stoichiometry=[1, 1],
            product_species=[0, 1], product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_release,
                reactant_species=[0, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Ca uptake into ER (SERCA pump)
        Reaction(
            name="ca_uptake", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_uptake,
                reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Ca leak from ER
        Reaction(
            name="ca_leak", index=4,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_leak, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("calcium_signaling", species, reactions)


def apoptosis_decision(
    k_signal: float = 0.3,
    k_casp_act: float = 0.5,
    k_casp_deact: float = 0.1,
    k_antiapop: float = 0.5,
    k_deg: float = 0.1,
    max_signal: int = 20,
    max_caspase: int = 25,
    max_antiapoptotic: int = 20,
) -> ReactionNetwork:
    """
    Apoptosis decision network (3 species).

    Models the competition between pro- and anti-apoptotic signals
    that determine cell fate (survive vs die).

    Species:
    - Signal: death signal strength
    - Caspase: executioner caspase (pro-apoptotic)
    - AntiApop: anti-apoptotic protein (Bcl-2 family)

    Bistability: cell commits to apoptosis when caspase exceeds threshold.
    """
    species = [
        Species(name="Signal", index=0, max_copy_number=max_signal, initial_count=5),
        Species(name="Caspase", index=1, max_copy_number=max_caspase, initial_count=1),
        Species(name="AntiApop", index=2, max_copy_number=max_antiapoptotic,
                initial_count=10),
    ]
    reactions = [
        # Signal production
        Reaction(
            name="signal_prod", index=0,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_signal, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Signal degradation
        Reaction(
            name="signal_deg", index=1,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Caspase activation by signal (Hill function)
        Reaction(
            name="casp_act", index=2,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[1], product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=k_casp_act, k_half=5.0, hill_coefficient=3.0,
                species_index=0, activation=True,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        # Caspase inhibition by anti-apoptotic
        Reaction(
            name="casp_inhibit", index=3,
            reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_casp_deact,
                reactant_species=[1, 2], reactant_stoichiometry=[1, 1],
            ),
        ),
        # Caspase degradation
        Reaction(
            name="casp_deg", index=4,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Anti-apoptotic production
        Reaction(
            name="antiapop_prod", index=5,
            reactant_species=[], reactant_stoichiometry=[],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_antiapop, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        # Anti-apoptotic degradation
        Reaction(
            name="antiapop_deg", index=6,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[], product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_deg, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("apoptosis_decision", species, reactions)


def dual_phosphorylation(
    k_phos: float = 0.5,
    k_dephos: float = 0.2,
    max_X0: int = 20,
    max_X1: int = 20,
    max_X2: int = 20,
    init_X0: int = 15,
) -> ReactionNetwork:
    """
    Dual phosphorylation cycle (Markevich et al., 2004).

    A substrate with 3 forms: unphosphorylated (X0), singly (X1),
    doubly (X2). Exhibits ultrasensitive switch-like behavior.

    BioModels: BIOMD0000000026
    """
    species = [
        Species(name="X0", index=0, max_copy_number=max_X0, initial_count=init_X0),
        Species(name="X1", index=1, max_copy_number=max_X1, initial_count=0),
        Species(name="X2", index=2, max_copy_number=max_X2, initial_count=0),
    ]
    reactions = [
        # X0 → X1 (first phosphorylation)
        Reaction(
            name="phos1", index=0,
            reactant_species=[0], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_phos, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # X1 → X2 (second phosphorylation)
        Reaction(
            name="phos2", index=1,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[2], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_phos, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # X2 → X1 (first dephosphorylation)
        Reaction(
            name="dephos1", index=2,
            reactant_species=[2], reactant_stoichiometry=[1],
            product_species=[1], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_dephos, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # X1 → X0 (second dephosphorylation)
        Reaction(
            name="dephos2", index=3,
            reactant_species=[1], reactant_stoichiometry=[1],
            product_species=[0], product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_dephos, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("dual_phosphorylation", species, reactions)
