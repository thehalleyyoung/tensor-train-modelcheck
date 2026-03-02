"""
Load reaction network definitions from YAML/JSON/SBML files.

Reads a structured network definition and constructs a ReactionNetwork
by calling the existing add_species(), add_reaction(), etc. methods.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from tn_check.cme.reaction_network import ReactionNetwork, KineticsType

logger = logging.getLogger(__name__)


def load_network(path: str | Path) -> ReactionNetwork:
    """Load a ReactionNetwork from a YAML, JSON, or SBML file.

    Args:
        path: Path to the network definition file.

    Returns:
        A fully constructed ReactionNetwork.

    Raises:
        ValueError: If the file format is unsupported or the definition is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path}")

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix in (".xml", ".sbml"):
        return _load_sbml(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix} "
            "(use .yaml, .yml, .json, .xml, or .sbml)"
        )

    return _build_network_from_dict(data)


def validate_network_dict(data: dict[str, Any]) -> list[str]:
    """Validate a network definition dictionary without building it.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if "species" not in data:
        errors.append("Missing required field: 'species'")
    elif not isinstance(data["species"], list) or len(data["species"]) == 0:
        errors.append("'species' must be a non-empty list")

    if "reactions" not in data:
        errors.append("Missing required field: 'reactions'")
    elif not isinstance(data["reactions"], list) or len(data["reactions"]) == 0:
        errors.append("'reactions' must be a non-empty list")

    if errors:
        return errors

    species_names = set()
    for i, sp in enumerate(data["species"]):
        if "name" not in sp:
            errors.append(f"Species {i}: missing 'name'")
        else:
            if sp["name"] in species_names:
                errors.append(f"Species {i}: duplicate name '{sp['name']}'")
            species_names.add(sp["name"])

    for i, rxn in enumerate(data["reactions"]):
        if "name" not in rxn:
            errors.append(f"Reaction {i}: missing 'name'")

        kinetics = rxn.get("kinetics", "mass_action")
        if kinetics == "mass_action":
            if "rate_constant" not in rxn:
                errors.append(f"Reaction {i} ('{rxn.get('name', '?')}'): mass_action requires 'rate_constant'")
            # Validate species references
            for sp_name in rxn.get("reactants", {}):
                if sp_name not in species_names:
                    errors.append(f"Reaction {i}: unknown reactant species '{sp_name}'")
            for sp_name in rxn.get("products", {}):
                if sp_name not in species_names:
                    errors.append(f"Reaction {i}: unknown product species '{sp_name}'")
        elif kinetics == "hill":
            if "hill" not in rxn:
                errors.append(f"Reaction {i} ('{rxn.get('name', '?')}'): hill kinetics requires 'hill' block")
            else:
                h = rxn["hill"]
                for field in ("regulator", "v_max", "k_half", "hill_coefficient"):
                    if field not in h:
                        errors.append(f"Reaction {i} hill: missing '{field}'")
                if "regulator" in h and h["regulator"] not in species_names:
                    errors.append(f"Reaction {i} hill: unknown regulator '{h['regulator']}'")
        elif kinetics == "michaelis_menten":
            if "michaelis_menten" not in rxn:
                errors.append(f"Reaction {i} ('{rxn.get('name', '?')}'): michaelis_menten kinetics requires 'michaelis_menten' block")
            else:
                mm = rxn["michaelis_menten"]
                for field in ("substrate", "product", "enzyme", "k_cat", "k_m"):
                    if field not in mm:
                        errors.append(f"Reaction {i} michaelis_menten: missing '{field}'")
                for field in ("substrate", "product", "enzyme"):
                    if field in mm and mm[field] not in species_names:
                        errors.append(f"Reaction {i} michaelis_menten: unknown species '{mm[field]}'")
        else:
            errors.append(f"Reaction {i}: unsupported kinetics type '{kinetics}'")

    return errors


def _build_network_from_dict(data: dict[str, Any]) -> ReactionNetwork:
    """Build a ReactionNetwork from a parsed dictionary."""
    errors = validate_network_dict(data)
    if errors:
        raise ValueError("Invalid network definition:\n  " + "\n  ".join(errors))

    net = ReactionNetwork(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
    )

    # Add species
    for sp_def in data["species"]:
        net.add_species(
            name=sp_def["name"],
            max_copy_number=sp_def.get("max_copy_number", 50),
            initial_count=sp_def.get("initial_count", 0),
            description=sp_def.get("description", ""),
        )

    # Add reactions
    for rxn_def in data["reactions"]:
        kinetics = rxn_def.get("kinetics", "mass_action")

        if kinetics == "mass_action":
            net.add_reaction(
                name=rxn_def["name"],
                reactants=rxn_def.get("reactants", {}),
                products=rxn_def.get("products", {}),
                rate_constant=rxn_def["rate_constant"],
                description=rxn_def.get("description", ""),
            )
        elif kinetics == "hill":
            h = rxn_def["hill"]
            net.add_reaction_with_hill(
                name=rxn_def["name"],
                products=rxn_def.get("products", {}),
                regulator=h["regulator"],
                v_max=h["v_max"],
                k_half=h["k_half"],
                hill_coefficient=h["hill_coefficient"],
                activation=h.get("activation", True),
                description=rxn_def.get("description", ""),
            )
        elif kinetics == "michaelis_menten":
            mm = rxn_def["michaelis_menten"]
            net.add_reaction_with_michaelis_menten(
                name=rxn_def["name"],
                substrate=mm["substrate"],
                product=mm["product"],
                enzyme=mm["enzyme"],
                k_cat=mm["k_cat"],
                k_m=mm["k_m"],
                description=rxn_def.get("description", ""),
            )

    return net


# ---------------------------------------------------------------------------
# SBML import
# ---------------------------------------------------------------------------

_SBML_NS_PREFIXES = [
    "http://www.sbml.org/sbml/level3/version2/core",
    "http://www.sbml.org/sbml/level3/version1/core",
    "http://www.sbml.org/sbml/level2/version5",
    "http://www.sbml.org/sbml/level2/version4",
    "http://www.sbml.org/sbml/level2/version3",
    "http://www.sbml.org/sbml/level2",
    "",
]


def _sbml_find(parent: ET.Element, tag: str) -> ET.Element | None:
    """Find an SBML element trying common namespace prefixes."""
    for ns in _SBML_NS_PREFIXES:
        prefix = f"{{{ns}}}" if ns else ""
        elem = parent.find(f"{prefix}{tag}")
        if elem is not None:
            return elem
    return None


def _sbml_findall(parent: ET.Element, tag: str) -> list[ET.Element]:
    """Find all SBML elements trying common namespace prefixes."""
    for ns in _SBML_NS_PREFIXES:
        prefix = f"{{{ns}}}" if ns else ""
        elems = parent.findall(f"{prefix}{tag}")
        if elems:
            return elems
    return []


def _detect_hill_kinetics(
    math_elem: ET.Element | None,
    local_params: dict[str, float],
    species_names: set[str],
) -> dict[str, Any] | None:
    """Attempt to detect Hill-function kinetics from SBML math/parameters.

    Heuristic: if parameters named v_max/Vmax/V and K/K_half/Kd and
    n/hill/h are present, treat as Hill kinetics.
    """
    param_lower = {k.lower(): (k, v) for k, v in local_params.items()}

    v_max_keys = ["vmax", "v_max", "v"]
    k_keys = ["k", "k_half", "kd", "km", "k_m"]
    h_keys = ["n", "hill", "h", "hill_coefficient", "hillcoeff"]

    v_max = k_half = hill_coeff = None
    for k in v_max_keys:
        if k in param_lower:
            v_max = param_lower[k][1]
            break
    for k in k_keys:
        if k in param_lower:
            k_half = param_lower[k][1]
            break
    for k in h_keys:
        if k in param_lower:
            hill_coeff = param_lower[k][1]
            break

    if v_max is not None and k_half is not None and hill_coeff is not None:
        return {"v_max": v_max, "k_half": k_half, "hill_coefficient": hill_coeff}
    return None


def _load_sbml(path: Path) -> ReactionNetwork:
    """Parse an SBML XML file and construct a ReactionNetwork.

    Supports SBML Level 2 and Level 3 core. Extracts species and
    reactions, converting mass-action and Hill kinetics to native
    ReactionNetwork representations.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()

    # Find <model>
    model = _sbml_find(root, "model")
    if model is None:
        model = root  # Some files embed model at root

    model_id = model.get("id", "") or model.get("name", "")
    net_name = model_id or path.stem

    # Collect species
    species_list_el = _sbml_find(model, "listOfSpecies")
    if species_list_el is None:
        raise ValueError("SBML file has no <listOfSpecies>")

    species_elems = _sbml_findall(species_list_el, "species")
    if not species_elems:
        raise ValueError("SBML file has no <species> elements")

    species_names: list[str] = []
    species_initial: dict[str, int] = {}
    for sp_el in species_elems:
        sp_id = sp_el.get("id", sp_el.get("name", ""))
        if not sp_id:
            continue
        species_names.append(sp_id)
        # Initial amount or concentration (rounded to int for discrete CME)
        init_val = sp_el.get("initialAmount", sp_el.get("initialConcentration", "0"))
        try:
            species_initial[sp_id] = int(float(init_val))
        except (ValueError, TypeError):
            species_initial[sp_id] = 0

    species_set = set(species_names)
    default_max_copy = 50

    net = ReactionNetwork(name=net_name, description=f"Imported from SBML: {path.name}")
    for sp_id in species_names:
        net.add_species(
            name=sp_id,
            max_copy_number=default_max_copy,
            initial_count=species_initial.get(sp_id, 0),
        )

    # Collect reactions
    reaction_list_el = _sbml_find(model, "listOfReactions")
    if reaction_list_el is None:
        logger.warning("SBML file has no <listOfReactions>; returning species-only network")
        return net

    reaction_elems = _sbml_findall(reaction_list_el, "reaction")

    for rxn_el in reaction_elems:
        rxn_id = rxn_el.get("id", rxn_el.get("name", f"rxn_{len(net.reactions)}"))
        reversible = rxn_el.get("reversible", "false").lower() == "true"

        # Parse reactants
        reactants: dict[str, int] = {}
        reactants_el = _sbml_find(rxn_el, "listOfReactants")
        if reactants_el is not None:
            for sr in _sbml_findall(reactants_el, "speciesReference"):
                sp_ref = sr.get("species", "")
                stoich = sr.get("stoichiometry", "1")
                if sp_ref in species_set:
                    reactants[sp_ref] = int(float(stoich))

        # Parse products
        products: dict[str, int] = {}
        products_el = _sbml_find(rxn_el, "listOfProducts")
        if products_el is not None:
            for sr in _sbml_findall(products_el, "speciesReference"):
                sp_ref = sr.get("species", "")
                stoich = sr.get("stoichiometry", "1")
                if sp_ref in species_set:
                    products[sp_ref] = int(float(stoich))

        # Extract kinetic parameters
        local_params: dict[str, float] = {}
        kinetic_law = _sbml_find(rxn_el, "kineticLaw")
        if kinetic_law is not None:
            lop = _sbml_find(kinetic_law, "listOfParameters") or \
                  _sbml_find(kinetic_law, "listOfLocalParameters")
            if lop is not None:
                for p in _sbml_findall(lop, "parameter") + \
                         _sbml_findall(lop, "localParameter"):
                    pid = p.get("id", p.get("name", ""))
                    pval = p.get("value", "0")
                    try:
                        local_params[pid] = float(pval)
                    except (ValueError, TypeError):
                        pass

        # Determine kinetics type
        math_el = _sbml_find(kinetic_law, "math") if kinetic_law is not None else None
        hill_info = _detect_hill_kinetics(math_el, local_params, species_set)

        if hill_info is not None:
            # Find the regulator: a reactant species or modifier
            modifiers_el = _sbml_find(rxn_el, "listOfModifiers")
            regulator = None
            if modifiers_el is not None:
                for mod in _sbml_findall(modifiers_el, "modifierSpeciesReference"):
                    mod_sp = mod.get("species", "")
                    if mod_sp in species_set:
                        regulator = mod_sp
                        break
            if regulator is None and reactants:
                regulator = next(iter(reactants))
            if regulator is None:
                regulator = species_names[0]

            net.add_reaction_with_hill(
                name=rxn_id,
                products=products if products else {regulator: 0},
                regulator=regulator,
                v_max=hill_info["v_max"],
                k_half=hill_info["k_half"],
                hill_coefficient=hill_info["hill_coefficient"],
                activation=True,
                description=f"Hill kinetics (from SBML)",
            )
        else:
            # Default to mass-action
            rate_key_candidates = ["k", "rate", "c", "kf", "kr", "k1", "k_f",
                                   "rate_constant", "kcat"]
            rate_constant = 1.0
            for rk in rate_key_candidates:
                if rk in local_params:
                    rate_constant = local_params[rk]
                    break
            else:
                # Take the first parameter if available
                if local_params:
                    rate_constant = next(iter(local_params.values()))

            net.add_reaction(
                name=rxn_id,
                reactants=reactants,
                products=products,
                rate_constant=rate_constant,
                description=f"Mass-action (from SBML)",
            )

        # Handle reversible reactions by adding a reverse reaction
        if reversible and not hill_info:
            rev_rate = 1.0
            rev_keys = ["kr", "k_r", "k_rev", "k2", "krev"]
            for rk in rev_keys:
                if rk in local_params:
                    rev_rate = local_params[rk]
                    break

            rev_name = f"{rxn_id}_reverse"
            if rev_name not in [r.name for r in net.reactions]:
                net.add_reaction(
                    name=rev_name,
                    reactants=products,
                    products=reactants,
                    rate_constant=rev_rate,
                    description=f"Reverse of {rxn_id} (from SBML)",
                )

    return net


STARTER_TEMPLATE = """\
# TN-Check reaction network definition
# See: https://github.com/tn-check/tn-check for documentation

name: my_network
description: "A starter reaction network template"

species:
  - name: A
    max_copy_number: 50
    initial_count: 0
    description: "Species A"

  - name: B
    max_copy_number: 50
    initial_count: 0
    description: "Species B"

reactions:
  # Production of A (zeroth-order, constitutive)
  - name: produce_A
    reactants: {}
    products: {A: 1}
    rate_constant: 1.0
    kinetics: mass_action
    description: "Constitutive production of A"

  # Degradation of A (first-order)
  - name: degrade_A
    reactants: {A: 1}
    products: {}
    rate_constant: 0.1
    kinetics: mass_action
    description: "Linear degradation of A"

  # A catalyzes production of B (first-order in A, catalytic)
  - name: produce_B
    reactants: {A: 1}
    products: {A: 1, B: 1}
    rate_constant: 0.5
    kinetics: mass_action
    description: "A catalyzes production of B"

  # Degradation of B
  - name: degrade_B
    reactants: {B: 1}
    products: {}
    rate_constant: 0.05
    kinetics: mass_action
    description: "Linear degradation of B"

  # --- Hill kinetics example (uncomment to use) ---
  # - name: hill_production
  #   products: {B: 1}
  #   kinetics: hill
  #   hill:
  #     regulator: A
  #     v_max: 10.0
  #     k_half: 5.0
  #     hill_coefficient: 2.0
  #     activation: true
  #   description: "Hill-type activation of B by A"

  # --- Michaelis-Menten example (uncomment to use) ---
  # Requires an enzyme species; add one to the species list first.
  # - name: enzymatic_conversion
  #   kinetics: michaelis_menten
  #   michaelis_menten:
  #     substrate: A
  #     product: B
  #     enzyme: E
  #     k_cat: 1.0
  #     k_m: 10.0
  #   description: "Enzyme-catalyzed conversion of A to B"
"""
