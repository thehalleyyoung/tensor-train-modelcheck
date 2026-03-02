"""
Command-line interface for TN-Check.

Provides subcommands for model checking, certificate verification,
network inspection, and template generation.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
import time
from pathlib import Path

import click

from tn_check import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _resolve_property(
    property: str | None,
    steady_state: tuple[str, ...] | None,
    extinction: str | None,
    net: object,
) -> str:
    """Resolve a CSL property from either a full string or shorthand flags.

    Args:
        property: Full CSL formula string, or None if using shorthands.
        steady_state: Tuple of (species_name, threshold) for --steady-state.
        extinction: Species name for --extinction.
        net: A ReactionNetwork (used to map species names to indices).

    Returns:
        A CSL formula string.
    """
    if property:
        return property

    species_names = [sp.name for sp in net.species]  # type: ignore[attr-defined]

    if steady_state:
        sp_name, threshold_str = steady_state
        if sp_name in species_names:
            sp_idx = species_names.index(sp_name)
        else:
            try:
                sp_idx = int(sp_name)
            except ValueError:
                raise click.BadParameter(
                    f"Unknown species '{sp_name}'. "
                    f"Available: {', '.join(species_names)}"
                )
        return f"S>=0.5 [X_{sp_idx} >= {threshold_str}]"

    if extinction:
        sp_name = extinction
        if sp_name in species_names:
            sp_idx = species_names.index(sp_name)
        else:
            try:
                sp_idx = int(sp_name)
            except ValueError:
                raise click.BadParameter(
                    f"Unknown species '{sp_name}'. "
                    f"Available: {', '.join(species_names)}"
                )
        return f"P>=0.5 [true U<=100.0 X_{sp_idx} <= 0]"

    raise click.UsageError(
        "Provide a CSL PROPERTY argument, or use --steady-state / --extinction."
    )


def _format_result(
    result_data: dict,
    fmt: str,
) -> str:
    """Format a result dict as text, json, or csv."""
    if fmt == "json":
        return json.dumps(result_data, indent=2)
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(result_data.keys()))
        writer.writeheader()
        writer.writerow(result_data)
        return buf.getvalue().rstrip()
    else:
        lines = [f"{'='*50}"]
        for key, val in result_data.items():
            label = key.replace("_", " ").title().ljust(16)
            if isinstance(val, float):
                lines.append(f"{label}: {val:.6e}")
            else:
                lines.append(f"{label}: {val}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)


@click.group()
@click.version_option(__version__, prog_name="tn-check")
def main() -> None:
    """TN-Check: Certified CSL model checking on tensor-train CME states."""


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
@main.command()
@click.argument("network_file", type=click.Path(exists=True))
@click.argument("property", type=str, required=False, default=None)
@click.option("--max-bond-dim", "-chi", type=int, default=200, show_default=True,
              help="Maximum bond dimension for TT compression.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write result as JSON to this file.")
@click.option("--output-format", "-f", type=click.Choice(["text", "json", "csv"]),
              default="text", show_default=True,
              help="Output format for results.")
@click.option("--steady-state", nargs=2, type=str, default=None,
              metavar="SPECIES THRESHOLD",
              help="Shorthand: check steady-state probability that SPECIES >= THRESHOLD.")
@click.option("--extinction", type=str, default=None, metavar="SPECIES",
              help="Shorthand: check probability that SPECIES reaches 0.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def check(network_file: str, property: str | None, max_bond_dim: int,
          output: str | None, output_format: str,
          steady_state: tuple[str, ...] | None, extinction: str | None,
          verbose: bool) -> None:
    """Check a CSL property against a reaction network.

    NETWORK_FILE is a YAML/JSON/SBML reaction network definition.
    PROPERTY is a CSL formula string, e.g. 'P>=0.9 [true U<=10.0 X_1 >= 5]'.

    Instead of a full CSL formula, you can use shorthand flags:

    \b
      --steady-state mRNA 5    →  S>=0.5 [X_i >= 5]
      --extinction   Protein   →  P>=0.5 [true U<=100 X_i <= 0]

    Species in the property are referenced by index (X_0, X_1, …) matching
    the order they appear in the network file.
    """
    _setup_logging(verbose)
    from tn_check.network_loader import load_network
    from tn_check.cme import compile_network_to_mpo, deterministic_initial_state
    from tn_check.checker.csl_ast import parse_csl
    from tn_check.checker.model_checker import CSLModelChecker

    net = load_network(network_file)

    formula_str = _resolve_property(property, steady_state, extinction, net)

    click.echo(f"Loaded network '{net.name}' ({net.num_species} species, {net.num_reactions} reactions)")

    formula = parse_csl(formula_str)
    click.echo(f"CSL property: {formula}")

    click.echo("Compiling CME generator …")
    t0 = time.time()
    mpo = compile_network_to_mpo(net, max_bond_dim=max_bond_dim)
    init = deterministic_initial_state(net)
    checker = CSLModelChecker(mpo, physical_dims=tuple(net.physical_dims))
    result = checker.check(formula, init, max_bond_dim=max_bond_dim)
    elapsed = time.time() - t0

    out_data = {
        "network": net.name,
        "property": str(formula),
        "verdict": result.verdict.value,
        "probability_lower": result.probability_lower,
        "probability_upper": result.probability_upper,
        "total_error": result.total_error,
        "wall_time_s": round(elapsed, 3),
    }

    click.echo(_format_result(out_data, output_format))

    if output:
        Path(output).write_text(json.dumps(out_data, indent=2))
        click.echo(f"Result written to {output}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
@main.command()
@click.argument("network_file", type=click.Path(exists=True))
@click.argument("property", type=str, required=False, default=None)
@click.option("--max-bond-dim", "-chi", type=int, default=200, show_default=True,
              help="Maximum bond dimension for TT compression.")
@click.option("--certificate", "-c", type=click.Path(), default=None,
              help="Write verification certificate JSON to this file.")
@click.option("--output-format", "-f", type=click.Choice(["text", "json", "csv"]),
              default="text", show_default=True,
              help="Output format for results.")
@click.option("--steady-state", nargs=2, type=str, default=None,
              metavar="SPECIES THRESHOLD",
              help="Shorthand: check steady-state probability that SPECIES >= THRESHOLD.")
@click.option("--extinction", type=str, default=None, metavar="SPECIES",
              help="Shorthand: check probability that SPECIES reaches 0.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def verify(network_file: str, property: str | None, max_bond_dim: int,
           certificate: str | None, output_format: str,
           steady_state: tuple[str, ...] | None, extinction: str | None,
           verbose: bool) -> None:
    """Check a CSL property and produce a formal verification certificate.

    Like 'check', but also runs the independent CertificateVerifier on
    the verification trace and writes a certificate JSON.

    Supports --steady-state and --extinction shorthands (see 'check --help').
    """
    _setup_logging(verbose)
    from tn_check.network_loader import load_network
    from tn_check.cme import compile_network_to_mpo, deterministic_initial_state
    from tn_check.checker.csl_ast import parse_csl
    from tn_check.checker.model_checker import CSLModelChecker
    from tn_check.verifier import CertificateVerifier

    net = load_network(network_file)

    formula_str = _resolve_property(property, steady_state, extinction, net)

    click.echo(f"Loaded network '{net.name}' ({net.num_species} species, {net.num_reactions} reactions)")

    formula = parse_csl(formula_str)
    click.echo(f"CSL property: {formula}")

    click.echo("Compiling CME generator …")
    t0 = time.time()
    mpo = compile_network_to_mpo(net, max_bond_dim=max_bond_dim)
    init = deterministic_initial_state(net)
    checker = CSLModelChecker(mpo, physical_dims=tuple(net.physical_dims))
    result = checker.check(formula, init, max_bond_dim=max_bond_dim)
    elapsed = time.time() - t0

    # Finalize and verify the trace
    trace = checker.trace
    trace.model_name = net.name
    trace.time_horizon = 0.0
    trace.max_bond_dim = max_bond_dim
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    out_data = {
        "network": net.name,
        "property": str(formula),
        "verdict": result.verdict.value,
        "probability_lower": result.probability_lower,
        "probability_upper": result.probability_upper,
        "total_error": result.total_error,
        "wall_time_s": round(elapsed, 3),
        "certificate": "SOUND" if report.overall_sound else "FAILED",
        "checks_passed": report.num_passed,
        "checks_total": report.num_checks,
    }

    click.echo(_format_result(out_data, output_format))

    cert_path = certificate or "certificate.json"
    trace.to_json(cert_path)
    click.echo(f"Certificate written to {cert_path}")

    if not report.overall_sound:
        click.echo("\nFailed checks:")
        for c in report.checks:
            if not c.passed:
                click.echo(f"  ✗ {c.name}: {c.message}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
@main.command()
@click.argument("network_file", type=click.Path(exists=True))
def info(network_file: str) -> None:
    """Display statistics about a reaction network.

    Shows species count, reaction count, state space size estimate,
    and per-species/per-reaction details.
    """
    from tn_check.network_loader import load_network
    import math

    net = load_network(network_file)

    dims = net.physical_dims
    state_space = math.prod(dims) if dims else 0

    click.echo(f"Network: {net.name}")
    if net.description:
        click.echo(f"Description: {net.description}")
    click.echo(f"\nSpecies ({net.num_species}):")
    for sp in net.species:
        click.echo(f"  {sp.name:12s}  max_copy={sp.max_copy_number:4d}  init={sp.initial_count}")
    click.echo(f"\nReactions ({net.num_reactions}):")
    for rxn in net.reactions:
        click.echo(f"  {rxn.name:20s}  kinetics={rxn.kinetics_type.value}")
    click.echo(f"\nState space size: {state_space:,} ({net.num_species} dimensions)")
    if state_space > 0:
        click.echo(f"Log10 state space: {math.log10(state_space):.1f}")


# ---------------------------------------------------------------------------
# init-network
# ---------------------------------------------------------------------------
@main.command("init-network")
@click.argument("output_file", type=click.Path(), default="network.yaml")
def init_network(output_file: str) -> None:
    """Generate a starter YAML reaction network template.

    Creates a commented template file with examples of mass-action,
    Hill, and Michaelis-Menten kinetics.
    """
    from tn_check.network_loader import STARTER_TEMPLATE

    path = Path(output_file)
    if path.exists():
        click.confirm(f"'{output_file}' already exists. Overwrite?", abort=True)
    path.write_text(STARTER_TEMPLATE)
    click.echo(f"Starter template written to {output_file}")


# ---------------------------------------------------------------------------
# check-network
# ---------------------------------------------------------------------------
@main.command("check-network")
@click.argument("network_file", type=click.Path(exists=True))
def check_network(network_file: str) -> None:
    """Validate a network definition file without running model checking.

    Checks for structural errors: missing fields, unknown species references,
    invalid kinetics parameters. Supports YAML, JSON, and SBML formats.
    """
    import yaml as _yaml

    path = Path(network_file)

    # SBML files: just try to load, no dict-level validation
    if path.suffix in (".xml", ".sbml"):
        from tn_check.network_loader import load_network
        try:
            net = load_network(network_file)
            click.echo(f"✓ SBML network '{net.name}' is valid ({net.num_species} species, {net.num_reactions} reactions)")
        except Exception as exc:
            click.echo(f"Validation FAILED: {exc}", err=True)
            sys.exit(1)
        return

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = _yaml.safe_load(text)
    else:
        data = json.loads(text)

    from tn_check.network_loader import validate_network_dict
    errors = validate_network_dict(data)

    if errors:
        click.echo(f"Validation FAILED ({len(errors)} error(s)):", err=True)
        for e in errors:
            click.echo(f"  ✗ {e}", err=True)
        sys.exit(1)
    else:
        # Also try building to catch runtime errors
        from tn_check.network_loader import load_network
        try:
            net = load_network(network_file)
            click.echo(f"✓ Network '{net.name}' is valid ({net.num_species} species, {net.num_reactions} reactions)")
        except Exception as exc:
            click.echo(f"Validation FAILED: {exc}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
