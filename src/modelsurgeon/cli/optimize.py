"""Unified, read-only optimize planning command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.config_io import ConfigurationFileError, load_settings
from modelsurgeon.optimization import (
    OptimizePlanError,
    build_optimize_plan,
    write_optimize_plan,
)


def optimize_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML or TOML settings file"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Source model path or immutable model identifier"),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Immutable source model revision"),
    ] = None,
    preset: Annotated[
        str,
        typer.Option(help="Bounded optimization preset: fast, balanced, or quality"),
    ] = "balanced",
    hardware_profile: Annotated[
        str,
        typer.Option("--hardware-profile", help="Hardware envelope identifier"),
    ] = "cpu-small",
    quality_profile: Annotated[
        str | None,
        typer.Option("--quality-profile", help="Quality target: fast, balanced, or quality"),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Print an approval-gated execution plan; never mutates in this release",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Persist the canonical plan JSON without overwriting"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete plan as JSON"),
    ] = False,
) -> None:
    """Resolve configuration and produce one deterministic optimization plan."""

    overrides: dict[str, object] = {}
    if model is not None:
        overrides["model.path"] = model
    if revision is not None:
        overrides["model.revision"] = revision
    try:
        settings = load_settings(config, cli_overrides=overrides)
        plan = build_optimize_plan(
            settings,
            preset=preset,
            hardware_profile=hardware_profile,
            quality_profile=quality_profile,
            dry_run=not execute,
        )
        if output is not None:
            write_optimize_plan(output, plan, allow_overwrite=settings.safety.allow_overwrite)
    except (ConfigurationFileError, OptimizePlanError, OSError, ValueError) as error:
        if output_json:
            typer.echo(
                json.dumps(
                    {"record_type": "error", "category": "optimize", "message": str(error)},
                    sort_keys=True,
                ),
                err=True,
            )
        else:
            typer.echo(f"optimize error: {error}", err=True)
        raise typer.Exit(2) from error

    if output_json:
        typer.echo(plan.canonical_json())
    else:
        typer.echo(
            f"{plan.outcome.value} {plan.plan_id} preset={plan.preset.value} "
            f"hardware={plan.hardware_profile.profile_id} quality={plan.quality_profile.profile_id}"
        )
        typer.echo(f"evaluations={plan.budget.evaluations} repair_steps={plan.budget.repair_steps}")
        if plan.uncertainties:
            for uncertainty in plan.uncertainties:
                typer.echo(f"uncertainty: {uncertainty}")
