"""Unified optimize planning and first-party execution command."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.adapters import ModelFamily, ModelFormat
from modelsurgeon.config_io import ConfigurationFileError, load_settings
from modelsurgeon.experiments import (
    ApprovalReuse,
    EvidenceKeyRecord,
    EvidenceKeyStatus,
    OptimizationPackageError,
    write_reproducibility_package,
)
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.optimization import (
    OptimizePlanError,
    build_optimize_plan,
    write_optimize_plan,
)
from modelsurgeon.optimization_orchestrator import (
    OptimizeOrchestrator,
    OptimizeOrchestratorError,
    load_optimize_runtime,
)
from modelsurgeon.provider_kind import ProviderKind


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
    model_format: Annotated[
        ModelFormat | None,
        typer.Option(
            "--model-format",
            help="Source model container: huggingface, safetensors, or gguf",
        ),
    ] = None,
    model_family: Annotated[
        ModelFamily | None,
        typer.Option(
            "--model-family",
            help="Explicit architecture family for ambiguous GGUF metadata",
        ),
    ] = None,
    calibration_text: Annotated[
        Path | None,
        typer.Option(
            "--calibration-text",
            help="Local UTF-8 calibration text for first-party Hugging Face execution",
        ),
    ] = None,
    runtime_cli: Annotated[
        Path | None,
        typer.Option("--runtime-cli", help="Pinned llama-cli executable for GGUF execution"),
    ] = None,
    runtime_perplexity: Annotated[
        Path | None,
        typer.Option(
            "--runtime-perplexity",
            help="Pinned llama-perplexity executable for GGUF quality measurement",
        ),
    ] = None,
    runtime_bench: Annotated[
        Path | None,
        typer.Option(
            "--runtime-bench",
            help="Pinned llama-bench executable for GGUF deployment measurement",
        ),
    ] = None,
    runtime_revision: Annotated[
        str | None,
        typer.Option("--runtime-revision", help="Expected llama.cpp commit for GGUF tools"),
    ] = None,
    provider: Annotated[
        ProviderKind | None,
        typer.Option(
            "--provider",
            "--provider-kind",
            help="Optional text-model provider kind; none keeps the direct no-LLM path",
        ),
    ] = None,
    provider_id: Annotated[
        str | None,
        typer.Option("--provider-id", help="Configured provider identity"),
    ] = None,
    provider_model: Annotated[
        str | None,
        typer.Option("--provider-model", help="Configured text-model identity"),
    ] = None,
    provider_revision: Annotated[
        str | None,
        typer.Option("--provider-revision", help="Immutable text-model revision"),
    ] = None,
    provider_model_path: Annotated[
        Path | None,
        typer.Option("--provider-model-path", help="Local GGUF text-model path"),
    ] = None,
    provider_runtime_revision: Annotated[
        str | None,
        typer.Option("--provider-runtime-revision", help="Local provider runtime revision"),
    ] = None,
    provider_endpoint: Annotated[
        str | None,
        typer.Option("--provider-endpoint", help="Absolute compatible-provider endpoint"),
    ] = None,
    provider_api_key_env: Annotated[
        str | None,
        typer.Option(
            "--provider-api-key-env",
            help="Environment variable containing a provider key",
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Explicitly disable conversational provider use"),
    ] = False,
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
            help=(
                "Execute the bounded workflow with the first-party Hugging Face runtime; "
                "--runtime may select an external adapter"
            ),
        ),
    ] = False,
    state: Annotated[
        Path | None,
        typer.Option("--state", help="Atomic JSON workflow state used for resume"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume the matching incomplete workflow"),
    ] = False,
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Record approval code; may be repeated"),
    ] = None,
    approval_expires_at: Annotated[
        str | None,
        typer.Option("--approval-expires-at", help="Expiry for newly recorded approvals"),
    ] = None,
    approval_reuse: Annotated[
        list[str] | None,
        typer.Option(
            "--approval-reuse",
            help="Approval reuse as code=one_time or code=reusable; may be repeated",
        ),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="Set approved runtime override as name=value"),
    ] = None,
    runtime: Annotated[
        str | None,
        typer.Option("--runtime", help="Trusted runtime factory as module:factory"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Persist the canonical plan JSON without overwriting"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete plan as JSON"),
    ] = False,
    package: Annotated[
        Path | None,
        typer.Option(
            "--package",
            help="Write a signed final reproducibility package after execution",
        ),
    ] = None,
    package_key_id: Annotated[
        str | None,
        typer.Option("--package-key-id", help="Signing key ID for --package"),
    ] = None,
    package_key_env: Annotated[
        str | None,
        typer.Option(
            "--package-key-env",
            help="Environment variable containing the package signing key",
        ),
    ] = None,
) -> None:
    """Plan or execute one bounded, resumable optimization workflow."""

    overrides: dict[str, object] = {}
    if model is not None:
        overrides["model.path"] = model
    if revision is not None:
        overrides["model.revision"] = revision
    if model_format is not None:
        overrides["model.format"] = model_format.value
    if model_family is not None:
        overrides["model.family"] = model_family.value
    if calibration_text is not None:
        overrides["calibration.dataset"] = str(calibration_text)
    if runtime_cli is not None:
        overrides["runtime.llama_cli"] = runtime_cli
    if runtime_perplexity is not None:
        overrides["runtime.llama_perplexity"] = runtime_perplexity
    if runtime_bench is not None:
        overrides["runtime.llama_bench"] = runtime_bench
    if runtime_revision is not None:
        overrides["runtime.expected_revision"] = runtime_revision
    if no_llm:
        overrides.update(
            {
                "provider.kind": ProviderKind.NONE.value,
                "provider.provider_id": "none",
                "provider.model_id": "none",
                "provider.model_revision": "none",
                "provider.model_path": None,
                "provider.runtime_revision": None,
                "provider.endpoint": None,
                "provider.api_key_env": None,
            }
        )
    else:
        provider_overrides = {
            "provider.kind": provider,
            "provider.provider_id": provider_id,
            "provider.model_id": provider_model,
            "provider.model_revision": provider_revision,
            "provider.model_path": provider_model_path,
            "provider.runtime_revision": provider_runtime_revision,
            "provider.endpoint": provider_endpoint,
            "provider.api_key_env": provider_api_key_env,
        }
        overrides.update(
            {key: value for key, value in provider_overrides.items() if value is not None}
        )
    try:
        if resume and not execute:
            raise OptimizePlanError("--resume requires --execute")
        if (approve or override or approval_reuse or approval_expires_at) and not execute:
            raise OptimizePlanError("approval options and --override require --execute")
        settings = load_settings(config, cli_overrides=overrides)
        plan = build_optimize_plan(
            settings,
            preset=preset,
            hardware_profile=hardware_profile,
            quality_profile=quality_profile,
            dry_run=not execute,
        )
        run_record = None
        if execute:
            if state is None:
                raise OptimizePlanError("--state is required with --execute")
            override_values: dict[str, str] = {}
            for item in override or []:
                name, separator, value = item.partition("=")
                if not separator or not name.strip() or not value.strip():
                    raise OptimizePlanError("--override values must use name=value syntax")
                override_values[name.strip()] = value
            reuse_values: dict[str, str] = {}
            for item in approval_reuse or []:
                name, separator, value = item.partition("=")
                if not separator or not name.strip():
                    raise OptimizePlanError(
                        "--approval-reuse values must use code=one_time or code=reusable"
                    )
                try:
                    reuse_values[name.strip()] = ApprovalReuse(value.strip()).value
                except ValueError as error:
                    raise OptimizePlanError(
                        "--approval-reuse policy must be one_time or reusable"
                    ) from error
            selected_runtime = (
                build_first_party_optimize_runtime(plan)
                if runtime is None
                else load_optimize_runtime(runtime)
            )
            run_record = OptimizeOrchestrator(plan, state).run(
                selected_runtime,
                resume=resume,
                approvals=tuple(approve or ()),
                overrides=override_values,
                approval_expires_at=approval_expires_at,
                approval_reuse=reuse_values,
            )
            if package is not None:
                if not package_key_id or not package_key_env:
                    raise OptimizePlanError(
                        "--package requires --package-key-id and --package-key-env"
                    )
                key_material_text = os.environ.get(package_key_env)
                if not key_material_text:
                    raise OptimizePlanError(
                        "package signing key environment variable is unavailable: "
                        f"{package_key_env}"
                    )
                key_material = key_material_text.encode("utf-8")
                signing_key = EvidenceKeyRecord(
                    package_key_id,
                    hashlib.sha256(key_material).hexdigest(),
                    EvidenceKeyStatus.ACTIVE,
                )
                stage_records = run_record.to_record()["stages"]
                if not isinstance(stage_records, list):
                    raise OptimizePlanError("run stages are not a JSON array")
                stage_evidence = [
                    item["result"]
                    for item in stage_records
                    if isinstance(item, Mapping) and item.get("result") is not None
                ]
                write_reproducibility_package(
                    package,
                    plan=plan,
                    run=run_record,
                    decision_evidence=stage_evidence,
                    signing_key=signing_key,
                    signing_key_material=key_material,
                )
        if output is not None:
            if run_record is None:
                write_optimize_plan(output, plan, allow_overwrite=settings.safety.allow_overwrite)
            else:
                if output.exists() and not settings.safety.allow_overwrite:
                    raise OptimizePlanError(
                        f"refusing to overwrite existing run artifact: {output}"
                    )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    run_record.canonical_json() + "\n", encoding="utf-8", newline="\n"
                )
    except (
        ConfigurationFileError,
        OptimizePlanError,
        OptimizeOrchestratorError,
        OptimizationPackageError,
        OSError,
        ValueError,
    ) as error:
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

    if run_record is not None:
        if output_json:
            typer.echo(run_record.canonical_json())
        else:
            typer.echo(
                f"{run_record.outcome.value} {run_record.run_id} "
                f"status={run_record.status.value} "
                f"cursor={run_record.cursor}/{len(run_record.stages)}"
            )
            for reason in run_record.reasons:
                typer.echo(f"reason: {reason}")
    elif output_json:
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
