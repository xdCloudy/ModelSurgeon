"""Local-first setup and first-run diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.config import ProviderConfig
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.setup import (
    DEFAULT_MIN_FREE_BYTES,
    SetupReport,
    SetupRequest,
    default_data_root,
    diagnose_setup,
    initialize_setup,
)

setup_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _provider(
    kind: ProviderKind,
    provider_id: str | None,
    model_id: str | None,
    revision: str | None,
    model_path: Path | None,
    runtime_revision: str | None,
    endpoint: str | None,
    api_key_env: str | None,
) -> ProviderConfig:
    if kind is ProviderKind.NONE:
        return ProviderConfig()
    return ProviderConfig(
        kind=kind,
        provider_id=provider_id or "",
        model_id=model_id or "",
        model_revision=revision or "",
        model_path=model_path,
        runtime_revision=runtime_revision,
        endpoint=endpoint,
        api_key_env=api_key_env,
    )


def _request(
    *,
    data_dir: Path | None,
    fixture: Path | None,
    offline: bool,
    min_free_gb: float,
    provider: ProviderKind,
    provider_id: str | None,
    provider_model: str | None,
    provider_revision: str | None,
    provider_model_path: Path | None,
    provider_runtime_revision: str | None,
    provider_endpoint: str | None,
    provider_api_key_env: str | None,
    create_missing: bool,
) -> SetupRequest:
    if min_free_gb <= 0:
        raise ValueError("--min-free-gb must be positive")
    return SetupRequest(
        data_root=(data_dir or default_data_root()).expanduser(),
        fixture=fixture.expanduser() if fixture is not None else None,
        offline=offline,
        min_free_bytes=int(min_free_gb * (1 << 30)),
        provider=_provider(
            provider,
            provider_id,
            provider_model,
            provider_revision,
            provider_model_path,
            provider_runtime_revision,
            provider_endpoint,
            provider_api_key_env,
        ),
        create_missing=create_missing,
    )


def _emit(report: SetupReport, *, output_json: bool) -> None:
    record = report.to_record()
    if output_json:
        typer.echo(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return
    typer.echo(
        f"setup {record['status']} outcome={record['outcome']} "
        f"data_root={record['data_root']}"
    )
    for check in report.checks:
        typer.echo(
            f"{check.outcome.value} {check.category}:{check.code} — {check.message}"
        )
    if record["config_path"] is not None:
        typer.echo(f"config: {record['config_path']}")


@setup_app.command("diagnostics")
def diagnostics_command(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir", help="User-owned location for large mutable ModelSurgeon data"
        ),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture", help="Existing local supported-fixture manifest; never downloaded"
        ),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline", help="Require local fixture/no-LLM operation and make no network request"
        ),
    ] = False,
    min_free_gb: Annotated[
        float,
        typer.Option(
            "--min-free-gb",
            help="Declared free-space budget for setup metadata and fixture preparation",
        ),
    ] = DEFAULT_MIN_FREE_BYTES / (1 << 30),
    provider: Annotated[
        ProviderKind,
        typer.Option("--provider", help="Optional conversational provider kind; default is none"),
    ] = ProviderKind.NONE,
    provider_id: Annotated[str | None, typer.Option("--provider-id")] = None,
    provider_model: Annotated[str | None, typer.Option("--provider-model")] = None,
    provider_revision: Annotated[str | None, typer.Option("--provider-revision")] = None,
    provider_model_path: Annotated[Path | None, typer.Option("--provider-model-path")] = None,
    provider_runtime_revision: Annotated[
        str | None, typer.Option("--provider-runtime-revision")
    ] = None,
    provider_endpoint: Annotated[str | None, typer.Option("--provider-endpoint")] = None,
    provider_api_key_env: Annotated[str | None, typer.Option("--provider-api-key-env")] = None,
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit one canonical JSON report")
    ] = False,
) -> None:
    """Check local first-run readiness without creating files or contacting providers."""
    try:
        report = diagnose_setup(
            _request(
                data_dir=data_dir,
                fixture=fixture,
                offline=offline,
                min_free_gb=min_free_gb,
                provider=provider,
                provider_id=provider_id,
                provider_model=provider_model,
                provider_revision=provider_revision,
                provider_model_path=provider_model_path,
                provider_runtime_revision=provider_runtime_revision,
                provider_endpoint=provider_endpoint,
                provider_api_key_env=provider_api_key_env,
                create_missing=False,
            )
        )
    except (ValueError, OSError) as error:
        typer.echo(f"setup configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    _emit(report, output_json=output_json)
    if report.status.value == "failed":
        raise typer.Exit(2)


@setup_app.command("init")
def init_command(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir", help="User-owned location for large mutable ModelSurgeon data"
        ),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture", help="Existing local supported-fixture manifest; never downloaded"
        ),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline", help="Initialize without network access; requires a local fixture"
        ),
    ] = False,
    min_free_gb: Annotated[
        float,
        typer.Option(
            "--min-free-gb",
            help="Declared free-space budget for setup metadata and fixture preparation",
        ),
    ] = DEFAULT_MIN_FREE_BYTES / (1 << 30),
    provider: Annotated[
        ProviderKind,
        typer.Option("--provider", help="Optional conversational provider kind; default is none"),
    ] = ProviderKind.NONE,
    provider_id: Annotated[str | None, typer.Option("--provider-id")] = None,
    provider_model: Annotated[str | None, typer.Option("--provider-model")] = None,
    provider_revision: Annotated[str | None, typer.Option("--provider-revision")] = None,
    provider_model_path: Annotated[Path | None, typer.Option("--provider-model-path")] = None,
    provider_runtime_revision: Annotated[
        str | None, typer.Option("--provider-runtime-revision")
    ] = None,
    provider_endpoint: Annotated[str | None, typer.Option("--provider-endpoint")] = None,
    provider_api_key_env: Annotated[str | None, typer.Option("--provider-api-key-env")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the first-run manifest and README")
    ] = False,
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit one canonical JSON report")
    ] = False,
) -> None:
    """Create the local data layout and a secret-free first-run manifest."""
    try:
        report = initialize_setup(
            _request(
                data_dir=data_dir,
                fixture=fixture,
                offline=offline,
                min_free_gb=min_free_gb,
                provider=provider,
                provider_id=provider_id,
                provider_model=provider_model,
                provider_revision=provider_revision,
                provider_model_path=provider_model_path,
                provider_runtime_revision=provider_runtime_revision,
                provider_endpoint=provider_endpoint,
                provider_api_key_env=provider_api_key_env,
                create_missing=True,
            ),
            force=force,
        )
    except (ValueError, OSError) as error:
        typer.echo(f"setup configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    _emit(report, output_json=output_json)
    if report.status.value == "failed":
        raise typer.Exit(2)


__all__ = ["setup_app"]
