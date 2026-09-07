"""Provider selection diagnostics for non-interactive callers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.config_io import ConfigurationFileError, load_settings
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.providers import provider_diagnostics

provider_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _overrides(
    *,
    provider: ProviderKind | None,
    provider_id: str | None,
    provider_model: str | None,
    provider_revision: str | None,
    provider_endpoint: str | None,
    provider_api_key_env: str | None,
    no_llm: bool,
) -> dict[str, object]:
    if no_llm:
        return {
            "provider.kind": ProviderKind.NONE.value,
            "provider.provider_id": "none",
            "provider.model_id": "none",
            "provider.model_revision": "none",
            "provider.endpoint": None,
            "provider.api_key_env": None,
        }
    values = {
        "provider.kind": provider,
        "provider.provider_id": provider_id,
        "provider.model_id": provider_model,
        "provider.model_revision": provider_revision,
        "provider.endpoint": provider_endpoint,
        "provider.api_key_env": provider_api_key_env,
    }
    return {key: value for key, value in values.items() if value is not None}


@provider_app.command("diagnostics")
def diagnostics_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML or TOML settings file"),
    ] = None,
    provider: Annotated[ProviderKind | None, typer.Option("--provider", "--provider-kind")] = None,
    provider_id: Annotated[str | None, typer.Option("--provider-id")] = None,
    provider_model: Annotated[str | None, typer.Option("--provider-model")] = None,
    provider_revision: Annotated[str | None, typer.Option("--provider-revision")] = None,
    provider_endpoint: Annotated[str | None, typer.Option("--provider-endpoint")] = None,
    provider_api_key_env: Annotated[str | None, typer.Option("--provider-api-key-env")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm")] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit one JSON diagnostic record"),
    ] = False,
) -> None:
    """Report provider availability without starting a provider or model."""

    try:
        settings = load_settings(
            config,
            cli_overrides=_overrides(
                provider=provider,
                provider_id=provider_id,
                provider_model=provider_model,
                provider_revision=provider_revision,
                provider_endpoint=provider_endpoint,
                provider_api_key_env=provider_api_key_env,
                no_llm=no_llm,
            ),
        )
    except (ConfigurationFileError, ValueError) as error:
        payload = {
            "record_type": "error",
            "category": "provider_configuration",
            "code": getattr(error, "code", "invalid_configuration"),
            "message": str(error),
        }
        if output_json:
            typer.echo(json.dumps(payload, sort_keys=True), err=True)
        else:
            typer.echo(f"provider configuration error [{payload['code']}]: {error}", err=True)
        raise typer.Exit(2) from error

    diagnostic = provider_diagnostics(settings)
    if output_json:
        typer.echo(diagnostic.canonical_json())
    else:
        typer.echo(
            f"{diagnostic.status.value} {diagnostic.code} "
            f"kind={diagnostic.kind.value} provider={diagnostic.provider_id} "
            f"model={diagnostic.model_id}"
        )
        typer.echo(diagnostic.message)


__all__ = ["provider_app"]
