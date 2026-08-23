"""ModelSurgeon command-line entry point."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from modelsurgeon.adapters.huggingface import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.graph import walk_named_modules
from modelsurgeon.logging import LogFormat, configure_logging

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def configure_cli_logging(
    log_level: Annotated[
        str,
        typer.Option(help="Logging threshold"),
    ] = "INFO",
    log_format: Annotated[
        LogFormat,
        typer.Option(help="Human-readable or JSON structured logs"),
    ] = LogFormat.HUMAN,
) -> None:
    """Configure logging only after the CLI is invoked."""
    configure_logging(level=log_level, output_format=log_format)


@app.command()
def inspect(
    model: Annotated[str, typer.Argument(help="Hugging Face model ID or local path")],
    revision: Annotated[str | None, typer.Option(help="Immutable model revision")] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Allow model repository code; disabled by default"),
    ] = False,
    device_map: Annotated[
        str,
        typer.Option(help="Placement strategy; defaults to CPU"),
    ] = "cpu",
    dtype: Annotated[
        HuggingFaceDType,
        typer.Option(help="Requested model compute dtype"),
    ] = HuggingFaceDType.AUTO,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit newline-delimited JSON records"),
    ] = False,
) -> None:
    """Load and enumerate a Hugging Face causal language model."""
    loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            dtype=dtype,
        )
    )
    for record in walk_named_modules(loaded.model):
        payload = {
            "component_id": str(record.component_id),
            "module_type": record.module_type,
            "parameter_count": record.parameter_count,
        }
        if output_json:
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            typer.echo(
                f"{payload['component_id']:<64} "
                f"{payload['module_type']:<24} {payload['parameter_count'] or 0:>12}"
            )


if __name__ == "__main__":  # pragma: no cover
    app()
