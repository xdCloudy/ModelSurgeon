"""ModelSurgeon command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.adapters import ArchitectureDetectionError
from modelsurgeon.adapters.huggingface import (
    HuggingFaceDependencyError,
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    HuggingFaceModelError,
    HuggingFaceRevisionError,
)
from modelsurgeon.cli.experiment import (
    ExperimentCommandError,
    load_experiment_runtime,
    read_mutation_request,
    run_single_mutation_experiment,
    write_experiment_result,
)
from modelsurgeon.cli.inspection import inspect_huggingface_model
from modelsurgeon.cli.proof import first_surgeon_proof_command
from modelsurgeon.cli.proof_evidence import first_surgeon_evidence_command
from modelsurgeon.cli.proof_hf import first_surgeon_hf_proof_command
from modelsurgeon.cli.search import search_command
from modelsurgeon.cli.surgeon import predict_surgeon_command, train_surgeon_command
from modelsurgeon.logging import LogFormat, configure_logging

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _inspection_error(
    category: str,
    error: Exception,
    *,
    exit_code: int,
    output_json: bool,
) -> None:
    if output_json:
        typer.echo(
            json.dumps(
                {"record_type": "error", "category": category, "message": str(error)},
                sort_keys=True,
            ),
            err=True,
        )
    else:
        typer.echo(f"{category} error: {error}", err=True)
    raise typer.Exit(exit_code)


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
    try:
        inspection = inspect_huggingface_model(
            HuggingFaceLoadRequest(
                model=model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                dtype=dtype,
            )
        )
    except HuggingFaceDependencyError as exc:
        _inspection_error("dependency", exc, exit_code=3, output_json=output_json)
    except HuggingFaceRevisionError as exc:
        _inspection_error("revision", exc, exit_code=5, output_json=output_json)
    except HuggingFaceModelError as exc:
        _inspection_error("model", exc, exit_code=4, output_json=output_json)
    except (ArchitectureDetectionError, ValueError) as exc:
        _inspection_error("adapter", exc, exit_code=6, output_json=output_json)

    for payload in inspection.records():
        if output_json:
            typer.echo(json.dumps(payload, sort_keys=True))
        elif payload["record_type"] == "model":
            typer.echo(
                f"model {payload['source']}@{payload['resolved_revision']} "
                f"family={payload['family']} parameters={payload['parameter_count']}"
            )
        else:
            attributes = json.dumps(
                payload["attributes"], sort_keys=True, separators=(",", ":")
            )
            typer.echo(
                f"{payload['component_id']:<64} {payload['kind']:<24} {attributes}"
            )


@app.command()
def experiment(
    mutation: Annotated[
        Path,
        typer.Argument(help="Canonical MutationRequest JSON file"),
    ],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Adapter runtime factory as module:factory",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve and print the plan without mutation"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Save the structured result without overwriting"),
    ] = None,
    include_local_paths: Annotated[
        bool,
        typer.Option(
            "--include-local-paths",
            help="Include local source paths in provenance output",
        ),
    ] = False,
) -> None:
    """Resolve, evaluate, and roll back one transactional mutation experiment."""

    try:
        request = read_mutation_request(mutation)
        experiment_runtime = load_experiment_runtime(runtime)
        result = run_single_mutation_experiment(
            request,
            experiment_runtime,
            dry_run=dry_run,
        )
        if output is not None:
            write_experiment_result(
                output,
                result,
                redact_local_paths=not include_local_paths,
            )
        typer.echo(result.canonical_json(redact_local_paths=not include_local_paths))
    except KeyboardInterrupt:
        typer.echo("experiment interrupted; transaction rolled back", err=True)
        raise typer.Exit(130) from None
    except (ExperimentCommandError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"experiment error: {error}", err=True)
        raise typer.Exit(2) from error


app.command("first-surgeon-proof")(first_surgeon_proof_command)
app.command("first-surgeon-hf-proof")(first_surgeon_hf_proof_command)
app.command("first-surgeon-evidence")(first_surgeon_evidence_command)
app.command("train-surgeon")(train_surgeon_command)
app.command("predict-surgeon")(predict_surgeon_command)
app.command("search")(search_command)


if __name__ == "__main__":  # pragma: no cover
    app()
