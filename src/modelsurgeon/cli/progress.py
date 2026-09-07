"""Non-interactive progress, pause, resume, and diagnostics CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.progress import (
    ProgressError,
    ProgressOutcome,
    ProgressStage,
    ProgressStatus,
    ProgressStore,
)

progress_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
_DEFAULT_ROOT = Path("artifacts/progress")


def _store(root: Path) -> ProgressStore:
    return ProgressStore(root)


def _parse_stages(value: str) -> tuple[ProgressStage, ...]:
    try:
        stages = tuple(ProgressStage(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ProgressError(f"unknown progress stage in {value!r}") from error
    if not stages:
        raise ProgressError("at least one progress stage is required")
    return stages


def _run(operation: Callable[[], object], output_json: bool) -> None:
    try:
        value = operation()
    except (ProgressError, OSError, ValueError) as error:
        if output_json:
            typer.echo(
                json.dumps(
                    {"record_type": "error", "category": "progress", "message": str(error)},
                    sort_keys=True,
                ),
                err=True,
            )
        else:
            typer.echo(f"progress error: {error}", err=True)
        raise typer.Exit(2) from error
    record = value.to_record() if hasattr(value, "to_record") else value
    separators = (",", ":") if output_json else (", ", ": ")
    typer.echo(json.dumps(record, sort_keys=True, separators=separators))


@progress_app.command("init")
def init_command(
    campaign_id: Annotated[str, typer.Argument(help="Stable campaign identifier")],
    stages: Annotated[str, typer.Option(help="Comma-separated ordered stages")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    source_artifact_digest: Annotated[str | None, typer.Option("--source-artifact")] = None,
    total: Annotated[int | None, typer.Option("--total", min=1)] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Create a crash-consistent campaign state."""

    _run(
        lambda: _store(root).start(
            campaign_id,
            _parse_stages(stages),
            source_artifact_digest=source_artifact_digest,
            total=total,
        ),
        output_json,
    )


@progress_app.command("event")
def event_command(
    campaign_id: Annotated[str, typer.Argument()],
    stage: Annotated[str | None, typer.Option("--stage")] = None,
    status: Annotated[str, typer.Option("--status")] = "running",
    completed: Annotated[int, typer.Option("--completed", min=0)] = 0,
    total: Annotated[int | None, typer.Option("--total", min=1)] = None,
    elapsed_seconds: Annotated[float, typer.Option("--elapsed-seconds", min=0)] = 0.0,
    next_work: Annotated[str, typer.Option("--next-work")] = "",
    detail: Annotated[str | None, typer.Option("--detail")] = None,
    outcome: Annotated[str | None, typer.Option("--outcome")] = None,
    accepted_artifact_digest: Annotated[str | None, typer.Option("--accepted-artifact")] = None,
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Append one ordered progress event."""

    def operation() -> object:
        store = _store(root)
        parsed_stage = None if stage is None else ProgressStage(stage)
        parsed_status = ProgressStatus(status)
        parsed_outcome = None if outcome is None else ProgressOutcome(outcome)
        return store.record(
            campaign_id,
            event_type="stage_update",
            stage=parsed_stage,
            status=parsed_status,
            completed=completed,
            total=total,
            elapsed_seconds=elapsed_seconds,
            next_work=tuple(item.strip() for item in next_work.split(",") if item.strip()),
            detail=detail,
            outcome=parsed_outcome,
            accepted_artifact_digest=accepted_artifact_digest,
        )

    _run(operation, output_json)


@progress_app.command("show")
def show_command(
    campaign_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show current progress as stable JSON or readable text."""

    def operation() -> object:
        snapshot = _store(root).snapshot(campaign_id)
        if output_json:
            return snapshot
        typer.echo(
            f"{snapshot.status.value} {snapshot.campaign_id} "
            f"completed={snapshot.completed}/{snapshot.total or '?'} "
            f"next={','.join(snapshot.next_work) or 'none'}"
        )
        return {}

    _run(operation, output_json)


@progress_app.command("pause")
def pause_command(
    campaign_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Pause without changing the accepted artifact identity."""

    _run(lambda: _store(root).pause(campaign_id), output_json)


@progress_app.command("cancel")
def cancel_command(
    campaign_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Cancel and retain a recoverable last accepted state."""

    _run(lambda: _store(root).cancel(campaign_id), output_json)


@progress_app.command("resume")
def resume_command(
    campaign_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume from the last atomically accepted event."""

    _run(lambda: _store(root).resume(campaign_id), output_json)


@progress_app.command("diagnostics")
def diagnostics_command(
    campaign_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Emit or persist a redacted diagnostic bundle."""

    def operation() -> object:
        store = _store(root)
        if output is not None:
            store.write_diagnostics(campaign_id, output)
        return store.diagnostics(campaign_id)

    _run(operation, output_json)


__all__ = ["progress_app"]
