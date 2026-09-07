"""Offline static evidence explorer CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.explain.explorer import (
    ExplorerCell,
    ExplorerCellStatus,
    ExplorerError,
    generate_explorer,
    write_explorer,
)


def _load_cells(path: Path) -> tuple[str, str, str, str, tuple[ExplorerCell, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExplorerError("explorer input is unreadable or invalid JSON") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("cells"), list):
        raise ExplorerError("explorer input requires a cells list")
    cells: list[ExplorerCell] = []
    for item in raw["cells"]:
        if not isinstance(item, Mapping):
            raise ExplorerError("explorer cells must be objects")
        metrics_raw = item.get("metrics", {})
        lineage_raw = item.get("lineage", [])
        diff_raw = item.get("architecture_diff", {})
        if (
            not isinstance(metrics_raw, Mapping)
            or not all(isinstance(key, str) for key in metrics_raw)
            or not isinstance(lineage_raw, list)
            or not all(isinstance(value, str) for value in lineage_raw)
            or not isinstance(diff_raw, Mapping)
        ):
            raise ExplorerError("explorer cell fields are malformed")
        try:
            status = ExplorerCellStatus(str(item["status"]))
        except (KeyError, ValueError) as error:
            raise ExplorerError("explorer cell status is invalid") from error
        cells.append(
            ExplorerCell(
                str(item["cell_id"]),
                status,
                str(item["source_id"]),
                tuple(sorted((key, value) for key, value in metrics_raw.items())),
                tuple(sorted(lineage_raw)),
                dict(diff_raw),
                None if item.get("failure_reason") is None else str(item["failure_reason"]),
            )
        )
    return (
        str(raw.get("title", "ModelSurgeon evidence explorer")),
        str(raw.get("source_report_id", "canonical-evidence")),
        str(raw.get("cost_metric", "cost")),
        str(raw.get("quality_metric", "quality")),
        tuple(cells),
    )


def explorer_command(
    input_path: Annotated[Path, typer.Argument(help="Canonical evidence cells JSON")],
    output: Annotated[Path, typer.Option("--output", help="Static HTML output path")],
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit artifact metadata JSON")
    ] = False,
) -> None:
    """Generate a self-contained offline benchmark and lineage explorer."""

    try:
        title, source_report_id, cost_metric, quality_metric, cells = _load_cells(input_path)
        artifact = generate_explorer(
            cells,
            title=title,
            source_report_id=source_report_id,
            cost_metric=cost_metric,
            quality_metric=quality_metric,
        )
        write_explorer(output, artifact)
    except (ExplorerError, OSError, ValueError) as error:
        typer.echo(f"explorer error: {error}", err=True)
        raise typer.Exit(2) from error
    if output_json:
        typer.echo(json.dumps(artifact.to_record(), sort_keys=True, separators=(",", ":")))
    else:
        typer.echo(f"{artifact.artifact_id} cells={artifact.cell_count} output={output}")


__all__ = ["explorer_command"]
