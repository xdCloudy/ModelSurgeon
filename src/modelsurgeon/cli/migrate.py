"""CLI for explicit, fail-closed schema migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.migration import MigrationKind, MigrationRefusal, migrate_record
from modelsurgeon.utils.atomic_files import atomic_write_text


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MigrationRefusal(f"migration input is missing or invalid JSON: {path}") from error


def _write_json(path: Path, payload: object) -> None:
    try:
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            replace_existing=False,
        )
    except FileExistsError as error:
        raise MigrationRefusal(f"refusing to overwrite migration output: {path}") from error


def migrate_command(
    input_path: Annotated[Path, typer.Argument(help="Versioned JSON record to migrate")],
    output: Annotated[
        Path | None,
        typer.Option(help="Write the migrated record without overwriting an existing file"),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(help="Record family: config, campaign, or evidence; default is auto-detect"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the migration report as JSON"),
    ] = False,
) -> None:
    """Migrate a supported v2.0 record before direct execution or resume."""

    try:
        selected = None if kind is None else MigrationKind(kind)
        result = migrate_record(_read_json(input_path), kind=selected)
        if output is not None:
            _write_json(output, result.to_record())
    except (MigrationRefusal, OSError, ValueError) as error:
        typer.echo(f"migration refused: {error}", err=True)
        raise typer.Exit(2) from error
    if output_json:
        typer.echo(json.dumps(result.report(), sort_keys=True))
    else:
        target = "stdout" if output is None else str(output)
        typer.echo(
            f"migration {result.kind.value}: {result.source_schema_version} -> "
            f"{result.target_schema_version}; output={target}"
        )


__all__ = ["migrate_command"]
