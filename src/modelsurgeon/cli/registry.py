"""Local artifact registry CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.registry import LocalArtifactRegistry, RegistryError

registry_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
_DEFAULT_ROOT = Path("artifacts/registry")


def _registry(root: Path) -> LocalArtifactRegistry:
    return LocalArtifactRegistry(root)


def _emit(value: object, output_json: bool) -> None:
    if output_json:
        typer.echo(json.dumps(value, sort_keys=True, separators=(",", ":")))
    else:
        typer.echo(json.dumps(value, sort_keys=True, indent=2))


def _run(operation: Callable[[], object], output_json: bool) -> None:
    try:
        value = operation()
    except (RegistryError, OSError, ValueError) as error:
        if output_json:
            typer.echo(
                json.dumps(
                    {"record_type": "error", "category": "registry", "message": str(error)},
                    sort_keys=True,
                ),
                err=True,
            )
        else:
            typer.echo(f"registry error: {error}", err=True)
        raise typer.Exit(2) from error
    _emit(value.to_record() if hasattr(value, "to_record") else value, output_json)


@registry_app.command("list")
def list_command(
    root: Annotated[Path, typer.Option("--root", help="Local registry directory")] = _DEFAULT_ROOT,
    kind: Annotated[str | None, typer.Option("--kind", help="Filter by artifact kind")] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Filter by tag")] = None,
    page_size: Annotated[int, typer.Option("--page-size", min=1)] = 100,
    cursor: Annotated[str | None, typer.Option("--cursor", help="Digest cursor")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List visible local artifacts without network transfer."""

    _run(
        lambda: _registry(root).list(kind=kind, tag=tag, page_size=page_size, cursor=cursor),
        output_json,
    )


@registry_app.command("inspect")
def inspect_command(
    reference: Annotated[str, typer.Argument(help="Digest or alias")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect one catalog record."""

    _run(lambda: _registry(root).inspect(reference), output_json)


@registry_app.command("verify")
def verify_command(
    reference: Annotated[str, typer.Argument(help="Digest or alias")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify object bytes and catalog metadata offline."""

    _run(lambda: _registry(root).verify(reference), output_json)


@registry_app.command("tag")
def tag_command(
    reference: Annotated[str, typer.Argument(help="Digest or alias")],
    tag: Annotated[str, typer.Argument(help="Stable tag")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Attach a tag without changing object identity."""

    _run(lambda: _registry(root).tag(reference, tag), output_json)


@registry_app.command("export")
def export_command(
    reference: Annotated[str, typer.Argument(help="Digest or alias")],
    destination: Annotated[Path, typer.Argument(help="Output bundle path")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Export one artifact as a complete offline bundle."""

    _run(
        lambda: {"destination": str(_registry(root).export_bundle(reference, destination))},
        output_json,
    )


@registry_app.command("import")
def import_command(
    bundle: Annotated[Path, typer.Argument(help="Input bundle path")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    license_id: Annotated[str | None, typer.Option("--license")] = None,
    allow_unsigned: Annotated[bool, typer.Option("--allow-unsigned")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify and publish a complete bundle only after all checks pass."""

    _run(
        lambda: _registry(root).import_bundle(
            bundle,
            accepted_licenses=() if license_id is None else (license_id,),
            allow_unsigned=allow_unsigned,
        ),
        output_json,
    )


@registry_app.command("gc")
def gc_command(
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually delete unprotected objects")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report or remove only unprotected local artifacts."""

    _run(lambda: _registry(root).collect_garbage(dry_run=not apply), output_json)


@registry_app.command("compare")
def compare_command(
    left: Annotated[str, typer.Argument(help="First digest or alias")],
    right: Annotated[str, typer.Argument(help="Second digest or alias")],
    root: Annotated[Path, typer.Option("--root")] = _DEFAULT_ROOT,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compare immutable artifact metadata and content identity."""

    _run(lambda: _registry(root).compare(left, right), output_json)


__all__ = ["registry_app"]
