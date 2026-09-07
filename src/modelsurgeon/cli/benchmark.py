"""Deterministic, resumable competitor benchmark matrix commands."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast, runtime_checkable

import typer

from modelsurgeon.evaluation import (
    DEFAULT_BENCHMARK_PROTOCOL,
    BenchmarkProtocolManifest,
    ProtocolApplicability,
    render_benchmark_protocol,
)
from modelsurgeon.experiments.identity import canonical_identity_json

BENCHMARK_CLI_SCHEMA_VERSION = 1


class BenchmarkCommandError(ValueError):
    """Raised when a matrix is malformed or cannot be resumed safely."""


class BenchmarkProtocolInvalid(BenchmarkCommandError):
    """The matrix or its immutable protocol is invalid."""


class BenchmarkMethodFailure(BenchmarkCommandError):
    """An executor failed for a benchmark cell."""


class BenchmarkComparisonFailure(BenchmarkCommandError):
    """A completed matrix contains a failed or invalid comparison."""


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BenchmarkProtocolInvalid(f"{path} must be an object")
    return cast(dict[str, object], value)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkProtocolInvalid(f"{path} must be a non-empty string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_identity_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkProtocolInvalid(f"cannot read JSON manifest: {path}") from error
    return _object(raw, "manifest")


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_identity_json(payload) + "\n")
    except FileExistsError as error:
        raise BenchmarkCommandError(f"refusing to overwrite immutable file: {path}") from error
    except OSError as error:
        raise BenchmarkCommandError(f"cannot write {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class BenchmarkCellPlan:
    """One planned cell, including a retained skip decision."""

    cell_id: str
    model_rung: str
    method_id: str
    task_id: str
    applicability: str
    command: tuple[str, ...]
    status: str
    reason: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "model_rung": self.model_rung,
            "method_id": self.method_id,
            "task_id": self.task_id,
            "applicability": self.applicability,
            "command": list(self.command),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMatrixPlan:
    """Content-addressed plan persisted before any external method runs."""

    protocol_id: str
    protocol_schema_version: int
    executor: str
    cells: tuple[BenchmarkCellPlan, ...]
    plan_id: str
    schema_version: int = BENCHMARK_CLI_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "benchmark_matrix_plan",
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "protocol_id": self.protocol_id,
            "protocol_schema_version": self.protocol_schema_version,
            "executor": self.executor,
            "cells": [cell.to_record() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCellResult:
    cell_id: str
    status: str
    reason: str | None
    metrics: Mapping[str, object]
    started_at: float | None = None
    finished_at: float | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "status": self.status,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMatrixState:
    plan_id: str
    protocol_id: str
    results: tuple[BenchmarkCellResult, ...]
    schema_version: int = BENCHMARK_CLI_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "benchmark_matrix_state",
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "protocol_id": self.protocol_id,
            "results": [result.to_record() for result in self.results],
        }


@runtime_checkable
class BenchmarkExecutor(Protocol):
    """Trusted method adapter loaded explicitly by the caller."""

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]: ...


class SubprocessBenchmarkExecutor:
    """Bounded JSON-line adapter for a tiny real external-method smoke."""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 60.0) -> None:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise BenchmarkCommandError("subprocess command must be a non-empty string array")
        if timeout_seconds <= 0:
            raise BenchmarkCommandError("subprocess timeout must be positive")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]:
        payload = canonical_identity_json(dict(cell))
        try:
            completed = subprocess.run(
                self.command,
                input=payload + "\n",
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BenchmarkMethodFailure(
                "external benchmark method exceeded its time budget"
            ) from error
        if completed.returncode != 0:
            raise BenchmarkMethodFailure(
                f"external benchmark method exited with status {completed.returncode}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise BenchmarkMethodFailure("external method did not emit one JSON result") from error
        if not isinstance(result, dict):
            raise BenchmarkMethodFailure("external method result must be an object")
        return cast(dict[str, object], result)


class DeterministicFakeBenchmarkExecutor:
    """Offline executor used by tests and dry CI smoke paths."""

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]:
        cell_id = _string(cell.get("cell_id"), "cell_id")
        value = int(cell_id[-8:], 16) / 0xFFFFFFFF
        return {"status": "success", "metrics": {"quality": value}, "artifact": None}


def fake_executor_factory() -> BenchmarkExecutor:
    return DeterministicFakeBenchmarkExecutor()


def load_benchmark_executor(specification: str) -> BenchmarkExecutor:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise BenchmarkCommandError("executor must use module:factory syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        executor = factory()
    except Exception as error:
        raise BenchmarkCommandError(f"cannot load benchmark executor {specification!r}") from error
    if not isinstance(executor, BenchmarkExecutor):
        raise BenchmarkCommandError("executor does not implement execute(cell)")
    return executor


def _cell_id(model_rung: str, method_id: str, task_id: str, protocol_id: str) -> str:
    return "cell_" + _canonical_digest(
        {
            "protocol_id": protocol_id,
            "model_rung": model_rung,
            "method_id": method_id,
            "task_id": task_id,
        }
    )


def build_benchmark_plan(
    protocol: BenchmarkProtocolManifest = DEFAULT_BENCHMARK_PROTOCOL,
    *,
    executor: str = "modelsurgeon.cli.benchmark:fake_executor_factory",
) -> BenchmarkMatrixPlan:
    """Create a deterministic plan without downloading models or running methods."""

    cells: list[BenchmarkCellPlan] = []
    for cell in sorted(protocol.cells, key=lambda item: item.cell_key):
        supported = cell.applicability is ProtocolApplicability.SUPPORTED
        status = "pending" if supported else "unsupported"
        cells.append(
            BenchmarkCellPlan(
                _cell_id(*cell.cell_key, protocol.protocol_id),
                cell.model_rung,
                cell.method_id,
                cell.task_id,
                cell.applicability.value,
                (
                    "modelsurgeon",
                    "benchmark",
                    "run",
                    "--cell-id",
                    _cell_id(*cell.cell_key, protocol.protocol_id),
                ),
                status,
                None if supported else cell.reason,
            )
        )
    identity = {
        "schema_version": BENCHMARK_CLI_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_schema_version": protocol.schema_version,
        "executor": executor,
        "cells": [cell.to_record() for cell in cells],
    }
    return BenchmarkMatrixPlan(
        protocol.protocol_id,
        protocol.schema_version,
        executor,
        tuple(cells),
        "plan_" + _canonical_digest(identity),
    )


def _plan_from_record(raw: Mapping[str, object]) -> BenchmarkMatrixPlan:
    if raw.get("record_type") != "benchmark_matrix_plan":
        raise BenchmarkProtocolInvalid("not a benchmark matrix plan")
    if raw.get("schema_version") != BENCHMARK_CLI_SCHEMA_VERSION:
        raise BenchmarkProtocolInvalid("unsupported benchmark CLI schema version")
    cells_raw = raw.get("cells")
    if not isinstance(cells_raw, list):
        raise BenchmarkProtocolInvalid("plan cells must be an array")
    cells: list[BenchmarkCellPlan] = []
    for index, value in enumerate(cells_raw):
        item = _object(value, f"cells[{index}]")
        command = item.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise BenchmarkProtocolInvalid(f"cells[{index}].command must be a string array")
        cells.append(
            BenchmarkCellPlan(
                _string(item.get("cell_id"), f"cells[{index}].cell_id"),
                _string(item.get("model_rung"), f"cells[{index}].model_rung"),
                _string(item.get("method_id"), f"cells[{index}].method_id"),
                _string(item.get("task_id"), f"cells[{index}].task_id"),
                _string(item.get("applicability"), f"cells[{index}].applicability"),
                tuple(command),
                _string(item.get("status"), f"cells[{index}].status"),
                _optional_string(item.get("reason"), f"cells[{index}].reason"),
            )
        )
    protocol_schema_version = raw.get("protocol_schema_version")
    if not isinstance(protocol_schema_version, int) or isinstance(protocol_schema_version, bool):
        raise BenchmarkProtocolInvalid("protocol_schema_version must be an integer")
    return BenchmarkMatrixPlan(
        _string(raw.get("protocol_id"), "protocol_id"),
        protocol_schema_version,
        _string(raw.get("executor"), "executor"),
        tuple(cells),
        _string(raw.get("plan_id"), "plan_id"),
    )


def _state_from_record(raw: Mapping[str, object]) -> BenchmarkMatrixState:
    if raw.get("record_type") != "benchmark_matrix_state":
        raise BenchmarkProtocolInvalid("not a benchmark matrix state")
    results_raw = raw.get("results")
    if not isinstance(results_raw, list):
        raise BenchmarkProtocolInvalid("state results must be an array")
    results: list[BenchmarkCellResult] = []
    for index, value in enumerate(results_raw):
        item = _object(value, f"results[{index}]")
        metrics = item.get("metrics")
        results.append(
            BenchmarkCellResult(
                _string(item.get("cell_id"), f"results[{index}].cell_id"),
                _string(item.get("status"), f"results[{index}].status"),
                _optional_string(item.get("reason"), f"results[{index}].reason"),
                _object(metrics, f"results[{index}].metrics"),
                float(cast(int | float, item["started_at"]))
                if isinstance(item.get("started_at"), (int, float))
                else None,
                float(cast(int | float, item["finished_at"]))
                if isinstance(item.get("finished_at"), (int, float))
                else None,
            )
        )
    return BenchmarkMatrixState(
        _string(raw.get("plan_id"), "plan_id"),
        _string(raw.get("protocol_id"), "protocol_id"),
        tuple(results),
    )


def _load_plan(path: Path) -> BenchmarkMatrixPlan:
    return _plan_from_record(_read_json(path))


def _load_state(path: Path) -> BenchmarkMatrixState:
    return _state_from_record(_read_json(path))


def _validate_state(plan: BenchmarkMatrixPlan, state: BenchmarkMatrixState) -> None:
    if state.plan_id != plan.plan_id or state.protocol_id != plan.protocol_id:
        raise BenchmarkProtocolInvalid("state belongs to a different immutable plan")
    known = {cell.cell_id for cell in plan.cells}
    actual = [result.cell_id for result in state.results]
    if len(actual) != len(set(actual)) or not set(actual).issubset(known):
        raise BenchmarkProtocolInvalid("state contains duplicate or unknown cell identities")


def initialize_state(plan: BenchmarkMatrixPlan) -> BenchmarkMatrixState:
    return BenchmarkMatrixState(plan.plan_id, plan.protocol_id, ())


def run_benchmark_matrix(
    plan: BenchmarkMatrixPlan,
    state: BenchmarkMatrixState,
    executor: BenchmarkExecutor,
) -> BenchmarkMatrixState:
    """Run only pending cells and persist immutable results in caller-owned state."""

    _validate_state(plan, state)
    results = {item.cell_id: item for item in state.results}
    for cell in plan.cells:
        if cell.status != "pending" or cell.cell_id in results:
            continue
        started = time.time()
        try:
            raw = dict(executor.execute(cell.to_record()))
            status = raw.get("status")
            if status != "success":
                raise BenchmarkMethodFailure(
                    _string(raw.get("reason"), f"{cell.cell_id}.reason")
                    if raw.get("reason") is not None
                    else "executor returned a non-success status"
                )
            metrics = raw.get("metrics", {})
            if not isinstance(metrics, dict):
                raise BenchmarkMethodFailure("executor metrics must be an object")
            results[cell.cell_id] = BenchmarkCellResult(
                cell.cell_id,
                "success",
                None,
                cast(dict[str, object], metrics),
                started,
                time.time(),
            )
        except BenchmarkMethodFailure as error:
            results[cell.cell_id] = BenchmarkCellResult(
                cell.cell_id, "failed", str(error), {}, started, time.time()
            )
    for cell in plan.cells:
        if cell.cell_id not in results and cell.status == "unsupported":
            results[cell.cell_id] = BenchmarkCellResult(
                cell.cell_id, "unsupported", cell.reason, {}, None, None
            )
    return BenchmarkMatrixState(
        plan.plan_id,
        plan.protocol_id,
        tuple(results[cell.cell_id] for cell in plan.cells if cell.cell_id in results),
    )


def audit_benchmark_matrix(
    plan: BenchmarkMatrixPlan,
    state: BenchmarkMatrixState,
) -> dict[str, object]:
    _validate_state(plan, state)
    expected = {cell.cell_id: cell.status for cell in plan.cells}
    statuses = {result.status for result in state.results}
    missing = sorted(set(expected) - {result.cell_id for result in state.results})
    failed = sorted(result.cell_id for result in state.results if result.status == "failed")
    invalid = sorted(
        result.cell_id
        for result in state.results
        if result.status not in {"success", "unsupported", "failed"}
    )
    return {
        "record_type": "benchmark_matrix_audit",
        "schema_version": BENCHMARK_CLI_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "protocol_id": plan.protocol_id,
        "cell_count": len(plan.cells),
        "result_count": len(state.results),
        "missing_cells": missing,
        "failed_cells": failed,
        "invalid_cells": invalid,
        "status_counts": {
            status: sum(result.status == status for result in state.results)
            for status in sorted(statuses)
        },
        "valid": not missing and not failed and not invalid,
    }


def render_benchmark_matrix_report(
    plan: BenchmarkMatrixPlan,
    state: BenchmarkMatrixState,
    *,
    format: str,
) -> str:
    audit = audit_benchmark_matrix(plan, state)
    if format == "json":
        return (
            canonical_identity_json(
                {"plan": plan.to_record(), "state": state.to_record(), "audit": audit}
            )
            + "\n"
        )
    if format != "markdown":
        raise BenchmarkCommandError(f"unsupported report format: {format}")
    lines = [
        f"# Benchmark matrix {plan.plan_id}",
        "",
        f"- Protocol: `{plan.protocol_id}`",
        f"- Cells: {audit['result_count']}/{audit['cell_count']}",
        f"- Valid: **{audit['valid']}**",
        "",
        "| Cell | Method | Task | Status | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_id = {cell.cell_id: cell for cell in plan.cells}
    for result in state.results:
        cell = by_id[result.cell_id]
        lines.append(
            f"| `{result.cell_id}` | `{cell.method_id}` | `{cell.task_id}` | "
            f"`{result.status}` | {result.reason or ''} |"
        )
    return "\n".join(lines) + "\n"


def _save_state(path: Path, state: BenchmarkMatrixState) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    _write_new(temporary, state.to_record())
    os.replace(temporary, path)


benchmark_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@benchmark_app.command("plan")
def plan_command(
    output: Annotated[Path, typer.Option("--output", help="New immutable plan JSON")],
    executor: Annotated[str, typer.Option("--executor", help="Executor module:factory")]
    = "modelsurgeon.cli.benchmark:fake_executor_factory",
) -> None:
    """Create a deterministic benchmark plan without executing any cell."""
    try:
        plan = build_benchmark_plan(executor=executor)
        _write_new(output, plan.to_record())
        typer.echo(canonical_identity_json(plan.to_record()))
    except BenchmarkCommandError as error:
        typer.echo(f"benchmark protocol error: {error}", err=True)
        raise typer.Exit(2) from error


@benchmark_app.command("protocol")
def protocol_command(
    format: Annotated[str, typer.Option("--format", help="json or markdown")] = "json",
) -> None:
    """Print the immutable benchmark protocol used to make plans."""
    try:
        typer.echo(render_benchmark_protocol(format=format), nl=False)
    except ValueError as error:
        typer.echo(f"benchmark protocol error: {error}", err=True)
        raise typer.Exit(2) from error


@benchmark_app.command("run")
def run_command(
    plan: Annotated[Path, typer.Option("--plan", help="Immutable benchmark plan JSON")],
    state: Annotated[Path, typer.Option("--state", help="Resumable state JSON")],
    executor: Annotated[
        str | None,
        typer.Option("--executor", help="Executor module:factory"),
    ] = None,
) -> None:
    """Run pending cells, preserving completed immutable cells for resume."""
    try:
        loaded_plan = _load_plan(plan)
        loaded_state = initialize_state(loaded_plan) if not state.exists() else _load_state(state)
        selected = executor or loaded_plan.executor
        result = run_benchmark_matrix(loaded_plan, loaded_state, load_benchmark_executor(selected))
        _save_state(state, result)
        typer.echo(canonical_identity_json(result.to_record()))
        audit = audit_benchmark_matrix(loaded_plan, result)
        if audit["failed_cells"]:
            raise typer.Exit(1)
    except BenchmarkProtocolInvalid as error:
        typer.echo(f"benchmark protocol error: {error}", err=True)
        raise typer.Exit(2) from error
    except BenchmarkCommandError as error:
        typer.echo(f"benchmark method error: {error}", err=True)
        raise typer.Exit(1) from error


@benchmark_app.command("resume")
def resume_command(
    plan: Annotated[Path, typer.Option("--plan", help="Immutable benchmark plan JSON")],
    state: Annotated[Path, typer.Option("--state", help="Resumable state JSON")],
    executor: Annotated[
        str | None,
        typer.Option("--executor", help="Executor module:factory"),
    ] = None,
) -> None:
    """Resume exactly the pending cells in an existing matrix state."""
    run_command(plan=plan, state=state, executor=executor)


@benchmark_app.command("import")
def import_command(
    plan: Annotated[Path, typer.Option("--plan", help="Immutable benchmark plan JSON")],
    state: Annotated[Path, typer.Option("--state", help="Resumable state JSON")],
    result: Annotated[Path, typer.Option("--result", help="One external cell result JSON")],
) -> None:
    """Import one externally executed result without overwriting existing cells."""
    try:
        loaded_plan = _load_plan(plan)
        loaded_state = initialize_state(loaded_plan) if not state.exists() else _load_state(state)
        _validate_state(loaded_plan, loaded_state)
        payload = _read_json(result)
        cell_id = _string(payload.get("cell_id"), "cell_id")
        if cell_id not in {cell.cell_id for cell in loaded_plan.cells}:
            raise BenchmarkProtocolInvalid("result references an unknown cell")
        if any(item.cell_id == cell_id for item in loaded_state.results):
            raise BenchmarkCommandError("cell already has an immutable result")
        status = _string(payload.get("status"), "status")
        if status not in {"success", "unsupported", "failed"}:
            raise BenchmarkProtocolInvalid("result status is invalid")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise BenchmarkProtocolInvalid("result metrics must be an object")
        imported = BenchmarkCellResult(
            cell_id,
            status,
            _optional_string(payload.get("reason"), "reason"),
            cast(dict[str, object], metrics),
        )
        updated = BenchmarkMatrixState(
            loaded_plan.plan_id,
            loaded_plan.protocol_id,
            (*loaded_state.results, imported),
        )
        _save_state(state, updated)
        typer.echo(canonical_identity_json(imported.to_record()))
    except BenchmarkProtocolInvalid as error:
        typer.echo(f"benchmark protocol error: {error}", err=True)
        raise typer.Exit(2) from error
    except BenchmarkCommandError as error:
        typer.echo(f"benchmark import error: {error}", err=True)
        raise typer.Exit(1) from error


@benchmark_app.command("audit")
def audit_command(
    plan: Annotated[Path, typer.Option("--plan", help="Immutable benchmark plan JSON")],
    state: Annotated[Path, typer.Option("--state", help="Resumable state JSON")],
) -> None:
    """Audit completeness and terminal outcomes with machine-readable output."""
    try:
        loaded_plan = _load_plan(plan)
        loaded_state = _load_state(state)
        audit = audit_benchmark_matrix(loaded_plan, loaded_state)
        typer.echo(canonical_identity_json(audit))
        if audit["invalid_cells"] or audit["missing_cells"]:
            raise typer.Exit(2)
        if audit["failed_cells"]:
            raise typer.Exit(1)
    except BenchmarkProtocolInvalid as error:
        typer.echo(f"benchmark protocol error: {error}", err=True)
        raise typer.Exit(2) from error


@benchmark_app.command("report")
def matrix_report_command(
    plan: Annotated[Path, typer.Option("--plan", help="Immutable benchmark plan JSON")],
    state: Annotated[Path, typer.Option("--state", help="Resumable state JSON")],
    format: Annotated[str, typer.Option("--format", help="json or markdown")] = "json",
    output: Annotated[Path | None, typer.Option("--output", help="New report file")] = None,
) -> None:
    """Render an auditable matrix report without changing its state."""
    try:
        rendered = render_benchmark_matrix_report(
            _load_plan(plan), _load_state(state), format=format
        )
        if output is not None:
            _write_new(output, {"rendered": rendered})
        typer.echo(rendered, nl=False)
    except BenchmarkCommandError as error:
        typer.echo(f"benchmark report error: {error}", err=True)
        raise typer.Exit(2) from error
