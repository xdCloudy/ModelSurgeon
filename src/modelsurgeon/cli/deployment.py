"""Plan, run, import, and audit physical deployment benchmark cells."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast, runtime_checkable

import typer

from modelsurgeon.evaluation.deployment_benchmark import (
    DEPLOYMENT_BENCHMARK_SCHEMA_VERSION,
    DeploymentArtifact,
    DeploymentBenchmarkError,
    DeploymentBenchmarkRecord,
    DeploymentFormat,
    DeploymentMeasurementPolicy,
    DeploymentOutcome,
    DeploymentRuntimeProfile,
    record_from_mapping,
    render_deployment_protocol,
)
from modelsurgeon.experiments.identity import canonical_identity_json

DEPLOYMENT_CLI_SCHEMA_VERSION = 1


class DeploymentCommandError(ValueError):
    """Raised when a deployment plan or state cannot be safely resumed."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentCommandError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(raw, dict):
        raise DeploymentCommandError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], raw)


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise DeploymentCommandError(f"refusing to overwrite immutable file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_identity_json(payload) + "\n", encoding="utf-8")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeploymentCommandError(f"{label} is required")
    return value


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DeploymentCommandError(f"{label} must be numeric")
    return int(value)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DeploymentCommandError(f"{label} must be numeric")
    return float(value)


def _plan_id(cells: Sequence[dict[str, object]], runner: str) -> str:
    digest = hashlib.sha256(
        canonical_identity_json(
            {
                "schema_version": DEPLOYMENT_CLI_SCHEMA_VERSION,
                "cells": list(cells),
                "runner": runner,
            }
        ).encode()
    ).hexdigest()
    return f"deployment_plan_{digest}"


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    runner: str
    cells: tuple[dict[str, object], ...]
    plan_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "deployment_benchmark_plan",
            "schema_version": DEPLOYMENT_CLI_SCHEMA_VERSION,
            "benchmark_schema_version": DEPLOYMENT_BENCHMARK_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "runner": self.runner,
            "cells": [dict(cell) for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class DeploymentResult:
    cell_id: str
    record: DeploymentBenchmarkRecord

    def to_record(self) -> dict[str, object]:
        return {"cell_id": self.cell_id, "record": self.record.to_record()}


@dataclass(frozen=True, slots=True)
class DeploymentState:
    plan_id: str
    results: tuple[DeploymentResult, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "deployment_benchmark_state",
            "schema_version": DEPLOYMENT_CLI_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "results": [item.to_record() for item in self.results],
        }


@runtime_checkable
class DeploymentRunner(Protocol):
    """Runner boundary for HF, llama.cpp, and external runtime adapters."""

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]: ...


def _record_for_cell(
    cell: Mapping[str, object], outcome: DeploymentOutcome, reason: str
) -> dict[str, object]:
    artifact_raw = cast(dict[str, object], cell["artifact"])
    profile_raw = cast(dict[str, object], cell["profile"])
    policy_raw = cast(dict[str, object], profile_raw["policy"])
    artifact = DeploymentArtifact(
        _string(artifact_raw["identifier"], "artifact identifier"),
        DeploymentFormat(_string(artifact_raw["format"], "artifact format")),
        _string(artifact_raw["digest"], "artifact digest"),
        _int(artifact_raw["size_bytes"], "artifact size"),
        cast(str | None, artifact_raw.get("path")),
    )
    profile = DeploymentRuntimeProfile(
        _string(profile_raw["runtime"], "runtime"),
        _int(profile_raw["cpu_threads"], "CPU threads"),
        _int(profile_raw["gpu_offload"], "GPU offload"),
        _string(profile_raw["device"], "device"),
        _string(profile_raw["hardware"], "hardware"),
        DeploymentMeasurementPolicy(
            _int(policy_raw["warmups"], "warmups"),
            _int(policy_raw["repetitions"], "repetitions"),
            _float(policy_raw["timeout_seconds"], "timeout"),
            _float(policy_raw["drift_tolerance"], "drift tolerance"),
        ),
    )
    return DeploymentBenchmarkRecord(artifact, profile, outcome, reason=reason).to_record()


class UnsupportedDeploymentRunner:
    """Safe default that retains an explicit unsupported result."""

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]:
        return _record_for_cell(
            cell, DeploymentOutcome.UNSUPPORTED, "no deployment runtime runner was configured"
        )


def unsupported_runner_factory() -> DeploymentRunner:
    return UnsupportedDeploymentRunner()


class SubprocessDeploymentRunner:
    """Bounded JSON-lines adapter for a real HF or GGUF deployment runner."""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 300.0) -> None:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise DeploymentCommandError(
                "deployment runner command must be a non-empty string array"
            )
        if timeout_seconds <= 0:
            raise DeploymentCommandError("deployment runner timeout must be positive")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds

    def execute(self, cell: Mapping[str, object]) -> Mapping[str, object]:
        try:
            completed = subprocess.run(
                self.command,
                input=canonical_identity_json(dict(cell)) + "\n",
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return _record_for_cell(cell, DeploymentOutcome.TIMEOUT, "deployment runner timed out")
        if completed.returncode != 0:
            text = f"{completed.stdout}\n{completed.stderr}".lower()
            outcome = (
                DeploymentOutcome.OOM
                if completed.returncode == 137 or "out of memory" in text
                else DeploymentOutcome.FAILED
            )
            return _record_for_cell(
                cell, outcome, f"deployment runner exited with status {completed.returncode}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise DeploymentCommandError(
                "deployment runner did not emit one JSON result"
            ) from error
        if not isinstance(result, dict):
            raise DeploymentCommandError("deployment runner result must be an object")
        return cast(dict[str, object], result)


def load_deployment_runner(specification: str) -> DeploymentRunner:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise DeploymentCommandError("runner must use module:factory syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        runner = factory()
    except Exception as error:
        raise DeploymentCommandError(f"cannot load deployment runner {specification!r}") from error
    if not isinstance(runner, DeploymentRunner):
        raise DeploymentCommandError("runner does not implement execute(cell)")
    return runner


def _plan_from_record(raw: Mapping[str, object]) -> DeploymentPlan:
    if (
        raw.get("record_type") != "deployment_benchmark_plan"
        or raw.get("schema_version") != DEPLOYMENT_CLI_SCHEMA_VERSION
    ):
        raise DeploymentCommandError("unsupported deployment plan schema")
    cells_raw = raw.get("cells")
    if (
        not isinstance(cells_raw, list)
        or not cells_raw
        or not all(isinstance(item, dict) for item in cells_raw)
    ):
        raise DeploymentCommandError("deployment plan cells must be a non-empty object array")
    cells = tuple(cast(dict[str, object], item) for item in cells_raw)
    runner = _string(raw.get("runner"), "runner")
    plan = DeploymentPlan(runner, cells, _string(raw.get("plan_id"), "plan_id"))
    if plan.plan_id != _plan_id(cells, runner):
        raise DeploymentCommandError("deployment plan identity does not match its content")
    return plan


def _state_from_record(raw: Mapping[str, object], plan: DeploymentPlan) -> DeploymentState:
    if (
        raw.get("record_type") != "deployment_benchmark_state"
        or raw.get("schema_version") != DEPLOYMENT_CLI_SCHEMA_VERSION
    ):
        raise DeploymentCommandError("unsupported deployment state schema")
    if raw.get("plan_id") != plan.plan_id:
        raise DeploymentCommandError("deployment state belongs to another plan")
    results_raw = raw.get("results", [])
    if not isinstance(results_raw, list):
        raise DeploymentCommandError("deployment state results must be an array")
    results: list[DeploymentResult] = []
    for index, item in enumerate(results_raw):
        if not isinstance(item, dict) or not isinstance(item.get("record"), dict):
            raise DeploymentCommandError(f"deployment result {index} is malformed")
        record = record_from_mapping(cast(dict[str, object], item["record"]))
        cell_id = _string(item.get("cell_id"), f"results[{index}].cell_id")
        if cell_id not in {cast(str, cell["cell_id"]) for cell in plan.cells}:
            raise DeploymentCommandError("deployment state references an unknown cell")
        results.append(DeploymentResult(cell_id, record))
    if len({item.cell_id for item in results}) != len(results):
        raise DeploymentCommandError("deployment state contains duplicate cells")
    return DeploymentState(plan.plan_id, tuple(results))


def _save_state(path: Path, state: DeploymentState) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(canonical_identity_json(state.to_record()) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_deployment_plan(
    artifacts: Sequence[DeploymentArtifact],
    profile: DeploymentRuntimeProfile,
    *,
    runner: str = "modelsurgeon.cli.deployment:unsupported_runner_factory",
) -> DeploymentPlan:
    cells: list[dict[str, object]] = []
    for artifact in sorted(
        artifacts, key=lambda item: (item.format.value, item.identifier, item.digest)
    ):
        identity = {"artifact": artifact.to_record(), "profile": profile.to_record()}
        cell_id = (
            "deployment_cell_"
            + hashlib.sha256(canonical_identity_json(identity).encode()).hexdigest()
        )
        cells.append({"cell_id": cell_id, **identity})
    if not cells:
        raise DeploymentCommandError("at least one deployment artifact is required")
    return DeploymentPlan(runner, tuple(cells), _plan_id(cells, runner))


def run_deployment_plan(
    plan: DeploymentPlan, state: DeploymentState, runner: DeploymentRunner
) -> DeploymentState:
    completed = {item.cell_id for item in state.results}
    results = list(state.results)
    for cell in plan.cells:
        cell_id = cast(str, cell["cell_id"])
        if cell_id in completed:
            continue
        record = record_from_mapping(runner.execute(cell))
        results.append(DeploymentResult(cell_id, record))
    return DeploymentState(plan.plan_id, tuple(results))


def audit_deployment_plan(plan: DeploymentPlan, state: DeploymentState) -> dict[str, object]:
    expected = {cast(str, cell["cell_id"]) for cell in plan.cells}
    actual = {item.cell_id for item in state.results}
    outcomes = {outcome.value: 0 for outcome in DeploymentOutcome}
    for item in state.results:
        outcomes[item.record.outcome.value] += 1
    return {
        "record_type": "deployment_benchmark_audit",
        "plan_id": plan.plan_id,
        "expected_cells": len(expected),
        "completed_cells": len(actual),
        "missing_cells": sorted(expected - actual),
        "outcomes": outcomes,
        "measured_cells": outcomes[DeploymentOutcome.MEASURED.value],
    }


deployment_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@deployment_app.command("protocol")
def protocol_command(format: Annotated[str, typer.Option("--format")] = "json") -> None:
    """Print the shared HF/GGUF metric and outcome protocol."""
    try:
        typer.echo(render_deployment_protocol(format=format), nl=False)
    except DeploymentBenchmarkError as error:
        typer.echo(f"deployment protocol error: {error}", err=True)
        raise typer.Exit(2) from error


@deployment_app.command("plan")
def plan_command(
    artifact: Annotated[
        list[str], typer.Option("--artifact", help="FORMAT=PATH; repeat for HF and GGUF")
    ],
    output: Annotated[Path, typer.Option("--output", help="New immutable deployment plan JSON")],
    runtime: Annotated[str, typer.Option("--runtime")] = "external",
    hardware: Annotated[str, typer.Option("--hardware")] = "unknown",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    threads: Annotated[int, typer.Option("--threads")] = 1,
    gpu_offload: Annotated[int, typer.Option("--gpu-offload")] = 0,
    runner: Annotated[
        str, typer.Option("--runner")
    ] = "modelsurgeon.cli.deployment:unsupported_runner_factory",
) -> None:
    """Create a content-addressed deployment plan without running a model."""
    try:
        artifacts: list[DeploymentArtifact] = []
        for spec in artifact:
            format_text, separator, path_text = spec.partition("=")
            if not separator or not path_text:
                raise DeploymentCommandError("--artifact must use FORMAT=PATH")
            artifacts.append(
                DeploymentArtifact.from_path(Path(path_text), DeploymentFormat(format_text))
            )
        profile = DeploymentRuntimeProfile(runtime, threads, gpu_offload, device, hardware)
        plan = build_deployment_plan(artifacts, profile, runner=runner)
        _write_new(output, plan.to_record())
        typer.echo(canonical_identity_json(plan.to_record()))
    except (DeploymentBenchmarkError, DeploymentCommandError, OSError, ValueError) as error:
        typer.echo(f"deployment plan error: {error}", err=True)
        raise typer.Exit(2) from error


@deployment_app.command("run")
def run_command(
    plan: Annotated[Path, typer.Option("--plan")],
    state: Annotated[Path, typer.Option("--state")],
    runner: Annotated[str | None, typer.Option("--runner")] = None,
) -> None:
    """Run pending deployment cells and preserve completed results for resume."""
    try:
        loaded_plan = _plan_from_record(_read_json(plan))
        loaded_state = (
            DeploymentState(loaded_plan.plan_id, ())
            if not state.exists()
            else _state_from_record(_read_json(state), loaded_plan)
        )
        result = run_deployment_plan(
            loaded_plan, loaded_state, load_deployment_runner(runner or loaded_plan.runner)
        )
        _save_state(state, result)
        typer.echo(canonical_identity_json(result.to_record()))
    except (DeploymentBenchmarkError, DeploymentCommandError, OSError, ValueError) as error:
        typer.echo(f"deployment run error: {error}", err=True)
        raise typer.Exit(1) from error


@deployment_app.command("resume")
def resume_command(
    plan: Annotated[Path, typer.Option("--plan")],
    state: Annotated[Path, typer.Option("--state")],
    runner: Annotated[str | None, typer.Option("--runner")] = None,
) -> None:
    """Resume exactly the cells missing from an existing state file."""
    run_command(plan=plan, state=state, runner=runner)


@deployment_app.command("audit")
def audit_command(
    plan: Annotated[Path, typer.Option("--plan")],
    state: Annotated[Path, typer.Option("--state")],
) -> None:
    """Audit completeness and terminal outcome counts."""
    try:
        loaded_plan = _plan_from_record(_read_json(plan))
        loaded_state = _state_from_record(_read_json(state), loaded_plan)
        typer.echo(canonical_identity_json(audit_deployment_plan(loaded_plan, loaded_state)))
    except (DeploymentBenchmarkError, DeploymentCommandError, OSError, ValueError) as error:
        typer.echo(f"deployment audit error: {error}", err=True)
        raise typer.Exit(2) from error
