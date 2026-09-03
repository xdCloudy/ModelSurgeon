"""Bounded feature extraction orchestration and partition persistence."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

import typer

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.features.cache import FeaturePartitionCache, FeaturePartitionKey
from modelsurgeon.features.schema import FeatureRecord
from modelsurgeon.graph import ComponentId

FEATURE_COMMAND_SCHEMA_VERSION = 1


class FeatureCommandError(RuntimeError):
    """Raised when feature extraction cannot be planned or persisted safely."""


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    name: str
    extractor: str
    extractor_version: str


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    group: FeatureGroup
    components: tuple[ComponentId, ...]
    cpu_only: bool
    max_records: int


@runtime_checkable
class FeatureRuntime(Protocol):
    """A trusted adapter which extracts one declared feature group."""

    def extract(
        self, request: FeatureRequest
    ) -> Mapping[ComponentId, tuple[FeatureRecord, ...]]: ...


@dataclass(frozen=True, slots=True)
class FeaturePlan:
    model_revision: str
    input_revision: str
    groups: tuple[FeatureGroup, ...]
    components: tuple[ComponentId, ...]
    max_records: int


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeatureCommandError(f"{label} must be non-empty text")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FeatureCommandError(f"{label} must be an object")
    return value


def _groups(value: object) -> tuple[FeatureGroup, ...]:
    if not isinstance(value, list) or not value:
        raise FeatureCommandError("groups must be a non-empty list")
    result: list[FeatureGroup] = []
    for raw in value:
        item = _object(raw, "feature group")
        if set(item) != {"name", "extractor", "extractor_version"}:
            raise FeatureCommandError("feature groups have missing or unknown fields")
        result.append(
            FeatureGroup(
                _text(item["name"], "group name"),
                _text(item["extractor"], "group extractor"),
                _text(item["extractor_version"], "group extractor version"),
            )
        )
    if tuple(item.name for item in result) != tuple(sorted({item.name for item in result})):
        raise FeatureCommandError("feature group names must be sorted and unique")
    identities = tuple((item.extractor, item.extractor_version) for item in result)
    if len(identities) != len(set(identities)):
        raise FeatureCommandError("feature extractor identities must be unique")
    return tuple(result)


def _components(value: object) -> tuple[ComponentId, ...]:
    if not isinstance(value, list) or not value:
        raise FeatureCommandError("components must be a non-empty list")
    try:
        result = tuple(ComponentId.parse(_text(item, "component")) for item in value)
    except ValueError as error:
        raise FeatureCommandError("components contain an invalid component ID") from error
    if result != tuple(sorted(set(result))):
        raise FeatureCommandError("components must be sorted and unique")
    return result


def load_feature_plan(path: Path) -> FeaturePlan:
    """Load a strict, deterministic feature-extraction manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureCommandError(f"cannot read feature config: {error}") from error
    record = _object(raw, "feature config")
    expected = {
        "schema_version",
        "model_revision",
        "input_revision",
        "groups",
        "components",
        "max_records",
    }
    if set(record) != expected:
        raise FeatureCommandError("feature config has missing or unknown fields")
    if record["schema_version"] != FEATURE_COMMAND_SCHEMA_VERSION:
        raise FeatureCommandError("unsupported feature-command schema version")
    max_records = record["max_records"]
    if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
        raise FeatureCommandError("max_records must be a positive integer")
    return FeaturePlan(
        _text(record["model_revision"], "model revision"),
        _text(record["input_revision"], "input revision"),
        _groups(record["groups"]),
        _components(record["components"]),
        max_records,
    )


def load_feature_runtime(specification: str) -> FeatureRuntime:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise FeatureCommandError("runtime must use module:factory syntax")
    try:
        runtime = getattr(importlib.import_module(module_name), attribute)()
    except Exception as error:
        raise FeatureCommandError(f"cannot load feature runtime {specification!r}") from error
    if not isinstance(runtime, FeatureRuntime):
        raise FeatureCommandError("feature runtime does not implement extract(request)")
    return runtime


def _filtered_components(
    components: tuple[ComponentId, ...],
    filters: tuple[str, ...],
) -> tuple[ComponentId, ...]:
    if not filters:
        return components
    selected = tuple(
        item for item in components if any(str(item).startswith(prefix) for prefix in filters)
    )
    if not selected:
        raise FeatureCommandError("component filters select no declared components")
    return selected


def run_feature_plan(
    plan: FeaturePlan,
    runtime: FeatureRuntime,
    *,
    cache_path: Path,
    cpu_only: bool,
    component_filters: tuple[str, ...] = (),
) -> dict[str, object]:
    """Extract, reuse, and atomically publish one bounded set of feature partitions."""

    components = _filtered_components(plan.components, component_filters)
    cache = FeaturePartitionCache(cache_path)
    remaining = plan.max_records
    outcomes: list[dict[str, object]] = []
    for group in plan.groups:
        keys = tuple(
            FeaturePartitionKey(
                plan.model_revision,
                plan.input_revision,
                component,
                group.extractor,
                group.extractor_version,
            )
            for component in components
        )
        cached = {key.component_id: cache.load(key) for key in keys}
        missing = tuple(component for component, partition in cached.items() if partition is None)
        reused_count = sum(
            len(partition.records) for partition in cached.values() if partition is not None
        )
        if not missing:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "reused",
                    "record_count": reused_count,
                    "reason": None,
                }
            )
            continue
        if remaining <= 0:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": "record budget exhausted",
                }
            )
            continue
        request = FeatureRequest(group, missing, cpu_only, remaining)
        try:
            extracted = runtime.extract(request)
        except Exception as error:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if set(extracted) != set(missing):
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": "runtime returned incomplete or undeclared components",
                }
            )
            continue
        records = tuple(record for component in missing for record in extracted[component])
        if not records:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": "runtime returned no feature records",
                }
            )
            continue
        if len(records) > remaining:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": "runtime exceeded record budget",
                }
            )
            continue
        invalid = any(
            record.component_id not in missing
            or record.extractor != group.extractor
            or record.extractor_version != group.extractor_version
            for record in records
        )
        if invalid:
            outcomes.append(
                {
                    "group": group.name,
                    "state": "skipped",
                    "record_count": 0,
                    "reason": "runtime records violate declared partition identity",
                }
            )
            continue
        for component in missing:
            component_records = extracted[component]
            if not component_records:
                raise FeatureCommandError(f"runtime returned an empty partition for {component}")
            cache.write(
                FeaturePartitionKey(
                    plan.model_revision,
                    plan.input_revision,
                    component,
                    group.extractor,
                    group.extractor_version,
                ),
                component_records,
            )
        remaining -= len(records)
        outcomes.append(
            {
                "group": group.name,
                "state": "extracted",
                "record_count": len(records),
                "reason": None,
            }
        )
    return {
        "schema_version": FEATURE_COMMAND_SCHEMA_VERSION,
        "record_type": "feature_partitions",
        "model_revision": plan.model_revision,
        "input_revision": plan.input_revision,
        "cpu_only": cpu_only,
        "components": [str(item) for item in components],
        "max_records": plan.max_records,
        "records_remaining": remaining,
        "groups": outcomes,
    }


def _write_output(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
    except FileExistsError as error:
        raise FeatureCommandError(f"output already exists: {path}") from error


def features_command(
    config: Annotated[Path, typer.Argument(help="Canonical feature-extraction JSON config")],
    runtime: Annotated[str, typer.Option("--runtime", help="Trusted runtime as module:factory")],
    cache: Annotated[Path, typer.Option("--cache", help="Feature partition cache directory")],
    cpu_only: Annotated[
        bool, typer.Option("--cpu/--allow-accelerator", help="Require CPU extraction")
    ] = True,
    component: Annotated[
        list[str] | None, typer.Option("--component", help="Component-ID prefix filter")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write canonical output without overwriting")
    ] = None,
) -> None:
    """Extract selected feature groups under a deterministic record budget."""

    try:
        record = run_feature_plan(
            load_feature_plan(config),
            load_feature_runtime(runtime),
            cache_path=cache,
            cpu_only=cpu_only,
            component_filters=tuple(component or ()),
        )
        payload = canonical_identity_json(record)
        if output is not None:
            _write_output(output, payload)
        typer.echo(payload)
    except (FeatureCommandError, OSError, ValueError) as error:
        typer.echo(f"features error: {error}", err=True)
        raise typer.Exit(2) from error
