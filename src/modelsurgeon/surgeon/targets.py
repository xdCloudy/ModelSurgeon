"""Versioned supervised target derivation for surgeon training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.experiments.schema import MetricObservation, MetricState, MutationExampleRecord

TARGET_SCHEMA_VERSION: Final[int] = 1


class TargetBuildError(ValueError):
    """Raised when metric observations cannot satisfy a target schema."""


class TargetDirection(StrEnum):
    """Interpretation of a positive post-minus-baseline delta."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True, slots=True)
class TargetMetricSpec:
    """One versioned continuous target and optional safety constraint."""

    name: str
    aliases: tuple[str, ...]
    unit: str
    direction: TargetDirection
    maximum_degradation: float | None = None
    required_for_safe_label: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.aliases or not self.unit:
            raise TargetBuildError("target metric name, aliases, and unit are required")
        if self.name not in self.aliases:
            raise TargetBuildError("target metric aliases must include the canonical name")
        if self.aliases != tuple(dict.fromkeys(self.aliases)):
            raise TargetBuildError("target metric aliases must be unique and ordered")
        if self.maximum_degradation is not None and (
            not math.isfinite(self.maximum_degradation) or self.maximum_degradation < 0
        ):
            raise TargetBuildError("maximum degradation must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "unit": self.unit,
            "direction": self.direction.value,
            "maximum_degradation": self.maximum_degradation,
            "required_for_safe_label": self.required_for_safe_label,
        }


@dataclass(frozen=True, slots=True)
class TargetSchema:
    """Canonical target schema used to derive labels from experiment observations."""

    metrics: tuple[TargetMetricSpec, ...]
    version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != TARGET_SCHEMA_VERSION:
            raise TargetBuildError(f"unsupported target schema version {self.version}")
        if not self.metrics:
            raise TargetBuildError("target schema requires at least one metric")
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise TargetBuildError("target metric names must be unique and canonical")
        aliases = [alias for item in self.metrics for alias in item.aliases]
        if len(aliases) != len(set(aliases)):
            raise TargetBuildError("target metric aliases cannot overlap")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "metrics": [item.to_record() for item in self.metrics],
        }


DEFAULT_TARGET_SCHEMA: Final[TargetSchema] = TargetSchema(
    tuple(
        sorted(
            (
                TargetMetricSpec(
                    "behavior",
                    ("behavior", "behavior_score", "task_score"),
                    "score",
                    TargetDirection.HIGHER_IS_BETTER,
                ),
                TargetMetricSpec(
                    "latency",
                    ("latency", "latency_seconds", "wall_seconds"),
                    "seconds",
                    TargetDirection.LOWER_IS_BETTER,
                ),
                TargetMetricSpec(
                    "loss",
                    ("loss", "mean_loss"),
                    "loss",
                    TargetDirection.LOWER_IS_BETTER,
                ),
                TargetMetricSpec(
                    "parameters",
                    ("parameters", "parameter_count"),
                    "parameters",
                    TargetDirection.LOWER_IS_BETTER,
                    required_for_safe_label=False,
                ),
                TargetMetricSpec(
                    "perplexity",
                    ("perplexity", "ppl"),
                    "perplexity",
                    TargetDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda item: item.name,
        )
    )
)


@dataclass(frozen=True, slots=True)
class TargetValue:
    name: str
    unit: str
    direction: TargetDirection
    value: float | None
    mask: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise TargetBuildError("target values require a name and unit")
        if self.mask:
            if self.value is None or not math.isfinite(self.value):
                raise TargetBuildError("present target values require a finite value")
            if self.reason is not None:
                raise TargetBuildError("present target values cannot carry a missingness reason")
        else:
            if self.value is not None:
                raise TargetBuildError("masked target values cannot carry a numeric value")
            if not self.reason:
                raise TargetBuildError("masked target values require an explicit reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "value": self.value,
            "mask": self.mask,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SupervisedTargets:
    """Continuous delta targets plus a threshold-derived safe-mutation label."""

    example_id: str
    values: tuple[TargetValue, ...]
    safe_mutation: bool | None
    safe_mutation_mask: bool
    safe_label_reason: str | None
    schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.example_id:
            raise TargetBuildError("supervised targets require an example identity")
        if self.schema_version != TARGET_SCHEMA_VERSION:
            raise TargetBuildError("unsupported supervised target schema version")
        names = tuple(value.name for value in self.values)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise TargetBuildError("target values must be unique and canonical")
        if self.safe_mutation_mask:
            if self.safe_mutation is None or self.safe_label_reason is not None:
                raise TargetBuildError("present safe labels require a boolean and no reason")
        elif self.safe_mutation is not None or not self.safe_label_reason:
            raise TargetBuildError("masked safe labels require an explicit reason")

    def value(self, name: str) -> TargetValue:
        for item in self.values:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "values": [item.to_record() for item in self.values],
            "safe_mutation": self.safe_mutation,
            "safe_mutation_mask": self.safe_mutation_mask,
            "safe_label_reason": self.safe_label_reason,
        }


def _observation_map(
    observations: Sequence[MetricObservation] | Sequence[Mapping[str, object]],
) -> dict[str, MetricObservation | Mapping[str, object]]:
    output: dict[str, MetricObservation | Mapping[str, object]] = {}
    for item in observations:
        name = item.name if isinstance(item, MetricObservation) else item.get("name")
        if not isinstance(name, str) or not name:
            raise TargetBuildError("metric observations require non-empty names")
        if name in output:
            raise TargetBuildError(f"duplicate metric observation {name!r}")
        output[name] = item
    return output


def _metric_fields(
    item: MetricObservation | Mapping[str, object],
) -> tuple[str, float | None, str | None, str | None]:
    if isinstance(item, MetricObservation):
        return item.state.value, item.value, item.unit, item.reason
    state = item.get("state")
    value = item.get("value")
    unit = item.get("unit")
    reason = item.get("reason")
    if not isinstance(state, str):
        raise TargetBuildError("metric state must be a string")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise TargetBuildError("metric value must be numeric or null")
    if unit is not None and not isinstance(unit, str):
        raise TargetBuildError("metric unit must be a string or null")
    if reason is not None and not isinstance(reason, str):
        raise TargetBuildError("metric reason must be a string or null")
    return state, None if value is None else float(value), unit, reason


def _find_observation(
    observations: Mapping[str, MetricObservation | Mapping[str, object]],
    spec: TargetMetricSpec,
) -> MetricObservation | Mapping[str, object] | None:
    matches = [observations[alias] for alias in spec.aliases if alias in observations]
    if len(matches) > 1:
        raise TargetBuildError(
            f"multiple aliases are present for target {spec.name!r}; keep one canonical observation"
        )
    return matches[0] if matches else None


def _measured_value(
    item: MetricObservation | Mapping[str, object] | None,
    *,
    spec: TargetMetricSpec,
    phase: str,
) -> tuple[float | None, str | None]:
    if item is None:
        return None, f"{phase} metric is absent"
    state, value, unit, reason = _metric_fields(item)
    if state != MetricState.MEASURED.value:
        detail = reason or state
        return None, f"{phase} metric is {state}: {detail}"
    if value is None or not math.isfinite(value):
        raise TargetBuildError(f"{phase} metric {spec.name!r} is measured without a finite value")
    # Older experiment records may omit units.  The target schema is authoritative in that
    # case; an explicit conflicting unit still fails closed through a masked target.
    if unit is not None and unit != spec.unit:
        return None, (
            f"{phase} metric unit {unit!r} does not match target schema unit {spec.unit!r}"
        )
    return value, None


def _record_view(
    example: MutationExampleRecord | Mapping[str, object],
) -> tuple[
    str,
    Sequence[MetricObservation] | Sequence[Mapping[str, object]],
    Sequence[MetricObservation] | Sequence[Mapping[str, object]],
]:
    if isinstance(example, MutationExampleRecord):
        return example.example_id, example.baseline_metrics, example.post_metrics
    example_id = example.get("example_id")
    baseline = example.get("baseline_metrics")
    post = example.get("post_metrics")
    if not isinstance(example_id, str) or not example_id:
        raise TargetBuildError("mutation example record requires example_id")
    if not isinstance(baseline, list) or not all(isinstance(item, Mapping) for item in baseline):
        raise TargetBuildError("mutation example record baseline_metrics must be a list of records")
    if not isinstance(post, list) or not all(isinstance(item, Mapping) for item in post):
        raise TargetBuildError("mutation example record post_metrics must be a list of records")
    return example_id, baseline, post


def derive_supervised_targets(
    example: MutationExampleRecord | Mapping[str, object],
    schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> SupervisedTargets:
    """Derive post-minus-baseline deltas and a fully masked-safe label contract."""

    example_id, baseline_raw, post_raw = _record_view(example)
    baseline = _observation_map(baseline_raw)
    post = _observation_map(post_raw)
    values: list[TargetValue] = []
    safe_failures: list[str] = []
    safe_missing: list[str] = []
    constrained_count = sum(
        spec.required_for_safe_label and spec.maximum_degradation is not None
        for spec in schema.metrics
    )

    for spec in schema.metrics:
        baseline_value, baseline_reason = _measured_value(
            _find_observation(baseline, spec),
            spec=spec,
            phase="baseline",
        )
        post_value, post_reason = _measured_value(
            _find_observation(post, spec),
            spec=spec,
            phase="post",
        )
        if baseline_value is None or post_value is None:
            reason = "; ".join(
                reason for reason in (baseline_reason, post_reason) if reason is not None
            )
            values.append(
                TargetValue(spec.name, spec.unit, spec.direction, None, False, reason)
            )
            if spec.required_for_safe_label and spec.maximum_degradation is not None:
                safe_missing.append(f"{spec.name}: {reason}")
            continue

        delta = post_value - baseline_value
        values.append(TargetValue(spec.name, spec.unit, spec.direction, delta, True))

        if spec.required_for_safe_label and spec.maximum_degradation is not None:
            degradation = (
                delta
                if spec.direction is TargetDirection.LOWER_IS_BETTER
                else -delta
            )
            if degradation > spec.maximum_degradation:
                safe_failures.append(
                    f"{spec.name} degradation {degradation:.12g} exceeds "
                    f"{spec.maximum_degradation:.12g} {spec.unit}"
                )

    if constrained_count == 0:
        return SupervisedTargets(
            example_id,
            tuple(values),
            None,
            False,
            "target schema defines no safety thresholds",
        )
    if safe_missing:
        return SupervisedTargets(
            example_id,
            tuple(values),
            None,
            False,
            "safe label is masked because required metrics are missing: " + "; ".join(safe_missing),
        )
    return SupervisedTargets(
        example_id,
        tuple(values),
        not safe_failures,
        True,
        None,
    )


def schema_with_thresholds(
    thresholds: Mapping[str, float],
    *,
    base: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> TargetSchema:
    """Return a schema with explicit per-target absolute degradation thresholds."""

    unknown = set(thresholds) - {item.name for item in base.metrics}
    if unknown:
        raise TargetBuildError(f"unknown target threshold names: {sorted(unknown)}")
    metrics = tuple(
        TargetMetricSpec(
            item.name,
            item.aliases,
            item.unit,
            item.direction,
            thresholds.get(item.name, item.maximum_degradation),
            item.required_for_safe_label,
        )
        for item in base.metrics
    )
    return TargetSchema(metrics, base.version)