"""Leakage-safe pairwise and cumulative mutation interaction evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

INTERACTION_DATASET_SCHEMA_VERSION: Final[int] = 1
INTERACTION_PROTOCOL_REVISION: Final[str] = "mutation-interaction-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InteractionDatasetError(ValueError):
    """Raised when interaction evidence, reconciliation, or splits are unsafe."""


class InteractionKind(StrEnum):
    SINGLE = "single"
    ORDERED_PAIR = "ordered_pair"
    CUMULATIVE = "cumulative"


class InteractionOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


_METRIC_UNITS: Final[dict[str, str]] = {
    "quality_loss": "quality_loss",
    "load_time_seconds": "s",
    "prefill_tokens_per_second": "tokens/s",
    "decode_tokens_per_second": "tokens/s",
    "latency_seconds": "s",
    "peak_ram_bytes": "bytes",
    "peak_vram_bytes": "bytes",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractionDatasetError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise InteractionDatasetError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InteractionDatasetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InteractionDatasetError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class InteractionMetric:
    name: str
    unit: str
    value: float

    def __post_init__(self) -> None:
        expected = _METRIC_UNITS.get(self.name)
        if expected is None or self.unit != expected:
            raise InteractionDatasetError(f"unknown or mismatched interaction metric: {self.name}")
        _finite(self.value, f"{self.name} value")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "unit": self.unit, "value": self.value}


@dataclass(frozen=True, slots=True)
class InteractionProvenance:
    benchmark_record_ids: tuple[str, ...]
    protocol_revision: str
    tool_revision: str
    command: tuple[str, ...]
    configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.benchmark_record_ids or any(
            not item.strip() for item in self.benchmark_record_ids
        ):
            raise InteractionDatasetError("interaction provenance requires benchmark record IDs")
        if self.benchmark_record_ids != tuple(sorted(set(self.benchmark_record_ids))):
            raise InteractionDatasetError("benchmark record IDs must be unique and canonical")
        _text(self.protocol_revision, "protocol revision")
        _text(self.tool_revision, "tool revision")
        if not self.command or any(not item.strip() for item in self.command):
            raise InteractionDatasetError("interaction provenance requires a command")
        try:
            _canonical(dict(self.configuration))
        except (TypeError, ValueError) as error:
            raise InteractionDatasetError(
                "interaction configuration must be canonical JSON"
            ) from error

    def to_record(self) -> dict[str, object]:
        return {
            "benchmark_record_ids": list(self.benchmark_record_ids),
            "protocol_revision": self.protocol_revision,
            "tool_revision": self.tool_revision,
            "command": list(self.command),
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True, slots=True)
class InteractionExample:
    """One measured or explicit terminal interaction cell."""

    kind: InteractionKind
    model_family: str
    model_revision: str
    root_state_id: str
    parent_state_id: str
    result_state_id: str
    source_artifact_digest: str
    corpus_id: str
    hardware_profile_id: str
    runtime_id: str
    seed: int
    mutation_ids: tuple[str, ...]
    ancestor_state_ids: tuple[str, ...]
    outcome: InteractionOutcome
    metrics: tuple[InteractionMetric, ...]
    provenance: InteractionProvenance
    topology_distance: int | None = None
    expected_additive_metrics: tuple[InteractionMetric, ...] = ()
    non_additivity_metrics: tuple[InteractionMetric, ...] = ()
    lineage_group_id: str | None = None
    reason: str | None = None
    schema_version: int = INTERACTION_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTION_DATASET_SCHEMA_VERSION:
            raise InteractionDatasetError("unsupported interaction example schema version")
        for label, value in (
            ("model family", self.model_family),
            ("model revision", self.model_revision),
            ("corpus ID", self.corpus_id),
            ("hardware profile ID", self.hardware_profile_id),
            ("runtime ID", self.runtime_id),
        ):
            _text(value, label)
        for label, value in (
            ("root state ID", self.root_state_id),
            ("parent state ID", self.parent_state_id),
            ("result state ID", self.result_state_id),
        ):
            if not value.startswith("state_"):
                raise InteractionDatasetError(f"{label} must be a state ID")
        _digest(self.source_artifact_digest, "source artifact digest")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise InteractionDatasetError("interaction seed must be unsigned")
        if self.topology_distance is not None and (
            isinstance(self.topology_distance, bool) or self.topology_distance < 0
        ):
            raise InteractionDatasetError("topology distance must be unsigned")
        if not self.mutation_ids or any(not item.strip() for item in self.mutation_ids):
            raise InteractionDatasetError("interaction examples require mutation IDs")
        if len(self.mutation_ids) != len(set(self.mutation_ids)):
            raise InteractionDatasetError("mutation IDs must be unique within a sequence")
        if self.ancestor_state_ids != tuple(sorted(set(self.ancestor_state_ids))):
            raise InteractionDatasetError("ancestor state IDs must be unique and canonical")
        names = tuple(item.name for item in self.metrics)
        expected_names = tuple(item.name for item in self.expected_additive_metrics)
        delta_names = tuple(item.name for item in self.non_additivity_metrics)
        for values, label in (
            (names, "observed metrics"),
            (expected_names, "expected metrics"),
            (delta_names, "non-additivity metrics"),
        ):
            if values != tuple(sorted(set(values))):
                raise InteractionDatasetError(f"{label} must be unique and canonical")
        if expected_names and expected_names != names:
            raise InteractionDatasetError("expected metrics must cover observed metric names")
        if delta_names and delta_names != names:
            raise InteractionDatasetError("non-additivity metrics must cover observed metric names")
        if self.outcome is InteractionOutcome.MEASURED:
            if not self.metrics:
                raise InteractionDatasetError("measured interactions require observed metrics")
            if self.reason is not None:
                raise InteractionDatasetError("measured interactions cannot carry a reason")
        else:
            if self.metrics or self.expected_additive_metrics or self.non_additivity_metrics:
                raise InteractionDatasetError("terminal interactions cannot carry partial metrics")
            _text(self.reason or "", "interaction outcome reason")
        if self.kind is InteractionKind.SINGLE and len(self.mutation_ids) != 1:
            raise InteractionDatasetError("single interactions require one mutation ID")
        if self.kind is InteractionKind.ORDERED_PAIR and len(self.mutation_ids) != 2:
            raise InteractionDatasetError("ordered pairs require exactly two mutation IDs")
        if self.kind is InteractionKind.CUMULATIVE and len(self.mutation_ids) < 2:
            raise InteractionDatasetError("cumulative interactions require at least two mutations")

    @property
    def sequence_length(self) -> int:
        return len(self.mutation_ids)

    @property
    def reconciled(self) -> bool:
        return bool(
            self.outcome is InteractionOutcome.MEASURED
            and self.expected_additive_metrics
            and self.non_additivity_metrics
        )

    @property
    def resolved_lineage_group_id(self) -> str:
        if self.lineage_group_id is not None:
            _text(self.lineage_group_id, "lineage group ID")
            return self.lineage_group_id
        digest = hashlib.sha256(
            _canonical(
                {
                    "model_revision": self.model_revision,
                    "source_artifact_digest": self.source_artifact_digest,
                    "root_state_id": self.root_state_id,
                }
            ).encode()
        ).hexdigest()
        return f"interaction_lineage_{digest}"

    @property
    def example_id(self) -> str:
        identity = self._identity_record()
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"interaction_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "root_state_id": self.root_state_id,
            "parent_state_id": self.parent_state_id,
            "result_state_id": self.result_state_id,
            "source_artifact_digest": self.source_artifact_digest,
            "corpus_id": self.corpus_id,
            "hardware_profile_id": self.hardware_profile_id,
            "runtime_id": self.runtime_id,
            "seed": self.seed,
            "topology_distance": self.topology_distance,
            "mutation_ids": list(self.mutation_ids),
            "ancestor_state_ids": list(self.ancestor_state_ids),
            "outcome": self.outcome.value,
            "metrics": [item.to_record() for item in self.metrics],
            "provenance": self.provenance.to_record(),
            "expected_additive_metrics": [
                item.to_record() for item in self.expected_additive_metrics
            ],
            "non_additivity_metrics": [item.to_record() for item in self.non_additivity_metrics],
            "lineage_group_id": self.resolved_lineage_group_id,
            "reason": self.reason,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["example_id"] = self.example_id
        record["sequence_length"] = self.sequence_length
        record["reconciled"] = self.reconciled
        return record


def build_cumulative_interaction(
    singles: Sequence[InteractionExample], cumulative: InteractionExample
) -> InteractionExample:
    """Attach additive expectation and non-additivity to cumulative evidence."""

    if len(singles) < 2 or any(item.kind is not InteractionKind.SINGLE for item in singles):
        raise InteractionDatasetError("cumulative reconciliation requires single mutations")
    if cumulative.kind not in (InteractionKind.ORDERED_PAIR, InteractionKind.CUMULATIVE):
        raise InteractionDatasetError("cumulative reconciliation requires sequence evidence")
    mutation_ids = tuple(item.mutation_ids[0] for item in singles)
    if cumulative.mutation_ids != mutation_ids:
        raise InteractionDatasetError("cumulative mutation order does not match source singles")
    context = (
        "model_revision",
        "root_state_id",
        "corpus_id",
        "hardware_profile_id",
        "runtime_id",
        "source_artifact_digest",
        "seed",
    )
    reference = singles[0]
    for name in context:
        if any(getattr(item, name) != getattr(reference, name) for item in singles) or getattr(
            reference, name
        ) != getattr(cumulative, name):
            raise InteractionDatasetError(f"cumulative context mismatch in {name}")
    if cumulative.outcome is not InteractionOutcome.MEASURED:
        return cumulative
    single_metrics = [{item.name: item for item in single.metrics} for single in singles]
    observed = {item.name: item for item in cumulative.metrics}
    if not single_metrics or any(
        set(metrics) != set(single_metrics[0]) for metrics in single_metrics
    ):
        raise InteractionDatasetError("cumulative single metric names do not reconcile")
    if set(single_metrics[0]) != set(observed):
        raise InteractionDatasetError("cumulative metric names do not reconcile")
    expected = tuple(
        InteractionMetric(
            name,
            _METRIC_UNITS[name],
            sum(metrics[name].value for metrics in single_metrics),
        )
        for name in sorted(observed)
    )
    non_additive = tuple(
        InteractionMetric(name, _METRIC_UNITS[name], observed[name].value - expected_item.value)
        for name, expected_item in ((item.name, item) for item in expected)
    )
    return replace(
        cumulative,
        expected_additive_metrics=expected,
        non_additivity_metrics=non_additive,
    )


def build_pairwise_interaction(
    first: InteractionExample,
    second: InteractionExample,
    cumulative: InteractionExample,
) -> InteractionExample:
    """Attach additive expectation and non-additivity to A→B or B→A evidence."""

    if cumulative.kind is not InteractionKind.ORDERED_PAIR:
        raise InteractionDatasetError("pairwise reconciliation requires an ordered-pair cell")
    return build_cumulative_interaction((first, second), cumulative)


@dataclass(frozen=True, slots=True)
class InteractionSplit:
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            set(self.train_group_ids),
            set(self.validation_group_ids),
            set(self.test_group_ids),
        )
        if any(not group for group in groups):
            raise InteractionDatasetError("every interaction split must contain groups")
        if any(
            len(group) != len(values)
            for group, values in zip(
                groups,
                (self.train_group_ids, self.validation_group_ids, self.test_group_ids),
                strict=True,
            )
        ):
            raise InteractionDatasetError("interaction split group IDs must be unique")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise InteractionDatasetError("interaction split groups must be disjoint")

    def to_record(self) -> dict[str, object]:
        return {
            "train_group_ids": list(self.train_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


@dataclass(frozen=True, slots=True)
class InteractionDataset:
    examples: tuple[InteractionExample, ...]
    split: InteractionSplit
    dataset_revision: str
    protocol_revision: str = INTERACTION_PROTOCOL_REVISION
    schema_version: int = INTERACTION_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTION_DATASET_SCHEMA_VERSION:
            raise InteractionDatasetError("unsupported interaction dataset schema version")
        if self.protocol_revision != INTERACTION_PROTOCOL_REVISION:
            raise InteractionDatasetError("unsupported interaction protocol revision")
        _text(self.dataset_revision, "dataset revision")
        ids = tuple(example.example_id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise InteractionDatasetError("interaction examples must be non-empty and unique")
        group_partition = {group: "train" for group in self.split.train_group_ids}
        group_partition.update({group: "validation" for group in self.split.validation_group_ids})
        group_partition.update({group: "test" for group in self.split.test_group_ids})
        by_partition: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
        for example in self.examples:
            group = example.resolved_lineage_group_id
            partition = group_partition.get(group)
            if partition is None:
                raise InteractionDatasetError("every example lineage must appear in one split")
            leakage_keys = self._leakage_keys(example)
            by_partition[partition].update(leakage_keys)
        partitions = (by_partition["train"], by_partition["validation"], by_partition["test"])
        if (
            partitions[0] & partitions[1]
            or partitions[0] & partitions[2]
            or partitions[1] & partitions[2]
        ):
            raise InteractionDatasetError(
                "model, corpus, state, or ancestor leakage crosses splits"
            )

    @staticmethod
    def _leakage_keys(example: InteractionExample) -> set[str]:
        return {
            f"model:{example.model_revision}",
            f"artifact:{example.source_artifact_digest}",
            f"corpus:{example.corpus_id}",
            f"root:{example.root_state_id}",
            *(f"ancestor:{state_id}" for state_id in example.ancestor_state_ids),
        }

    @property
    def dataset_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"interaction_dataset_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": self.protocol_revision,
            "dataset_revision": self.dataset_revision,
            "split": self.split.to_record(),
            "examples": [example.to_record() for example in self.examples],
        }

    def partition_for(self, example_id: str) -> str:
        example = next((item for item in self.examples if item.example_id == example_id), None)
        if example is None:
            raise InteractionDatasetError(f"unknown interaction example {example_id}")
        group = example.resolved_lineage_group_id
        if group in self.split.train_group_ids:
            return "train"
        if group in self.split.validation_group_ids:
            return "validation"
        return "test"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["dataset_id"] = self.dataset_id
        record["coverage"] = {
            "examples": len(self.examples),
            "outcomes": {
                outcome.value: sum(example.outcome is outcome for example in self.examples)
                for outcome in InteractionOutcome
            },
            "kinds": {
                kind.value: sum(example.kind is kind for example in self.examples)
                for kind in InteractionKind
            },
            "reconciled_pairs": sum(example.reconciled for example in self.examples),
        }
        return record
