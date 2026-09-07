"""Matched no-repair and bounded-repair dataset contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.surgery.repair_outcome import (
    RepairBudget,
    RepairCost,
    RepairMethod,
    RepairMetricEvidence,
    RepairOutcomeStatus,
    RepairResult,
)

REPAIR_DATASET_SCHEMA_VERSION = 1
REPAIR_DATASET_PROTOCOL_REVISION = "repair-dataset-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RepairDatasetError(ValueError):
    """Raised when a repair dataset would lose a control, outcome, or lineage."""


class RepairDatasetPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepairDatasetError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise RepairDatasetError(f"{label} must be a lowercase SHA-256")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairDatasetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RepairDatasetError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RepairDatasetProvenance:
    """Reproduction identity for one paired repair measurement."""

    record_id: str
    source_revision: str
    protocol_revision: str
    tool_revision: str
    command: tuple[str, ...]
    configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        for label, value in (
            ("repair dataset record ID", self.record_id),
            ("source revision", self.source_revision),
            ("protocol revision", self.protocol_revision),
            ("tool revision", self.tool_revision),
        ):
            _text(value, label)
        if not self.command or any(not item.strip() for item in self.command):
            raise RepairDatasetError("repair dataset provenance requires a command")
        try:
            _canonical(dict(self.configuration))
        except (TypeError, ValueError) as error:
            raise RepairDatasetError(
                "repair dataset configuration must be canonical JSON"
            ) from error

    def to_record(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_revision": self.source_revision,
            "protocol_revision": self.protocol_revision,
            "tool_revision": self.tool_revision,
            "command": list(self.command),
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True, slots=True)
class RepairBaseline:
    """The no-repair control measured against the same damaged candidate."""

    parent_candidate_id: str
    source_artifact_digest: str
    candidate_artifact_digest: str
    budget: RepairBudget
    cost: RepairCost
    heldout_evidence: tuple[RepairMetricEvidence, ...]
    reason: str = "no_repair_baseline"

    def __post_init__(self) -> None:
        _text(self.parent_candidate_id, "baseline parent candidate ID")
        _digest(self.source_artifact_digest, "baseline source artifact digest")
        _digest(self.candidate_artifact_digest, "baseline candidate artifact digest")
        if self.source_artifact_digest == self.candidate_artifact_digest:
            raise RepairDatasetError("baseline source and candidate artifacts must differ")
        self.budget.enforce(self.cost)
        if self.cost != RepairCost(0, 0, 0.0, 0.0, 0.0, 0):
            raise RepairDatasetError("no-repair baselines cannot incur repair cost")
        _text(self.reason, "baseline reason")
        if not self.heldout_evidence:
            raise RepairDatasetError("no-repair baselines require held-out evidence")

    @property
    def outcome(self) -> RepairOutcomeStatus:
        return RepairOutcomeStatus.REJECTED

    @property
    def outcome_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"repair_baseline_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "parent_candidate_id": self.parent_candidate_id,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "budget": self.budget.to_record(),
            "cost": self.cost.to_record(),
            "heldout_evidence": [item.to_record() for item in self.heldout_evidence],
            "reason": self.reason,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["outcome_id"] = self.outcome_id
        return record


@dataclass(frozen=True, slots=True)
class RepairDatasetExample:
    """One repair method/budget paired with its exact no-repair control."""

    model_family: str
    model_revision: str
    mutation_state_id: str
    mutation_kind: str
    mutation_severity: float
    source_artifact_digest: str
    candidate_artifact_digest: str
    dataset_revision: str
    tokenizer_digest: str
    hardware_profile_id: str
    budget_name: str
    seed: int
    lineage_group_id: str
    no_repair: RepairBaseline
    repair: RepairResult
    provenance: RepairDatasetProvenance
    schema_version: int = REPAIR_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_DATASET_SCHEMA_VERSION:
            raise RepairDatasetError("unsupported repair dataset schema version")
        for label, value in (
            ("model family", self.model_family),
            ("model revision", self.model_revision),
            ("mutation state ID", self.mutation_state_id),
            ("mutation kind", self.mutation_kind),
            ("dataset revision", self.dataset_revision),
            ("hardware profile ID", self.hardware_profile_id),
            ("budget name", self.budget_name),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        _digest(self.source_artifact_digest, "source artifact digest")
        _digest(self.candidate_artifact_digest, "candidate artifact digest")
        _digest(self.tokenizer_digest, "tokenizer digest")
        _finite(self.mutation_severity, "mutation severity")
        if self.mutation_severity < 0:
            raise RepairDatasetError("mutation severity cannot be negative")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise RepairDatasetError("repair dataset seed must be an unsigned 64-bit integer")
        if self.no_repair.parent_candidate_id != self.repair.request.parent_candidate_id:
            raise RepairDatasetError("repair and no-repair controls must share a candidate ID")
        if self.no_repair.source_artifact_digest != self.source_artifact_digest:
            raise RepairDatasetError("no-repair source artifact does not match the example")
        if self.no_repair.candidate_artifact_digest != self.candidate_artifact_digest:
            raise RepairDatasetError("no-repair candidate artifact does not match the example")
        request = self.repair.request
        if request.method is RepairMethod.NO_REPAIR:
            raise RepairDatasetError("repair examples must use a repair method")
        if request.budget != self.no_repair.budget:
            raise RepairDatasetError("repair and no-repair budgets must match")
        lineage = self.repair.lineage
        if lineage.source_artifact_digest != self.source_artifact_digest:
            raise RepairDatasetError("repair source lineage does not match the example")
        if lineage.parent_artifact_digest != self.candidate_artifact_digest:
            raise RepairDatasetError("repair parent lineage does not match the damaged candidate")
        if lineage.candidate_artifact_digest == self.candidate_artifact_digest:
            raise RepairDatasetError("repair child artifact must differ from the damaged candidate")
        for identity, label in (
            (request.source, "source"),
            (request.candidate, "candidate"),
        ):
            if identity.tokenizer_digest != self.tokenizer_digest:
                raise RepairDatasetError(f"repair {label} tokenizer does not match the example")
            if identity.data_revision != self.dataset_revision:
                raise RepairDatasetError(f"repair {label} data revision does not match the example")
        evidence_groups = {
            group
            for evidence in (*self.no_repair.heldout_evidence, *self.repair.heldout_evidence)
            for group in evidence.heldout_group_ids
        }
        if not evidence_groups:
            raise RepairDatasetError("repair examples require held-out evaluator groups")
        baseline_groups = {
            group
            for evidence in self.no_repair.heldout_evidence
            for group in evidence.heldout_group_ids
        }
        repair_groups = {
            group
            for evidence in self.repair.heldout_evidence
            for group in evidence.heldout_group_ids
        }
        if repair_groups and repair_groups != baseline_groups:
            raise RepairDatasetError(
                "repair and no-repair evaluations must use the same held-out groups"
            )

    @property
    def example_id(self) -> str:
        identity = {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "mutation_state_id": self.mutation_state_id,
            "mutation_kind": self.mutation_kind,
            "mutation_severity": self.mutation_severity,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "dataset_revision": self.dataset_revision,
            "tokenizer_digest": self.tokenizer_digest,
            "hardware_profile_id": self.hardware_profile_id,
            "budget_name": self.budget_name,
            "seed": self.seed,
            "lineage_group_id": self.lineage_group_id,
            "no_repair": self.no_repair.to_record(),
            "repair": self.repair.to_record(),
            "provenance": self.provenance.to_record(),
        }
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"repair_example_{digest}"

    @property
    def outcome(self) -> RepairOutcomeStatus:
        return self.repair.status

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "mutation_state_id": self.mutation_state_id,
            "mutation_kind": self.mutation_kind,
            "mutation_severity": self.mutation_severity,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "dataset_revision": self.dataset_revision,
            "tokenizer_digest": self.tokenizer_digest,
            "hardware_profile_id": self.hardware_profile_id,
            "budget_name": self.budget_name,
            "seed": self.seed,
            "lineage_group_id": self.lineage_group_id,
            "no_repair": self.no_repair.to_record(),
            "repair": self.repair.to_record(),
            "provenance": self.provenance.to_record(),
        }


@dataclass(frozen=True, slots=True)
class RepairDatasetSplit:
    """Disjoint lineage groups for train, validation, and test."""

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
            raise RepairDatasetError("every repair dataset split must contain groups")
        for values, group in zip(
            (self.train_group_ids, self.validation_group_ids, self.test_group_ids),
            groups,
            strict=True,
        ):
            if values != tuple(sorted(group)):
                raise RepairDatasetError("repair split group IDs must be unique and canonical")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise RepairDatasetError("repair lineage groups leak across splits")

    def to_record(self) -> dict[str, object]:
        return {
            "train_group_ids": list(self.train_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


@dataclass(frozen=True, slots=True)
class RepairDatasetConfig:
    """Coverage requirements for a published repair dataset."""

    dataset_revision: str
    required_budget_names: tuple[str, ...] = ("medium", "short", "zero")
    minimum_model_families: int = 2
    minimum_mutation_kinds: int = 2
    minimum_seeds: int = 2

    def __post_init__(self) -> None:
        _text(self.dataset_revision, "repair dataset revision")
        if self.required_budget_names != tuple(sorted(set(self.required_budget_names))):
            raise RepairDatasetError("required budget names must be unique and canonical")
        if any(not name for name in self.required_budget_names):
            raise RepairDatasetError("required budget names cannot be blank")
        for label, value in (
            ("minimum model families", self.minimum_model_families),
            ("minimum mutation kinds", self.minimum_mutation_kinds),
            ("minimum seeds", self.minimum_seeds),
        ):
            if isinstance(value, bool) or value <= 0:
                raise RepairDatasetError(f"{label} must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "dataset_revision": self.dataset_revision,
            "required_budget_names": list(self.required_budget_names),
            "minimum_model_families": self.minimum_model_families,
            "minimum_mutation_kinds": self.minimum_mutation_kinds,
            "minimum_seeds": self.minimum_seeds,
        }


@dataclass(frozen=True, slots=True)
class RepairDataset:
    """Validated paired repair outcomes with complete cost and provenance."""

    examples: tuple[RepairDatasetExample, ...]
    split: RepairDatasetSplit
    config: RepairDatasetConfig
    schema_version: int = REPAIR_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_DATASET_SCHEMA_VERSION:
            raise RepairDatasetError("unsupported repair dataset schema version")
        if not self.examples:
            raise RepairDatasetError("repair dataset requires examples")
        ids = tuple(example.example_id for example in self.examples)
        if len(ids) != len(set(ids)):
            raise RepairDatasetError("repair dataset example IDs must be unique")
        if any(
            example.dataset_revision != self.config.dataset_revision
            for example in self.examples
        ):
            raise RepairDatasetError("repair examples must use the configured dataset revision")
        split_groups = (
            set(self.split.train_group_ids)
            | set(self.split.validation_group_ids)
            | set(self.split.test_group_ids)
        )
        if any(example.lineage_group_id not in split_groups for example in self.examples):
            raise RepairDatasetError("every repair example lineage must appear in one split")
        families = {example.model_family for example in self.examples}
        mutation_kinds = {example.mutation_kind for example in self.examples}
        seeds = {example.seed for example in self.examples}
        budgets = {example.budget_name for example in self.examples}
        if len(families) < self.config.minimum_model_families:
            raise RepairDatasetError("repair dataset does not cover enough model families")
        if len(mutation_kinds) < self.config.minimum_mutation_kinds:
            raise RepairDatasetError("repair dataset does not cover enough mutation kinds")
        if len(seeds) < self.config.minimum_seeds:
            raise RepairDatasetError("repair dataset does not cover enough seeds")
        missing = set(self.config.required_budget_names) - budgets
        if missing:
            raise RepairDatasetError(
                f"repair dataset is missing required budgets: {sorted(missing)}"
            )

    @property
    def dataset_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"repair_dataset_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_DATASET_PROTOCOL_REVISION,
            "config": self.config.to_record(),
            "split": self.split.to_record(),
            "examples": [item.to_record() for item in self.examples],
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["dataset_id"] = self.dataset_id
        record["coverage"] = {
            "examples": len(self.examples),
            "model_families": len({item.model_family for item in self.examples}),
            "mutation_kinds": len({item.mutation_kind for item in self.examples}),
            "seeds": len({item.seed for item in self.examples}),
            "budgets": sorted({item.budget_name for item in self.examples}),
            "outcomes": {
                outcome.value: sum(item.outcome is outcome for item in self.examples)
                for outcome in RepairOutcomeStatus
            },
        }
        return record


def build_repair_dataset(
    examples: tuple[RepairDatasetExample, ...],
    split: RepairDatasetSplit,
    config: RepairDatasetConfig,
) -> RepairDataset:
    """Validate and publish a repair dataset without dropping terminal outcomes."""

    return RepairDataset(examples, split, config)
