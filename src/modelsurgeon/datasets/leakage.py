"""Deterministic leakage auditing across mutation dataset split strategies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.datasets.heldout_splits import HeldOutSplitManifest
from modelsurgeon.datasets.validation import validate_mutation_dataset
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import MutationExampleRecord
from modelsurgeon.features.schema import FeatureRecord
from modelsurgeon.surgery.contracts import MutationPrimitive

LEAKAGE_AUDIT_VERSION = "1"
SplitManifest: TypeAlias = GroupedSplitManifest | HeldOutSplitManifest


class DatasetLeakageError(ValueError):
    """Raised when a dataset cannot be certified leakage-free."""


class LeakageKind(StrEnum):
    MANIFEST_COVERAGE = "manifest_coverage"
    EXACT_CANDIDATE = "exact_candidate"
    NEAR_DUPLICATE_CANDIDATE = "near_duplicate_candidate"
    SHARED_COMPONENT = "shared_component"
    MODEL_ANCESTRY = "model_ancestry"
    TARGET_DERIVED_FEATURE = "target_derived_feature"


@dataclass(frozen=True, slots=True)
class ModelAncestry:
    """Explicit model lineage evidence used by the leakage auditor."""

    model_identifier: str
    ancestors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_identifier:
            raise DatasetLeakageError("model ancestry requires a model identifier")
        if self.ancestors != tuple(sorted(set(self.ancestors))):
            raise DatasetLeakageError("model ancestors must be unique and canonical")
        if any(not value for value in self.ancestors):
            raise DatasetLeakageError("model ancestors cannot contain blank identifiers")
        if self.model_identifier in self.ancestors:
            raise DatasetLeakageError("a model cannot list itself as an ancestor")

    def to_record(self) -> dict[str, object]:
        return {
            "model_identifier": self.model_identifier,
            "ancestors": list(self.ancestors),
        }


@dataclass(frozen=True, slots=True)
class LeakageAuditConfig:
    """Explicit non-inferential leakage policy inputs."""

    model_ancestry: tuple[ModelAncestry, ...] = ()
    target_feature_names: tuple[str, ...] = ()
    target_feature_extractors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifiers = tuple(item.model_identifier for item in self.model_ancestry)
        if identifiers != tuple(sorted(set(identifiers))):
            raise DatasetLeakageError("model ancestry entries must be unique and canonical")
        for label, values in (
            ("target feature names", self.target_feature_names),
            ("target feature extractors", self.target_feature_extractors),
        ):
            if values != tuple(sorted(set(values))) or any(not value for value in values):
                raise DatasetLeakageError(f"{label} must be unique canonical non-empty strings")

    def to_record(self) -> dict[str, object]:
        return {
            "model_ancestry": [item.to_record() for item in self.model_ancestry],
            "target_feature_names": list(self.target_feature_names),
            "target_feature_extractors": list(self.target_feature_extractors),
        }


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    kind: LeakageKind
    key: str
    partitions: tuple[SplitPartition, ...]
    example_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not self.key or not self.detail or not self.example_ids:
            raise DatasetLeakageError("leakage findings require key, examples, and detail")
        if self.partitions != tuple(sorted(set(self.partitions), key=lambda item: item.value)):
            raise DatasetLeakageError("finding partitions must be unique and canonical")
        if self.example_ids != tuple(sorted(set(self.example_ids))):
            raise DatasetLeakageError("finding example IDs must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "partitions": [item.value for item in self.partitions],
            "example_ids": list(self.example_ids),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    manifest_kind: str
    example_count: int
    findings: tuple[LeakageFinding, ...]
    config: LeakageAuditConfig
    version: str = LEAKAGE_AUDIT_VERSION

    def __post_init__(self) -> None:
        if self.version != LEAKAGE_AUDIT_VERSION:
            raise DatasetLeakageError(f"unsupported leakage audit version {self.version}")
        if not self.manifest_kind or self.example_count < 0:
            raise DatasetLeakageError("leakage audit report metadata is invalid")
        ordered = tuple(
            sorted(
                self.findings,
                key=lambda item: (
                    item.kind.value,
                    item.key,
                    tuple(partition.value for partition in item.partitions),
                    item.example_ids,
                ),
            )
        )
        if self.findings != ordered:
            raise DatasetLeakageError("leakage findings must use canonical ordering")

    @property
    def clean(self) -> bool:
        return not self.findings

    def require_clean(self) -> None:
        """Fail closed before training or downstream matrix construction."""

        if self.clean:
            return
        first = self.findings[0]
        raise DatasetLeakageError(
            f"dataset leakage audit failed: {first.kind.value}: {first.key}: {first.detail}"
        )

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "manifest_kind": self.manifest_kind,
            "example_count": self.example_count,
            "clean": self.clean,
            "config": self.config.to_record(),
            "findings": [item.to_record() for item in self.findings],
        }


def _manifest_kind(manifest: SplitManifest) -> str:
    if isinstance(manifest, GroupedSplitManifest):
        return f"grouped:{manifest.mode.value}"
    return f"heldout:{manifest.mode.value}"


def _partition_map(manifest: SplitManifest) -> dict[str, SplitPartition]:
    result: dict[str, SplitPartition] = {}
    for group in manifest.groups:
        for example_id in group.example_ids:
            result[example_id] = group.partition
    return result


def _partitions_for(
    example_ids: tuple[str, ...],
    partitions: dict[str, SplitPartition],
) -> tuple[SplitPartition, ...]:
    return tuple(
        sorted(
            {partitions[example_id] for example_id in example_ids if example_id in partitions},
            key=lambda item: item.value,
        )
    )


def _coverage_findings(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    example_ids = {example.example_id for example in examples}
    missing = tuple(sorted(example_ids - set(partitions)))
    if missing:
        findings.append(
            LeakageFinding(
                LeakageKind.MANIFEST_COVERAGE,
                "dataset_examples_missing_from_manifest",
                (),
                missing,
                "every validated example must have exactly one split assignment",
            )
        )
    extra = tuple(sorted(set(partitions) - example_ids))
    if extra:
        findings.append(
            LeakageFinding(
                LeakageKind.MANIFEST_COVERAGE,
                "manifest_examples_missing_from_dataset",
                tuple(sorted(set(partitions.values()), key=lambda item: item.value)),
                extra,
                "split manifest references examples absent from the audited dataset",
            )
        )
    return findings


def _numeric_placeholder(value: MutationPrimitive) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return {"numeric_parameter": True}
    return value


def _candidate_exact_key(example: MutationExampleRecord) -> str:
    return f"{example.model.identifier}:{example.mutation_id}"


def _candidate_near_key(example: MutationExampleRecord) -> str:
    request = example.mutation.plan.request
    payload = {
        "model_identifier": example.model.identifier,
        "kind": request.kind.value,
        "targets": [str(item) for item in request.targets],
        "parameters": [
            [key, _numeric_placeholder(value)] for key, value in request.parameters
        ],
    }
    digest = hashlib.sha256(canonical_identity_json(payload).encode("utf-8")).hexdigest()
    return f"near:{digest}"


def _cross_partition_groups(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
    key_fn,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        if example.example_id in partitions:
            grouped.setdefault(key_fn(example), []).append(example.example_id)
    return {
        key: tuple(sorted(ids))
        for key, ids in grouped.items()
        if len(_partitions_for(tuple(ids), partitions)) > 1
    }


def _candidate_findings(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    exact = _cross_partition_groups(examples, partitions, _candidate_exact_key)
    exact_ids = {example_id for ids in exact.values() for example_id in ids}
    for key, ids in exact.items():
        findings.append(
            LeakageFinding(
                LeakageKind.EXACT_CANDIDATE,
                key,
                _partitions_for(ids, partitions),
                ids,
                "the same model/canonical mutation candidate appears across partitions",
            )
        )

    near = _cross_partition_groups(examples, partitions, _candidate_near_key)
    by_id = {example.example_id: example for example in examples}
    for key, ids in near.items():
        mutation_ids = {by_id[example_id].mutation_id for example_id in ids}
        if len(mutation_ids) < 2:
            continue
        if set(ids).issubset(exact_ids):
            continue
        findings.append(
            LeakageFinding(
                LeakageKind.NEAR_DUPLICATE_CANDIDATE,
                key,
                _partitions_for(ids, partitions),
                ids,
                "structurally identical candidates differ only in numeric parameter values",
            )
        )
    return findings


def _component_findings(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
) -> list[LeakageFinding]:
    grouped: dict[str, list[str]] = {}
    for example in examples:
        if example.example_id not in partitions:
            continue
        for component in example.components:
            key = f"{example.model.identifier}:{component}"
            grouped.setdefault(key, []).append(example.example_id)
    findings: list[LeakageFinding] = []
    for key, raw_ids in grouped.items():
        ids = tuple(sorted(set(raw_ids)))
        found_partitions = _partitions_for(ids, partitions)
        if len(found_partitions) <= 1:
            continue
        findings.append(
            LeakageFinding(
                LeakageKind.SHARED_COMPONENT,
                key,
                found_partitions,
                ids,
                "the same canonical component of one model appears across partitions",
            )
        )
    return findings


def _ancestry_closure(config: LeakageAuditConfig) -> dict[str, frozenset[str]]:
    direct = {item.model_identifier: item.ancestors for item in config.model_ancestry}
    memo: dict[str, frozenset[str]] = {}

    def visit(model: str, stack: frozenset[str]) -> frozenset[str]:
        cached = memo.get(model)
        if cached is not None:
            return cached
        if model in stack:
            raise DatasetLeakageError(f"model ancestry contains a cycle at {model!r}")
        values = {model}
        next_stack = stack | {model}
        for ancestor in direct.get(model, ()):
            values.update(visit(ancestor, next_stack))
        result = frozenset(values)
        memo[model] = result
        return result

    for model in direct:
        visit(model, frozenset())
    return memo


def _ancestry_findings(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
    config: LeakageAuditConfig,
) -> list[LeakageFinding]:
    if not config.model_ancestry:
        return []
    closure = _ancestry_closure(config)
    examples_by_model: dict[str, list[str]] = {}
    for example in examples:
        if example.example_id in partitions:
            examples_by_model.setdefault(example.model.identifier, []).append(example.example_id)
    models = sorted(examples_by_model)
    findings: list[LeakageFinding] = []
    for left_index, left in enumerate(models):
        left_lineage = closure.get(left, frozenset({left}))
        for right in models[left_index + 1 :]:
            right_lineage = closure.get(right, frozenset({right}))
            shared = tuple(sorted(left_lineage & right_lineage))
            if not shared:
                continue
            ids = tuple(sorted((*examples_by_model[left], *examples_by_model[right])))
            found_partitions = _partitions_for(ids, partitions)
            if len(found_partitions) <= 1:
                continue
            key = f"lineage:{left}|{right}|{','.join(shared)}"
            findings.append(
                LeakageFinding(
                    LeakageKind.MODEL_ANCESTRY,
                    key,
                    found_partitions,
                    ids,
                    "models sharing explicit ancestry appear on different sides of the split",
                )
            )
    return findings


def _metadata(feature: FeatureRecord) -> dict[str, MutationPrimitive]:
    return dict(feature.metadata)


def _target_derived_reason(
    feature: FeatureRecord,
    config: LeakageAuditConfig,
) -> str | None:
    if feature.name in config.target_feature_names:
        return "feature name is explicitly classified as target-derived"
    if feature.extractor in config.target_feature_extractors:
        return "feature extractor is explicitly classified as target-derived"
    metadata = _metadata(feature)
    if metadata.get("target_derived") is True:
        return "feature metadata marks the value as target-derived"
    phase = metadata.get("source_phase")
    if isinstance(phase, str) and phase in {"post_mutation", "delta", "target"}:
        return f"feature metadata declares target-bearing source phase {phase!r}"
    return None


def _target_feature_findings(
    examples: tuple[MutationExampleRecord, ...],
    partitions: dict[str, SplitPartition],
    config: LeakageAuditConfig,
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    for example in examples:
        partition = partitions.get(example.example_id)
        for feature in example.pre_mutation_features:
            reason = _target_derived_reason(feature, config)
            if reason is None:
                continue
            identity = (
                f"{feature.component_id}:{feature.name}:"
                f"{feature.extractor}:{feature.extractor_version}"
            )
            findings.append(
                LeakageFinding(
                    LeakageKind.TARGET_DERIVED_FEATURE,
                    identity,
                    () if partition is None else (partition,),
                    (example.example_id,),
                    reason,
                )
            )
    return findings


def audit_dataset_leakage(
    examples: tuple[MutationExampleRecord, ...],
    manifest: SplitManifest,
    config: LeakageAuditConfig | None = None,
) -> LeakageAuditReport:
    """Audit one validated dataset/split manifest and return every leakage finding."""

    if not examples:
        raise DatasetLeakageError("leakage auditing requires at least one mutation example")
    validation = validate_mutation_dataset(examples)
    if not validation.valid:
        first = validation.issues[0]
        raise DatasetLeakageError(
            f"dataset validation failed at record {first.record_index} "
            f"{first.path}: {first.code}: {first.detail}"
        )
    resolved = config or LeakageAuditConfig()
    partitions = _partition_map(manifest)
    findings = [
        *_coverage_findings(examples, partitions),
        *_candidate_findings(examples, partitions),
        *_component_findings(examples, partitions),
        *_ancestry_findings(examples, partitions, resolved),
        *_target_feature_findings(examples, partitions, resolved),
    ]
    ordered = tuple(
        sorted(
            findings,
            key=lambda item: (
                item.kind.value,
                item.key,
                tuple(partition.value for partition in item.partitions),
                item.example_ids,
            ),
        )
    )
    return LeakageAuditReport(
        _manifest_kind(manifest),
        len(examples),
        ordered,
        resolved,
    )
