"""Leakage-safe construction of supervised mutation examples from experiment records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import (
    ExperimentOutcomeKind,
    ExperimentRecord,
    MetricState,
    MutationExampleRecord,
)
from modelsurgeon.features.cache import FeaturePartition
from modelsurgeon.features.schema import FeatureRecord

DATASET_BUILDER_VERSION = "1"


class DatasetBuildError(ValueError):
    """Raised when experiment/features cannot form a trustworthy training example."""


class FeatureInputRevisionSource(StrEnum):
    MANIFEST_ID = "manifest_id"
    DATASET_REVISION = "dataset_revision"


class DeltaTargetPolicy(StrEnum):
    REQUIRE_MEASURED = "require_measured"
    PRESERVE_MISSING = "preserve_missing"


class DatasetBuildExclusionReason(StrEnum):
    OUTCOME_POLICY = "outcome_policy"
    NO_MEASURED_DELTA = "no_measured_delta"
    DUPLICATE_RETRY = "duplicate_retry"


@dataclass(frozen=True, slots=True)
class MutationExampleBuildPolicy:
    allowed_outcomes: tuple[ExperimentOutcomeKind, ...] = (
        ExperimentOutcomeKind.SUCCEEDED,
        ExperimentOutcomeKind.REJECTED,
    )
    delta_target_policy: DeltaTargetPolicy = DeltaTargetPolicy.REQUIRE_MEASURED
    feature_input_revision_source: FeatureInputRevisionSource = (
        FeatureInputRevisionSource.MANIFEST_ID
    )

    def __post_init__(self) -> None:
        if not self.allowed_outcomes:
            raise DatasetBuildError("dataset build policy must allow at least one outcome")
        if len(self.allowed_outcomes) != len(set(self.allowed_outcomes)):
            raise DatasetBuildError("allowed outcomes must be unique")


@dataclass(frozen=True, slots=True)
class ExperimentFeatureJoin:
    experiment: ExperimentRecord
    pre_mutation_partitions: tuple[FeaturePartition, ...]

    def __post_init__(self) -> None:
        if not self.pre_mutation_partitions:
            raise DatasetBuildError("supervised examples require pre-mutation feature partitions")


@dataclass(frozen=True, slots=True)
class DatasetBuildExclusion:
    experiment_id: str
    mutation_id: str
    reason: DatasetBuildExclusionReason
    detail: str

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.mutation_id or not self.detail:
            raise DatasetBuildError("dataset build exclusions require complete identity and detail")

    def to_record(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "mutation_id": self.mutation_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MutationExampleBuildReport:
    version: str
    examples: tuple[MutationExampleRecord, ...]
    exclusions: tuple[DatasetBuildExclusion, ...]
    processed_count: int

    def __post_init__(self) -> None:
        if self.version != DATASET_BUILDER_VERSION:
            raise DatasetBuildError(f"unsupported dataset builder version {self.version}")
        if self.processed_count < 0:
            raise DatasetBuildError("dataset builder processed count cannot be negative")
        if len(self.examples) + len(self.exclusions) != self.processed_count:
            raise DatasetBuildError("dataset builder report counts are inconsistent")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "processed_count": self.processed_count,
            "example_count": len(self.examples),
            "exclusion_count": len(self.exclusions),
            "example_ids": [item.example_id for item in self.examples],
            "exclusions": [item.to_record() for item in self.exclusions],
        }


def _feature_identity(record: FeatureRecord) -> tuple[str, str, str, str]:
    return (
        str(record.component_id),
        record.name,
        record.extractor,
        record.extractor_version,
    )


def _expected_input_revision(
    record: ExperimentRecord,
    source: FeatureInputRevisionSource,
) -> str:
    if source is FeatureInputRevisionSource.MANIFEST_ID:
        return record.dataset.manifest_id
    return record.dataset.revision


def _validate_sample_context(record: ExperimentRecord, feature: FeatureRecord) -> None:
    context = feature.sample_context
    if context is None:
        return
    actual = (
        context.dataset,
        context.revision,
        context.split,
        context.tokenizer,
        context.tokenizer_revision,
    )
    expected = (
        record.dataset.identifier,
        record.dataset.revision,
        record.dataset.split,
        record.dataset.tokenizer,
        record.dataset.tokenizer_revision,
    )
    if actual != expected:
        raise DatasetBuildError(
            "pre-mutation feature sample context does not match experiment dataset/tokenizer"
        )


def _validated_features(
    join: ExperimentFeatureJoin,
    policy: MutationExampleBuildPolicy,
) -> tuple[tuple[FeatureRecord, ...], str]:
    experiment = join.experiment
    expected_input_revision = _expected_input_revision(
        experiment,
        policy.feature_input_revision_source,
    )
    snapshot_parts: list[dict[str, object]] = []
    features: list[FeatureRecord] = []
    for partition in join.pre_mutation_partitions:
        if partition.key.model_revision != experiment.model.revision:
            raise DatasetBuildError(
                "feature partition model revision is not the experiment pre-mutation revision"
            )
        if partition.key.input_revision != expected_input_revision:
            raise DatasetBuildError(
                "feature partition input revision does not match the configured experiment input"
            )
        snapshot_parts.append(
            {
                "key": partition.key.to_record(),
                "records_sha256": partition.records_sha256,
            }
        )
        for feature in partition.records:
            _validate_sample_context(experiment, feature)
            features.append(feature)

    ordered = tuple(sorted(features, key=_feature_identity))
    identities = tuple(_feature_identity(item) for item in ordered)
    if len(identities) != len(set(identities)):
        raise DatasetBuildError("pre-mutation feature identities are duplicated across partitions")

    canonical_snapshot = canonical_identity_json(
        sorted(
            snapshot_parts,
            key=lambda item: canonical_identity_json(item),
        )
    )
    digest = hashlib.sha256(canonical_snapshot.encode("utf-8")).hexdigest()
    return ordered, digest


def _derive_example_id(record: ExperimentRecord, feature_snapshot_digest: str) -> str:
    payload = {
        "builder_version": DATASET_BUILDER_VERSION,
        "experiment_id": record.experiment_id,
        "run_id": record.run_id,
        "mutation_id": record.mutation.mutation_id,
        "model_revision": record.model.revision,
        "dataset_manifest_id": record.dataset.manifest_id,
        "config_digest": record.versions.config_digest,
        "feature_snapshot_digest": feature_snapshot_digest,
    }
    encoded = canonical_identity_json(payload).encode("utf-8")
    return f"example_{hashlib.sha256(encoded).hexdigest()}"


def _build_one(
    join: ExperimentFeatureJoin,
    policy: MutationExampleBuildPolicy,
) -> tuple[MutationExampleRecord | None, DatasetBuildExclusion | None]:
    record = join.experiment
    if record.outcome.kind not in policy.allowed_outcomes:
        return None, DatasetBuildExclusion(
            record.experiment_id,
            record.mutation.mutation_id,
            DatasetBuildExclusionReason.OUTCOME_POLICY,
            f"outcome {record.outcome.kind.value} is excluded by dataset build policy",
        )

    if policy.delta_target_policy is DeltaTargetPolicy.REQUIRE_MEASURED and not any(
        metric.state is MetricState.MEASURED for metric in record.delta_metrics
    ):
        return None, DatasetBuildExclusion(
            record.experiment_id,
            record.mutation.mutation_id,
            DatasetBuildExclusionReason.NO_MEASURED_DELTA,
            "no measured delta metric is available as a supervised target",
        )

    features, feature_snapshot_digest = _validated_features(join, policy)
    example = MutationExampleRecord(
        example_id=_derive_example_id(record, feature_snapshot_digest),
        experiment_id=record.experiment_id,
        model=record.model,
        dataset=record.dataset,
        components=record.components,
        mutation=record.mutation,
        pre_mutation_features=features,
        baseline_metrics=record.baseline_metrics,
        post_metrics=record.post_metrics,
        delta_metrics=record.delta_metrics,
        outcome=record.outcome,
        hardware=record.hardware,
        versions=record.versions,
        seeds=record.seeds,
        timings=record.timings,
        quantization_control=record.quantization_control,
    )
    return example, None


def build_mutation_example(
    join: ExperimentFeatureJoin,
    policy: MutationExampleBuildPolicy | None = None,
) -> MutationExampleRecord | None:
    """Build one canonical example, returning ``None`` only for an explicit policy exclusion."""

    example, _ = _build_one(join, policy or MutationExampleBuildPolicy())
    return example


def build_mutation_examples(
    joins: Iterable[ExperimentFeatureJoin],
    policy: MutationExampleBuildPolicy | None = None,
) -> MutationExampleBuildReport:
    """Build canonical examples and collapse exact logical retries without hiding conflicts."""

    resolved = policy or MutationExampleBuildPolicy()
    examples: list[MutationExampleRecord] = []
    exclusions: list[DatasetBuildExclusion] = []
    by_id: dict[str, MutationExampleRecord] = {}
    processed = 0
    for join in joins:
        processed += 1
        example, exclusion = _build_one(join, resolved)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue
        if example is None:
            raise DatasetBuildError("dataset builder produced neither an example nor exclusion")
        previous = by_id.get(example.example_id)
        if previous is None:
            by_id[example.example_id] = example
            examples.append(example)
            continue
        if previous.to_record() != example.to_record():
            raise DatasetBuildError(
                f"logical retry {example.example_id} produced conflicting supervised content"
            )
        exclusions.append(
            DatasetBuildExclusion(
                example.experiment_id,
                example.mutation_id,
                DatasetBuildExclusionReason.DUPLICATE_RETRY,
                "exact logical retry duplicates an already-built mutation example",
            )
        )

    return MutationExampleBuildReport(
        DATASET_BUILDER_VERSION,
        tuple(examples),
        tuple(exclusions),
        processed,
    )
