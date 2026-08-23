"""Tests for leakage-safe supervised mutation example construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets.builder import (
    DatasetBuildError,
    DatasetBuildExclusionReason,
    DeltaTargetPolicy,
    ExperimentFeatureJoin,
    FeatureInputRevisionSource,
    MutationExampleBuildPolicy,
    build_mutation_example,
    build_mutation_examples,
)
from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    HardwareInventory,
    MemoryInventory,
    MetricObservation,
    MetricState,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
)
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionKey
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
    MutationRunRecord,
)


def _component() -> ComponentId:
    return ComponentId.parse("model.layers.0.mlp.up_proj")


def _model() -> ModelTarget:
    return ModelTarget("tiny/model", "model-rev-1", "llama", "safetensors", 128)


def _dataset() -> DatasetTarget:
    return DatasetTarget(
        "tiny-dataset",
        "dataset-rev-1",
        "validation",
        "manifest-1",
        "tiny-tokenizer",
        "tokenizer-rev-1",
    )


def _hardware() -> HardwareInventory:
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 4),
        MemoryInventory(1024, 512),
        DiskInventory("/tmp", 4096, 2048),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _mutation() -> MutationRunRecord:
    component = _component()
    delta = MutationDelta(parameters=-8, flops=-16, memory_bytes=-32, storage_bytes=0)
    plan = MutationPlan(
        MutationRequest(MutationKind.MASK, (component,)),
        (component,),
        (),
        delta,
    )
    return MutationRunRecord(
        plan,
        MutationProvenance("model-rev-1", "tool-rev", "/private/source"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )


def _record(
    *,
    attempt_id: str = "attempt-1",
    outcome: ExperimentOutcomeKind = ExperimentOutcomeKind.SUCCEEDED,
    delta_metrics: tuple[MetricObservation, ...] | None = None,
) -> ExperimentRecord:
    mutation = _mutation()
    reason = None if outcome is ExperimentOutcomeKind.SUCCEEDED else f"{outcome.value} fixture"
    return ExperimentRecord(
        "run-1",
        "experiment-1",
        attempt_id,
        _model(),
        _dataset(),
        mutation.plan.affected_components,
        mutation,
        (
            MetricObservation("loss", MetricState.MEASURED, 1.0),
            MetricObservation("top_k", MetricState.SKIPPED, reason="not configured"),
        ),
        (MetricObservation("loss", MetricState.MEASURED, 1.25),),
        delta_metrics
        if delta_metrics is not None
        else (MetricObservation("loss_delta", MetricState.MEASURED, 0.25),),
        ExperimentOutcome(outcome, reason),
        _hardware(),
        VersionContext("tool-rev", "config-digest", "eval-v1", 1, 1),
        SeedContext(1, 2, 3),
    )


def _sample_context(**overrides: str) -> FeatureSampleContext:
    values = {
        "dataset": "tiny-dataset",
        "revision": "dataset-rev-1",
        "split": "validation",
        "preprocessing_version": "prep-v1",
        "tokenizer": "tiny-tokenizer",
        "tokenizer_revision": "tokenizer-rev-1",
    }
    values.update(overrides)
    return FeatureSampleContext(
        values["dataset"],
        values["revision"],
        values["split"],
        ("sample-1",),
        values["preprocessing_version"],
        values["tokenizer"],
        values["tokenizer_revision"],
    )


def _feature(
    name: str = "weight_mean",
    *,
    sample_context: FeatureSampleContext | None = None,
) -> FeatureRecord:
    return FeatureRecord(
        _component(),
        name,
        FeatureKind.SCALAR,
        1.5,
        "float64",
        "weight_statistics",
        "1",
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
        sample_context,
    )


def _partition(
    record: ExperimentRecord,
    *features: FeatureRecord,
    model_revision: str | None = None,
    input_revision: str | None = None,
    digest: str = "a" * 64,
) -> FeaturePartition:
    if not features:
        features = (_feature(),)
    key = FeaturePartitionKey(
        model_revision or record.model.revision,
        input_revision or record.dataset.manifest_id,
        _component(),
        "weight_statistics",
        "1",
    )
    return FeaturePartition(key, tuple(features), digest)


def _join(record: ExperimentRecord | None = None) -> ExperimentFeatureJoin:
    resolved = record or _record()
    return ExperimentFeatureJoin(resolved, (_partition(resolved),))


def test_build_preserves_context_metrics_and_stable_retry_identity() -> None:
    first_record = _record(attempt_id="attempt-1")
    retry_record = _record(attempt_id="attempt-2")
    first = build_mutation_example(_join(first_record))
    retry = build_mutation_example(_join(retry_record))

    assert first is not None
    assert retry is not None
    assert first.example_id == retry.example_id
    assert first.experiment_id == first_record.experiment_id
    assert first.model == first_record.model
    assert first.dataset == first_record.dataset
    assert first.components == first_record.components
    assert first.mutation == first_record.mutation
    assert first.baseline_metrics == first_record.baseline_metrics
    assert first.post_metrics == first_record.post_metrics
    assert first.delta_metrics == first_record.delta_metrics
    assert first.pre_mutation_features == (_feature(),)


def test_post_mutation_or_wrong_input_revision_cannot_enter_inputs() -> None:
    record = _record()
    with pytest.raises(DatasetBuildError, match="pre-mutation revision"):
        build_mutation_example(
            ExperimentFeatureJoin(
                record,
                (_partition(record, model_revision="mutated-revision"),),
            )
        )

    with pytest.raises(DatasetBuildError, match="input revision"):
        build_mutation_example(
            ExperimentFeatureJoin(
                record,
                (_partition(record, input_revision="wrong-manifest"),),
            )
        )


def test_dataset_revision_can_be_selected_as_explicit_feature_input_identity() -> None:
    record = _record()
    policy = MutationExampleBuildPolicy(
        feature_input_revision_source=FeatureInputRevisionSource.DATASET_REVISION
    )
    example = build_mutation_example(
        ExperimentFeatureJoin(
            record,
            (_partition(record, input_revision=record.dataset.revision),),
        ),
        policy,
    )
    assert example is not None


def test_sample_context_must_match_experiment_dataset_and_tokenizer() -> None:
    record = _record()
    valid = _feature(sample_context=_sample_context())
    assert build_mutation_example(
        ExperimentFeatureJoin(record, (_partition(record, valid),))
    ) is not None

    bad_tokenizer = _feature(
        sample_context=_sample_context(tokenizer_revision="post-tokenizer")
    )
    with pytest.raises(DatasetBuildError, match="dataset/tokenizer"):
        build_mutation_example(
            ExperimentFeatureJoin(record, (_partition(record, bad_tokenizer),))
        )


def test_duplicate_feature_identity_across_partitions_fails_closed() -> None:
    record = _record()
    first = _partition(record, digest="a" * 64)
    second = _partition(record, digest="b" * 64)
    with pytest.raises(DatasetBuildError, match="duplicated across partitions"):
        build_mutation_example(ExperimentFeatureJoin(record, (first, second)))


def test_default_outcome_policy_includes_rejected_and_excludes_failed() -> None:
    rejected = _record(outcome=ExperimentOutcomeKind.REJECTED)
    failed = _record(outcome=ExperimentOutcomeKind.FAILED)

    rejected_example = build_mutation_example(_join(rejected))
    failed_example = build_mutation_example(_join(failed))
    assert rejected_example is not None
    assert rejected_example.outcome.kind is ExperimentOutcomeKind.REJECTED
    assert failed_example is None

    report = build_mutation_examples((_join(rejected), _join(failed)))
    assert len(report.examples) == 1
    assert report.exclusions[0].reason is DatasetBuildExclusionReason.OUTCOME_POLICY


def test_failed_outcomes_can_be_included_only_by_explicit_policy() -> None:
    failed = _record(outcome=ExperimentOutcomeKind.FAILED)
    policy = MutationExampleBuildPolicy(
        allowed_outcomes=(ExperimentOutcomeKind.FAILED,),
    )
    example = build_mutation_example(_join(failed), policy)
    assert example is not None
    assert example.outcome.kind is ExperimentOutcomeKind.FAILED


def test_missing_delta_targets_are_explicitly_excluded_or_preserved() -> None:
    missing = (
        MetricObservation("a", MetricState.ABSENT, reason="not collected"),
        MetricObservation("b", MetricState.FAILED, reason="evaluation failed"),
        MetricObservation("c", MetricState.SKIPPED, reason="tier stopped"),
    )
    record = _record(delta_metrics=missing)

    assert build_mutation_example(_join(record)) is None
    default_report = build_mutation_examples((_join(record),))
    assert default_report.exclusions[0].reason is DatasetBuildExclusionReason.NO_MEASURED_DELTA

    preserve = MutationExampleBuildPolicy(
        delta_target_policy=DeltaTargetPolicy.PRESERVE_MISSING
    )
    example = build_mutation_example(_join(record), preserve)
    assert example is not None
    assert example.delta_metrics == missing
    assert tuple(item.state for item in example.delta_metrics) == (
        MetricState.ABSENT,
        MetricState.FAILED,
        MetricState.SKIPPED,
    )


def test_exact_retry_is_emitted_once_and_reported_as_duplicate() -> None:
    first = _record(attempt_id="attempt-1")
    retry = _record(attempt_id="attempt-2")
    report = build_mutation_examples((_join(first), _join(retry)))

    assert report.processed_count == 2
    assert len(report.examples) == 1
    assert len(report.exclusions) == 1
    assert report.exclusions[0].reason is DatasetBuildExclusionReason.DUPLICATE_RETRY
    assert report.to_record()["example_count"] == 1


def test_conflicting_retry_with_same_stable_identity_fails_closed() -> None:
    first = _record(attempt_id="attempt-1")
    changed = _record(
        attempt_id="attempt-2",
        delta_metrics=(MetricObservation("loss_delta", MetricState.MEASURED, 0.5),),
    )
    with pytest.raises(DatasetBuildError, match="conflicting supervised content"):
        build_mutation_examples((_join(first), _join(changed)))


def test_feature_snapshot_digest_changes_example_identity() -> None:
    record = _record()
    first = build_mutation_example(
        ExperimentFeatureJoin(record, (_partition(record, digest="a" * 64),))
    )
    second = build_mutation_example(
        ExperimentFeatureJoin(record, (_partition(record, digest="b" * 64),))
    )
    assert first is not None
    assert second is not None
    assert first.example_id != second.example_id


def test_join_and_policy_validation_fail_early() -> None:
    with pytest.raises(DatasetBuildError, match="pre-mutation feature partitions"):
        ExperimentFeatureJoin(_record(), ())
    with pytest.raises(DatasetBuildError, match="at least one outcome"):
        MutationExampleBuildPolicy(allowed_outcomes=())
    with pytest.raises(DatasetBuildError, match="unique"):
        MutationExampleBuildPolicy(
            allowed_outcomes=(
                ExperimentOutcomeKind.SUCCEEDED,
                ExperimentOutcomeKind.SUCCEEDED,
            )
        )


def test_attempt_id_is_not_supervised_content() -> None:
    record = _record()
    retried = replace(record, attempt_id="attempt-retry")
    first = build_mutation_example(_join(record))
    second = build_mutation_example(_join(retried))
    assert first is not None and second is not None
    assert first.to_record() == second.to_record()
