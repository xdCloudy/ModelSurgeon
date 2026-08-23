"""Tests for versioned experiment and mutation-example dataset records."""

from __future__ import annotations

import pytest

from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    ExperimentSchemaError,
    HardwareInventory,
    MemoryInventory,
    MetricObservation,
    MetricState,
    ModelTarget,
    MutationExampleRecord,
    QuantizationControl,
    QuantizationControlKind,
    SeedContext,
    SoftwareInventory,
    StageTiming,
    VersionContext,
)
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
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
    return ModelTarget(
        "tiny/model",
        "model-rev-1",
        "llama",
        "safetensors",
        parameter_count=128,
    )


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


def _versions() -> VersionContext:
    return VersionContext("tool-rev", "config-digest", "1", 1, 1)


def _seeds() -> SeedContext:
    return SeedContext(1, 2, 3)


def _mutation() -> MutationRunRecord:
    component = _component()
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-8, flops=-16, memory_bytes=-32, storage_bytes=0)
    plan = MutationPlan(request, (component,), (), delta)
    return MutationRunRecord(
        plan,
        MutationProvenance("model-rev-1", "tool-rev", "/private/source"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )


def _precision() -> PrecisionProvenance:
    return PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64")


def _feature(name: str = "weight_mean") -> FeatureRecord:
    return FeatureRecord(
        _component(),
        name,
        FeatureKind.SCALAR,
        1.5,
        "float64",
        "weight_statistics",
        "1",
        _precision(),
    )


def _metrics() -> tuple[MetricObservation, ...]:
    return (
        MetricObservation(
            "cosine_similarity",
            MetricState.SKIPPED,
            reason="Tier 2 not configured",
        ),
        MetricObservation("perplexity", MetricState.MEASURED, 2.5, "ratio"),
    )


def test_metric_states_are_explicit_and_invalid_missingness_fails() -> None:
    records = (
        MetricObservation("a", MetricState.ABSENT, reason="not collected"),
        MetricObservation("b", MetricState.SKIPPED, reason="budget stopped"),
        MetricObservation("c", MetricState.FAILED, reason="backend error"),
        MetricObservation("d", MetricState.MEASURED, 1.25, precision=_precision()),
    )
    assert [item.to_record()["state"] for item in records] == [
        "absent",
        "skipped",
        "failed",
        "measured",
    ]
    with pytest.raises(ExperimentSchemaError, match="finite value"):
        MetricObservation("bad", MetricState.MEASURED)
    with pytest.raises(ExperimentSchemaError, match="require a reason"):
        MetricObservation("bad", MetricState.FAILED)
    with pytest.raises(ExperimentSchemaError, match="cannot carry numeric"):
        MetricObservation("bad", MetricState.SKIPPED, 0.0, reason="not run")


def test_matched_quantization_control_is_fully_representable() -> None:
    control = QuantizationControl(
        QuantizationControlKind.MATCHED_REQUANTIZATION,
        True,
        ("Q4_K",),
        (_component(),),
        7,
    )
    assert control.to_record() == {
        "kind": "matched_requantization",
        "complete": True,
        "codecs": ["Q4_K"],
        "affected_components": ["model.layers.0.mlp.up_proj"],
        "seed": 7,
    }
    assert QuantizationControl(QuantizationControlKind.NONE, True).to_record()["kind"] == "none"
    with pytest.raises(ExperimentSchemaError, match="cannot carry codec work"):
        QuantizationControl(QuantizationControlKind.NONE, True, ("Q4_K",))


def test_experiment_record_serializes_all_context_and_redacts_paths() -> None:
    mutation = _mutation()
    record = ExperimentRecord(
        "run-1",
        "experiment-1",
        "attempt-1",
        _model(),
        _dataset(),
        mutation.plan.affected_components,
        mutation,
        _metrics(),
        (MetricObservation("perplexity", MetricState.MEASURED, 2.75, "ratio"),),
        (MetricObservation("delta_perplexity", MetricState.MEASURED, 0.25, "ratio"),),
        ExperimentOutcome(ExperimentOutcomeKind.REJECTED, "quality threshold failed"),
        _hardware(),
        _versions(),
        _seeds(),
        (StageTiming("evaluation", 1.0, 0.5, tokens=32, candidates=1),),
        QuantizationControl(QuantizationControlKind.NONE, True),
    )
    payload = record.to_record()
    assert payload["schema_version"] == 1
    assert payload["components"] == ["model.layers.0.mlp.up_proj"]
    assert payload["mutation"]["provenance"]["input_path"] == "<redacted-local-path>"  # type: ignore[index]
    baseline = payload["baseline_metrics"]
    assert isinstance(baseline, list)
    assert baseline[0]["state"] == "skipped"
    assert payload["hardware"]["cuda"]["available"] is False  # type: ignore[index]


def test_mutation_example_preserves_pre_mutation_feature_precision() -> None:
    mutation = _mutation()
    example = MutationExampleRecord(
        "example-1",
        "experiment-1",
        _model(),
        _dataset(),
        mutation.plan.affected_components,
        mutation,
        (_feature(),),
        _metrics(),
        (MetricObservation("perplexity", MetricState.MEASURED, 2.75),),
        (MetricObservation("delta_perplexity", MetricState.MEASURED, 0.25),),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        _versions(),
        _seeds(),
    )
    payload = example.to_record()
    assert payload["mutation_id"] == mutation.mutation_id
    features = payload["pre_mutation_features"]
    assert isinstance(features, list)
    assert features[0]["precision"]["source"] == "high_precision"


def test_schema_rejects_revision_and_canonical_order_mismatches() -> None:
    mutation = _mutation()
    with pytest.raises(ExperimentSchemaError, match="input revisions"):
        ExperimentRecord(
            "run-1",
            "experiment-1",
            "attempt-1",
            ModelTarget("tiny/model", "wrong-rev", "llama", "safetensors"),
            _dataset(),
            mutation.plan.affected_components,
            mutation,
            (),
            (),
            (),
            ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
            _hardware(),
            _versions(),
            _seeds(),
        )

    with pytest.raises(ExperimentSchemaError, match="canonical names"):
        ExperimentRecord(
            "run-1",
            "experiment-1",
            "attempt-1",
            _model(),
            _dataset(),
            mutation.plan.affected_components,
            mutation,
            (
                MetricObservation("z", MetricState.ABSENT, reason="not collected"),
                MetricObservation("a", MetricState.ABSENT, reason="not collected"),
            ),
            (),
            (),
            ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
            _hardware(),
            _versions(),
            _seeds(),
        )
