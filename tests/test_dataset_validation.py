"""Tests for machine-readable mutation dataset validation."""

from __future__ import annotations

from copy import deepcopy

from modelsurgeon.datasets import (
    DatasetValidationConfig,
    DatasetValidationRule,
    ExperimentFeatureJoin,
    build_mutation_example,
    validate_mutation_dataset,
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
        MutationProvenance("model-rev-1", "tool-rev"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )


def _record() -> ExperimentRecord:
    mutation = _mutation()
    return ExperimentRecord(
        "run-1",
        "experiment-1",
        "attempt-1",
        ModelTarget("tiny/model", "model-rev-1", "llama", "safetensors", 128),
        DatasetTarget(
            "tiny-dataset",
            "dataset-rev-1",
            "validation",
            "manifest-1",
            "tiny-tokenizer",
            "tokenizer-rev-1",
        ),
        mutation.plan.affected_components,
        mutation,
        (MetricObservation("loss", MetricState.MEASURED, 1.0, "nats"),),
        (MetricObservation("loss", MetricState.MEASURED, 1.25, "nats"),),
        (MetricObservation("loss_delta", MetricState.MEASURED, 0.25, "nats"),),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool-rev", "config-digest", "eval-v1", 1, 1),
        SeedContext(1, 2, 3),
    )


def _feature() -> FeatureRecord:
    return FeatureRecord(
        _component(),
        "weight_mean",
        FeatureKind.SCALAR,
        1.5,
        "float64",
        "weight_statistics",
        "1",
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
        FeatureSampleContext(
            "tiny-dataset",
            "dataset-rev-1",
            "validation",
            ("sample-1",),
            "prep-v1",
            "tiny-tokenizer",
            "tokenizer-rev-1",
        ),
    )


def _example():
    record = _record()
    partition = FeaturePartition(
        FeaturePartitionKey(
            record.model.revision,
            record.dataset.manifest_id,
            _component(),
            "weight_statistics",
            "1",
        ),
        (_feature(),),
        "a" * 64,
    )
    example = build_mutation_example(ExperimentFeatureJoin(record, (partition,)))
    assert example is not None
    return example


def _raw() -> dict[str, object]:
    return deepcopy(_example().to_record())


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def test_valid_typed_and_serialized_examples_pass() -> None:
    example = _example()
    typed = validate_mutation_dataset((example,))
    raw = validate_mutation_dataset((example.to_record(),))

    assert typed.valid
    assert raw.valid
    assert typed.record_count == 1
    assert typed.issues == ()
    assert typed.to_record()["valid"] is True


def test_schema_rule_is_machine_readable() -> None:
    raw = _raw()
    raw.pop("outcome")
    report = validate_mutation_dataset((raw,))

    assert not report.valid
    issue = next(item for item in report.issues if item.rule is DatasetValidationRule.SCHEMA)
    assert issue.path == "$"
    assert issue.code == "root_fields"
    assert issue.example_id == _example().example_id


def test_finite_range_rule_catches_nonfinite_feature_and_invalid_seed() -> None:
    raw = _raw()
    feature = _dict(_list(raw["pre_mutation_features"])[0])
    feature["value"] = float("nan")
    seeds = _dict(raw["seeds"])
    seeds["mutation_seed"] = -1

    report = validate_mutation_dataset((raw,))
    finite = [item for item in report.issues if item.rule is DatasetValidationRule.FINITE_RANGE]
    assert {item.code for item in finite} >= {
        "finite_feature_value",
        "unsigned_64_bit_integer",
    }


def test_component_reference_rule_catches_invalid_and_mismatched_components() -> None:
    raw = _raw()
    raw["components"] = ["not a component id"]
    report = validate_mutation_dataset((raw,))

    component_issues = [
        item for item in report.issues if item.rule is DatasetValidationRule.COMPONENT_REFERENCE
    ]
    assert component_issues
    assert any(item.code == "invalid_component_id" for item in component_issues)

    mismatched = _raw()
    mismatched["components"] = ["model.layers.1.mlp.up_proj"]
    mismatch_report = validate_mutation_dataset((mismatched,))
    assert any(
        item.code == "affected_component_mismatch"
        for item in mismatch_report.issues
        if item.rule is DatasetValidationRule.COMPONENT_REFERENCE
    )


def test_revision_provenance_rule_checks_mutation_tool_and_feature_context() -> None:
    raw = _raw()
    mutation = _dict(raw["mutation"])
    provenance = _dict(mutation["provenance"])
    provenance["input_revision"] = "post-mutation-revision"
    provenance["tool_revision"] = "other-tool"
    feature = _dict(_list(raw["pre_mutation_features"])[0])
    sample = _dict(feature["sample_context"])
    sample["tokenizer_revision"] = "other-tokenizer"

    report = validate_mutation_dataset((raw,))
    provenance_issues = [
        item
        for item in report.issues
        if item.rule is DatasetValidationRule.REVISION_PROVENANCE
    ]
    assert {item.code for item in provenance_issues} >= {
        "model_revision_mismatch",
        "tool_revision_mismatch",
        "sample_context_mismatch",
    }


def test_target_calculation_rule_checks_post_minus_baseline_and_units() -> None:
    raw = _raw()
    delta = _dict(_list(raw["delta_metrics"])[0])
    delta["value"] = 0.5
    delta["unit"] = "bits"

    report = validate_mutation_dataset((raw,))
    target_issues = [
        item for item in report.issues if item.rule is DatasetValidationRule.TARGET_CALCULATION
    ]
    assert {item.code for item in target_issues} >= {
        "incorrect_delta",
        "delta_unit_mismatch",
    }


def test_target_tolerance_is_explicit_and_configurable() -> None:
    raw = _raw()
    delta = _dict(_list(raw["delta_metrics"])[0])
    delta["value"] = 0.25001

    strict = validate_mutation_dataset((raw,))
    relaxed = validate_mutation_dataset(
        (raw,),
        DatasetValidationConfig(target_absolute_tolerance=1e-3),
    )
    assert any(item.code == "incorrect_delta" for item in strict.issues)
    assert not any(item.code == "incorrect_delta" for item in relaxed.issues)


def test_duplicate_example_ids_are_reported_with_first_record_index() -> None:
    example = _example()
    report = validate_mutation_dataset((example, example))

    duplicate = next(
        item for item in report.issues if item.rule is DatasetValidationRule.DUPLICATE_ID
    )
    assert duplicate.record_index == 1
    assert duplicate.code == "duplicate_example_id"
    assert "record index 0" in duplicate.detail


def test_corrupt_fixture_exercises_every_declared_rule() -> None:
    schema = _raw()
    schema["schema_version"] = 99

    finite = _raw()
    _dict(finite["model"])["parameter_count"] = 0

    component = _raw()
    component["components"] = ["bad component"]

    revision = _raw()
    _dict(_dict(revision["mutation"])["provenance"])["input_revision"] = "wrong"

    target = _raw()
    _dict(_list(target["delta_metrics"])[0])["value"] = -999.0

    duplicate = _raw()
    report = validate_mutation_dataset(
        (schema, finite, component, revision, target, duplicate, duplicate)
    )
    rules = {item.rule for item in report.issues}
    assert rules == set(DatasetValidationRule)


def test_validator_collects_multiple_issues_instead_of_failing_fast() -> None:
    raw = _raw()
    raw["example_id"] = ""
    raw["components"] = []
    _dict(raw["seeds"])["data_seed"] = 1 << 64
    _dict(_list(raw["post_metrics"])[0])["value"] = float("inf")

    report = validate_mutation_dataset((raw,))
    assert not report.valid
    assert len(report.issues) >= 4
    assert tuple(item.to_record() for item in report.issues)
