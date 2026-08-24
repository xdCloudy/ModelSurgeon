"""Tests for deterministic leakage auditing across supported split strategies."""

from __future__ import annotations

from modelsurgeon.datasets.grouped_splits import (
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitGroup,
    SplitPartition,
    SplitRatios,
)
from modelsurgeon.datasets.heldout_splits import (
    HeldOutAssignmentStrategy,
    HeldOutGroup,
    HeldOutSplitManifest,
    HeldOutSplitMode,
)
from modelsurgeon.datasets.leakage import (
    DatasetLeakageError,
    LeakageAuditConfig,
    LeakageKind,
    ModelAncestry,
    audit_dataset_leakage,
)
from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    HardwareInventory,
    MemoryInventory,
    ModelTarget,
    MutationExampleRecord,
    SeedContext,
    SoftwareInventory,
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


def _hardware() -> HardwareInventory:
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "cpu", 4),
        MemoryInventory(1024, 512),
        DiskInventory("/tmp", 4096, 2048),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _example(
    example_id: str,
    model_identifier: str,
    component_path: str,
    *,
    revision: str = "v1",
    numeric_parameter: int | None = None,
    target_feature: bool = False,
) -> MutationExampleRecord:
    component = ComponentId.parse(component_path)
    parameters = () if numeric_parameter is None else (("strength", numeric_parameter),)
    request = MutationRequest(MutationKind.MASK, (component,), parameters)
    delta = MutationDelta()
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(revision, "tool-rev"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    features: tuple[FeatureRecord, ...] = ()
    if target_feature:
        features = (
            FeatureRecord(
                component,
                "target_proxy",
                FeatureKind.SCALAR,
                1.0,
                "float32",
                "fixture",
                "1",
                PrecisionProvenance(
                    PrecisionSource.HIGH_PRECISION,
                    "float32",
                    "float32",
                ),
                metadata=(("target_derived", True),),
            ),
        )
    return MutationExampleRecord(
        example_id,
        f"experiment-{example_id}",
        ModelTarget(model_identifier, revision, "llama", "safetensors"),
        DatasetTarget(
            "tiny-data",
            "data-rev",
            "validation",
            "manifest-1",
            "tokenizer",
            "tok-rev",
        ),
        (component,),
        mutation,
        features,
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool-rev", "config", "evaluator", 1, 1),
        SeedContext(1, 2, 3),
    )


def _grouped_manifest(
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    test_ids: tuple[str, ...] = (),
) -> GroupedSplitManifest:
    groups = []
    for group_id, partition, example_ids in (
        ("group-1", SplitPartition.TRAIN, train_ids),
        ("group-2", SplitPartition.VALIDATION, validation_ids),
        ("group-3", SplitPartition.TEST, test_ids),
    ):
        if example_ids:
            groups.append(
                SplitGroup(
                    group_id,
                    partition,
                    (f"manual:{group_id}",),
                    tuple(sorted(example_ids)),
                )
            )
    return GroupedSplitManifest(
        GroupedSplitMode.COMPONENT,
        7,
        SplitRatios(),
        tuple(groups),
    )


def _heldout_manifest(
    examples: tuple[MutationExampleRecord, ...],
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
) -> HeldOutSplitManifest:
    by_id = {example.example_id: example for example in examples}

    def group(key: str, partition: SplitPartition, ids: tuple[str, ...]) -> HeldOutGroup:
        selected = tuple(by_id[example_id] for example_id in ids)
        return HeldOutGroup(
            key,
            partition,
            tuple(sorted({item.model.identifier for item in selected})),
            tuple(sorted({item.model.revision for item in selected})),
            tuple(sorted(ids)),
        )

    groups = (
        group("heldout-a", SplitPartition.TRAIN, train_ids),
        group("heldout-b", SplitPartition.TEST, test_ids),
    )
    return HeldOutSplitManifest(
        HeldOutSplitMode.MODEL,
        HeldOutAssignmentStrategy.EXPLICIT,
        0,
        SplitRatios(),
        (),
        ("heldout-b",),
        groups,
    )


def _kinds(report) -> set[LeakageKind]:
    return {finding.kind for finding in report.findings}


def test_clean_grouped_and_heldout_manifests_pass_and_gate_training() -> None:
    examples = (
        _example("a", "model-A", "model.layers.0.mlp.up_proj"),
        _example("b", "model-B", "model.layers.1.mlp.up_proj"),
    )
    grouped = _grouped_manifest(("a",), ("b",))
    grouped_report = audit_dataset_leakage(examples, grouped)
    assert grouped_report.clean
    grouped_report.require_clean()

    heldout = _heldout_manifest(examples, ("a",), ("b",))
    heldout_report = audit_dataset_leakage(examples, heldout)
    assert heldout_report.clean
    assert heldout_report.manifest_kind == "heldout:model"
    heldout_report.require_clean()


def test_exact_candidate_and_shared_component_leakage_are_detected() -> None:
    examples = (
        _example("a", "same-model", "model.layers.0.mlp.up_proj"),
        _example("b", "same-model", "model.layers.0.mlp.up_proj"),
    )
    report = audit_dataset_leakage(
        examples,
        _grouped_manifest(("a",), ("b",)),
    )

    assert LeakageKind.EXACT_CANDIDATE in _kinds(report)
    assert LeakageKind.SHARED_COMPONENT in _kinds(report)
    assert not report.clean
    try:
        report.require_clean()
    except DatasetLeakageError as error:
        assert "dataset leakage audit failed" in str(error)
    else:
        raise AssertionError("training gate must reject a leaking dataset")


def test_near_duplicate_candidate_ignores_numeric_parameter_value() -> None:
    examples = (
        _example(
            "a",
            "same-model",
            "model.layers.0.mlp.up_proj",
            numeric_parameter=1,
        ),
        _example(
            "b",
            "same-model",
            "model.layers.0.mlp.up_proj",
            numeric_parameter=2,
        ),
    )
    report = audit_dataset_leakage(
        examples,
        _grouped_manifest(("a",), ("b",)),
    )

    assert LeakageKind.NEAR_DUPLICATE_CANDIDATE in _kinds(report)
    assert LeakageKind.EXACT_CANDIDATE not in _kinds(report)


def test_same_candidate_shape_on_different_models_is_not_duplicate_leakage() -> None:
    examples = (
        _example(
            "a",
            "model-A",
            "model.layers.0.mlp.up_proj",
            numeric_parameter=1,
        ),
        _example(
            "b",
            "model-B",
            "model.layers.0.mlp.up_proj",
            numeric_parameter=2,
        ),
    )
    report = audit_dataset_leakage(
        examples,
        _grouped_manifest(("a",), ("b",)),
    )

    assert LeakageKind.EXACT_CANDIDATE not in _kinds(report)
    assert LeakageKind.NEAR_DUPLICATE_CANDIDATE not in _kinds(report)
    assert LeakageKind.SHARED_COMPONENT not in _kinds(report)


def test_explicit_model_ancestry_detects_parent_child_and_sibling_leakage() -> None:
    examples = (
        _example("base", "base-model", "model.layers.0.mlp.up_proj"),
        _example("child", "child-model", "model.layers.0.mlp.up_proj"),
        _example("sibling", "sibling-model", "model.layers.1.mlp.up_proj"),
    )
    manifest = _grouped_manifest(("base",), ("child", "sibling"))
    config = LeakageAuditConfig(
        model_ancestry=(
            ModelAncestry("child-model", ("base-model",)),
            ModelAncestry("sibling-model", ("base-model",)),
        )
    )
    report = audit_dataset_leakage(examples, manifest, config)

    ancestry = [
        finding for finding in report.findings if finding.kind is LeakageKind.MODEL_ANCESTRY
    ]
    assert ancestry
    assert any("base-model|child-model" in finding.key for finding in ancestry)
    assert any("base-model|sibling-model" in finding.key for finding in ancestry)


def test_target_derived_feature_is_rejected_without_cross_partition_overlap() -> None:
    examples = (
        _example(
            "a",
            "model-A",
            "model.layers.0.mlp.up_proj",
            target_feature=True,
        ),
        _example("b", "model-B", "model.layers.1.mlp.up_proj"),
    )
    report = audit_dataset_leakage(
        examples,
        _grouped_manifest(("a",), ("b",)),
    )

    findings = [
        finding
        for finding in report.findings
        if finding.kind is LeakageKind.TARGET_DERIVED_FEATURE
    ]
    assert len(findings) == 1
    assert findings[0].example_ids == ("a",)
    assert findings[0].partitions == (SplitPartition.TRAIN,)


def test_target_feature_name_and_extractor_denylists_are_supported() -> None:
    example = _example("a", "model-A", "model.layers.0.mlp.up_proj")
    component = example.components[0]
    precision = PrecisionProvenance(
        PrecisionSource.HIGH_PRECISION,
        "float32",
        "float32",
    )
    named = FeatureRecord(
        component,
        "known_target",
        FeatureKind.SCALAR,
        1.0,
        "float32",
        "safe-extractor",
        "1",
        precision,
    )
    extracted = FeatureRecord(
        component,
        "other_feature",
        FeatureKind.SCALAR,
        2.0,
        "float32",
        "target-extractor",
        "1",
        precision,
    )
    example = MutationExampleRecord(
        example.example_id,
        example.experiment_id,
        example.model,
        example.dataset,
        example.components,
        example.mutation,
        tuple(sorted((named, extracted), key=lambda item: item.name)),
        example.baseline_metrics,
        example.post_metrics,
        example.delta_metrics,
        example.outcome,
        example.hardware,
        example.versions,
        example.seeds,
    )
    config = LeakageAuditConfig(
        target_feature_names=("known_target",),
        target_feature_extractors=("target-extractor",),
    )
    report = audit_dataset_leakage(
        (example,),
        _grouped_manifest(("a",), ()),
        config,
    )

    target_findings = [
        finding
        for finding in report.findings
        if finding.kind is LeakageKind.TARGET_DERIVED_FEATURE
    ]
    assert len(target_findings) == 2


def test_manifest_coverage_mismatch_is_machine_readable() -> None:
    examples = (
        _example("a", "model-A", "model.layers.0.mlp.up_proj"),
        _example("b", "model-B", "model.layers.1.mlp.up_proj"),
    )
    manifest = _grouped_manifest(("a", "ghost"), ())
    report = audit_dataset_leakage(examples, manifest)

    coverage = [
        finding for finding in report.findings if finding.kind is LeakageKind.MANIFEST_COVERAGE
    ]
    assert len(coverage) == 2
    assert {finding.key for finding in coverage} == {
        "dataset_examples_missing_from_manifest",
        "manifest_examples_missing_from_dataset",
    }


def test_ancestry_cycles_and_noncanonical_config_fail_closed() -> None:
    cyclic = LeakageAuditConfig(
        model_ancestry=(
            ModelAncestry("A", ("B",)),
            ModelAncestry("B", ("A",)),
        )
    )
    examples = (
        _example("a", "A", "model.layers.0.mlp.up_proj"),
        _example("b", "B", "model.layers.1.mlp.up_proj"),
    )
    try:
        audit_dataset_leakage(examples, _grouped_manifest(("a",), ("b",)), cyclic)
    except DatasetLeakageError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("cyclic ancestry must fail closed")

    for factory in (
        lambda: LeakageAuditConfig(target_feature_names=("b", "a")),
        lambda: LeakageAuditConfig(
            model_ancestry=(
                ModelAncestry("B", ()),
                ModelAncestry("A", ()),
            )
        ),
    ):
        try:
            factory()
        except DatasetLeakageError:
            pass
        else:
            raise AssertionError("noncanonical leakage config must fail")
