"""Tests for complete-model and architecture-family held-out dataset splits."""

from __future__ import annotations

from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.datasets.heldout_splits import (
    HeldOutAssignmentStrategy,
    HeldOutSplitConfig,
    HeldOutSplitError,
    HeldOutSplitMode,
    create_heldout_split,
    heldout_key,
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
    revision: str,
    family: str,
) -> MutationExampleRecord:
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta()
    plan = MutationPlan(request, (component,), (), delta)
    mutation = MutationRunRecord(
        plan,
        MutationProvenance(revision, "tool-rev"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return MutationExampleRecord(
        example_id,
        f"experiment-{example_id}",
        ModelTarget(model_identifier, revision, family, "safetensors"),
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
        (),
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool-rev", "config", "evaluator", 1, 1),
        SeedContext(1, 2, 3),
    )


def _partitions(manifest) -> dict[str, SplitPartition]:
    return {
        example_id: group.partition
        for group in manifest.groups
        for example_id in group.example_ids
    }


def test_model_revision_aliases_are_one_indivisible_model_group() -> None:
    examples = (
        _example("a-main", "model-A", "main", "llama"),
        _example("a-sha", "model-A", "abc123", "llama"),
        _example("b", "model-B", "v1", "qwen"),
        _example("c", "model-C", "v1", "mistral"),
    )
    manifest = create_heldout_split(
        examples,
        HeldOutSplitConfig(HeldOutSplitMode.MODEL, seed=7),
    )

    partitions = _partitions(manifest)
    assert partitions["a-main"] is partitions["a-sha"]
    group = next(item for item in manifest.groups if item.key == "model-A")
    assert group.example_ids == ("a-main", "a-sha")
    assert group.model_revisions == ("abc123", "main")
    assert sum(manifest.group_counts.values()) == 3


def test_explicit_model_holdout_supports_train_abc_test_unseen_d() -> None:
    examples = tuple(
        _example(letter, f"model-{letter}", "v1", family)
        for letter, family in (
            ("A", "llama"),
            ("B", "qwen"),
            ("C", "mistral"),
            ("D", "gemma"),
        )
    )
    manifest = create_heldout_split(
        examples,
        HeldOutSplitConfig(
            HeldOutSplitMode.MODEL,
            test_holdouts=("model-D",),
        ),
    )

    assert manifest.strategy is HeldOutAssignmentStrategy.EXPLICIT
    assert manifest.partition_for("D") is SplitPartition.TEST
    assert all(
        manifest.partition_for(value) is SplitPartition.TRAIN
        for value in ("A", "B", "C")
    )
    assert manifest.example_counts[SplitPartition.TEST] == 1
    assert manifest.example_counts[SplitPartition.TRAIN] == 3


def test_explicit_family_holdout_keeps_complete_unseen_family_out_of_train() -> None:
    examples = (
        _example("llama-a", "llama-A", "main", "llama"),
        _example("llama-b", "llama-B", "v1", "llama"),
        _example("qwen", "qwen-A", "v1", "qwen"),
        _example("mistral", "mistral-A", "v1", "mistral"),
        _example("gemma-a", "gemma-A", "main", "gemma"),
        _example("gemma-b", "gemma-B", "v2", "gemma"),
    )
    manifest = create_heldout_split(
        examples,
        HeldOutSplitConfig(
            HeldOutSplitMode.ARCHITECTURE_FAMILY,
            test_holdouts=("gemma",),
        ),
    )

    assert manifest.partition_for("gemma-a") is SplitPartition.TEST
    assert manifest.partition_for("gemma-b") is SplitPartition.TEST
    assert all(
        manifest.partition_for(example_id) is SplitPartition.TRAIN
        for example_id in ("llama-a", "llama-b", "qwen", "mistral")
    )
    gemma = next(group for group in manifest.groups if group.key == "gemma")
    assert gemma.model_identifiers == ("gemma-A", "gemma-B")


def test_explicit_validation_and_test_holdouts_are_disjoint_and_recorded() -> None:
    examples = (
        _example("a", "A", "v1", "llama"),
        _example("b", "B", "v1", "qwen"),
        _example("c", "C", "v1", "mistral"),
    )
    manifest = create_heldout_split(
        examples,
        HeldOutSplitConfig(
            HeldOutSplitMode.MODEL,
            validation_holdouts=("B",),
            test_holdouts=("C",),
        ),
    )

    assert manifest.partition_for("a") is SplitPartition.TRAIN
    assert manifest.partition_for("b") is SplitPartition.VALIDATION
    assert manifest.partition_for("c") is SplitPartition.TEST
    assert manifest.to_record()["validation_holdouts"] == ["B"]
    assert manifest.to_record()["test_holdouts"] == ["C"]


def test_seeded_family_split_requires_enough_distinct_families_with_guidance() -> None:
    examples = (
        _example("a", "A", "v1", "llama"),
        _example("b", "B", "v1", "qwen"),
    )
    try:
        create_heldout_split(
            examples,
            HeldOutSplitConfig(HeldOutSplitMode.ARCHITECTURE_FAMILY, seed=1),
        )
    except HeldOutSplitError as error:
        message = str(error)
        assert "at least 3 distinct architecture families" in message
        assert "use explicit holdouts" in message
    else:
        raise AssertionError("insufficient family counts should fail with guidance")


def test_seeded_model_split_requires_three_complete_models() -> None:
    examples = (
        _example("a", "A", "v1", "llama"),
        _example("b", "B", "v1", "llama"),
    )
    try:
        create_heldout_split(examples, HeldOutSplitConfig(HeldOutSplitMode.MODEL, seed=1))
    except HeldOutSplitError as error:
        assert "at least 3 distinct models" in str(error)
    else:
        raise AssertionError("seeded train/validation/test needs three model groups")


def test_seeded_split_is_deterministic_across_input_order() -> None:
    examples = tuple(
        _example(f"e{index}", f"model-{index}", "v1", family)
        for index, family in enumerate(("llama", "qwen", "mistral", "gemma"))
    )
    config = HeldOutSplitConfig(
        HeldOutSplitMode.MODEL,
        seed=123,
        ratios=SplitRatios(0.5, 0.25, 0.25),
    )
    first = create_heldout_split(examples, config)
    second = create_heldout_split(tuple(reversed(examples)), config)
    assert first.to_record() == second.to_record()
    assert first.strategy is HeldOutAssignmentStrategy.SEEDED_RATIOS


def test_family_keys_consume_only_supported_canonical_family_contract() -> None:
    supported = _example("q", "Q", "v1", "qwen")
    assert heldout_key(supported, HeldOutSplitMode.ARCHITECTURE_FAMILY) == "qwen"

    unsupported = _example("x", "X", "v1", "invented-family")
    try:
        create_heldout_split(
            (unsupported,),
            HeldOutSplitConfig(
                HeldOutSplitMode.ARCHITECTURE_FAMILY,
                test_holdouts=("invented-family",),
            ),
        )
    except HeldOutSplitError as error:
        assert "canonical #11 family" in str(error)
    else:
        raise AssertionError("unknown architecture family should fail closed")


def test_conflicting_family_for_same_model_identifier_fails_closed() -> None:
    examples = (
        _example("a", "same-model", "v1", "llama"),
        _example("b", "same-model", "v2", "qwen"),
    )
    try:
        create_heldout_split(
            examples,
            HeldOutSplitConfig(
                HeldOutSplitMode.MODEL,
                test_holdouts=("same-model",),
            ),
        )
    except HeldOutSplitError as error:
        assert "conflicting canonical families" in str(error)
    else:
        raise AssertionError("one model identifier cannot claim multiple families")


def test_explicit_holdout_must_exist_and_leave_training_group() -> None:
    examples = (
        _example("a", "A", "v1", "llama"),
        _example("b", "B", "v1", "qwen"),
    )
    try:
        create_heldout_split(
            examples,
            HeldOutSplitConfig(HeldOutSplitMode.MODEL, test_holdouts=("missing",)),
        )
    except HeldOutSplitError as error:
        assert "absent" in str(error)
        assert "available keys" in str(error)
    else:
        raise AssertionError("missing explicit holdout should fail")

    try:
        create_heldout_split(
            examples,
            HeldOutSplitConfig(HeldOutSplitMode.MODEL, test_holdouts=("A", "B")),
        )
    except HeldOutSplitError as error:
        assert "leaves no training group" in str(error)
    else:
        raise AssertionError("explicit holdout cannot consume all training groups")


def test_holdout_config_rejects_overlap_noncanonical_values_and_bad_seed() -> None:
    for factory in (
        lambda: HeldOutSplitConfig(
            HeldOutSplitMode.MODEL,
            validation_holdouts=("A",),
            test_holdouts=("A",),
        ),
        lambda: HeldOutSplitConfig(
            HeldOutSplitMode.MODEL,
            test_holdouts=("B", "A"),
        ),
        lambda: HeldOutSplitConfig(HeldOutSplitMode.MODEL, seed=-1),
    ):
        try:
            factory()
        except HeldOutSplitError:
            pass
        else:
            raise AssertionError("invalid held-out split config should fail")


def test_duplicate_dataset_is_rejected_before_holdout_assignment() -> None:
    duplicate = _example("duplicate", "A", "v1", "llama")
    try:
        create_heldout_split(
            (duplicate, duplicate),
            HeldOutSplitConfig(HeldOutSplitMode.MODEL, test_holdouts=("A",)),
        )
    except HeldOutSplitError as error:
        assert "duplicate_example_id" in str(error)
    else:
        raise AssertionError("duplicate dataset should fail validation before splitting")
