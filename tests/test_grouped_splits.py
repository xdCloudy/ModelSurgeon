"""Tests for deterministic component, layer, and mutation-family splits."""

from __future__ import annotations

from modelsurgeon.datasets.grouped_splits import (
    DatasetSplitError,
    GroupedSplitConfig,
    GroupedSplitMode,
    SplitPartition,
    SplitRatios,
    create_grouped_split,
    group_keys,
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
    components: tuple[str, ...],
    *,
    kind: MutationKind = MutationKind.MASK,
    parameters: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> MutationExampleRecord:
    ids = tuple(sorted(ComponentId.parse(value) for value in components))
    request = MutationRequest(kind, ids, parameters)
    delta = MutationDelta()
    plan = MutationPlan(request, ids, (), delta)
    mutation = MutationRunRecord(
        plan,
        MutationProvenance("model-rev", "tool-rev"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return MutationExampleRecord(
        example_id,
        "experiment-1",
        ModelTarget("tiny/model", "model-rev", "llama", "safetensors"),
        DatasetTarget(
            "tiny-data",
            "data-rev",
            "validation",
            "manifest-1",
            "tokenizer",
            "tok-rev",
        ),
        ids,
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


def _partition_map(manifest) -> dict[str, SplitPartition]:
    return {
        example_id: group.partition
        for group in manifest.groups
        for example_id in group.example_ids
    }


def test_component_overlap_forms_transitive_indivisible_group() -> None:
    examples = (
        _example("e1", ("model.layers.0.a", "model.layers.0.b")),
        _example("e2", ("model.layers.0.b", "model.layers.1.c")),
        _example("e3", ("model.layers.1.c", "model.layers.2.d")),
        _example("e4", ("model.layers.9.z",)),
    )
    manifest = create_grouped_split(
        examples,
        GroupedSplitConfig(GroupedSplitMode.COMPONENT, seed=7),
    )

    partitions = _partition_map(manifest)
    assert partitions["e1"] is partitions["e2"] is partitions["e3"]
    connected = next(group for group in manifest.groups if "e1" in group.example_ids)
    assert connected.example_ids == ("e1", "e2", "e3")
    assert connected.keys == (
        "component:model.layers.0.a",
        "component:model.layers.0.b",
        "component:model.layers.1.c",
        "component:model.layers.2.d",
    )
    assert sum(manifest.group_counts.values()) == 2


def test_layer_grouping_keeps_shared_and_transitively_bridged_layers_together() -> None:
    examples = (
        _example("e1", ("model.layers.0.self_attn.q_proj",)),
        _example("e2", ("model.layers.0.mlp.up_proj", "model.layers.1.mlp.down_proj")),
        _example("e3", ("model.layers.1.self_attn.q_proj",)),
        _example("e4", ("model.layers.8.mlp.up_proj",)),
    )
    manifest = create_grouped_split(
        examples,
        GroupedSplitConfig(GroupedSplitMode.LAYER, seed=11),
    )

    partitions = _partition_map(manifest)
    assert partitions["e1"] is partitions["e2"] is partitions["e3"]
    assert sum(manifest.group_counts.values()) == 2
    group = next(item for item in manifest.groups if "e2" in item.example_ids)
    assert group.keys == ("layer:model.layers.0", "layer:model.layers.1")


def test_layer_index_parameter_is_fallback_for_nonstandard_component_path() -> None:
    example = _example(
        "e1",
        ("model.transformer.stage.alpha",),
        parameters=(("layer_index", 4),),
    )
    assert group_keys(example, GroupedSplitMode.LAYER) == ("layer:model.layer.4",)


def test_layer_grouping_without_resolvable_layer_fails_with_guidance() -> None:
    example = _example("e1", ("model.embedding",))
    try:
        create_grouped_split(
            (example,),
            GroupedSplitConfig(GroupedSplitMode.LAYER, seed=1),
        )
    except DatasetSplitError as error:
        assert "no canonical layer identity" in str(error)
    else:
        raise AssertionError("unresolvable layer grouping should fail")


def test_mutation_family_uses_explicit_family_then_scope_then_kind() -> None:
    explicit_a = _example(
        "explicit-a",
        ("model.layers.0.a",),
        parameters=(("mutation_family", "attention-ablation"),),
    )
    explicit_b = _example(
        "explicit-b",
        ("model.layers.1.a",),
        parameters=(("mutation_family", "attention-ablation"),),
    )
    scoped = _example(
        "scoped",
        ("model.layers.2.a",),
        parameters=(("candidate_scope", "mlp_channel"),),
    )
    removed = _example(
        "removed",
        ("model.layers.3.a",),
        kind=MutationKind.REMOVE,
    )
    manifest = create_grouped_split(
        (explicit_a, explicit_b, scoped, removed),
        GroupedSplitConfig(GroupedSplitMode.MUTATION_FAMILY, seed=5),
    )

    group = next(item for item in manifest.groups if "explicit-a" in item.example_ids)
    assert group.example_ids == ("explicit-a", "explicit-b")
    assert group.keys == ("mutation_family:attention-ablation",)
    assert group_keys(scoped, GroupedSplitMode.MUTATION_FAMILY) == (
        "mutation_family:mask:mlp_channel",
    )
    assert group_keys(removed, GroupedSplitMode.MUTATION_FAMILY) == (
        "mutation_family:remove",
    )
    assert sum(manifest.group_counts.values()) == 3


def test_manifest_is_deterministic_across_input_order_and_records_seed_counts() -> None:
    examples = tuple(
        _example(f"e{index}", (f"model.layers.{index}.a",))
        for index in range(6)
    )
    config = GroupedSplitConfig(
        GroupedSplitMode.COMPONENT,
        seed=1234,
        ratios=SplitRatios(0.5, 0.25, 0.25),
    )
    first = create_grouped_split(examples, config)
    reversed_input = create_grouped_split(tuple(reversed(examples)), config)

    assert first.to_record() == reversed_input.to_record()
    record = first.to_record()
    assert record["seed"] == 1234
    assert record["algorithm"] == "connected-groups-greedy-v1"
    assert record["group_counts"] == {
        partition.value: first.group_counts[partition]
        for partition in SplitPartition
    }
    assert sum(first.group_counts.values()) == 6
    assert sum(first.example_counts.values()) == 6
    assert set(first.example_ids(SplitPartition.TRAIN)) <= {item.example_id for item in examples}


def test_seed_changes_which_equal_sized_groups_land_in_partitions() -> None:
    examples = tuple(
        _example(f"e{index}", (f"model.layers.{index}.a",))
        for index in range(9)
    )
    first = create_grouped_split(
        examples,
        GroupedSplitConfig(GroupedSplitMode.COMPONENT, seed=1),
    )
    second = create_grouped_split(
        examples,
        GroupedSplitConfig(GroupedSplitMode.COMPONENT, seed=2),
    )
    assert _partition_map(first) != _partition_map(second)


def test_every_configured_key_maps_to_exactly_one_partition() -> None:
    examples = (
        _example("e1", ("model.layers.0.a", "model.layers.1.a")),
        _example("e2", ("model.layers.1.a",)),
        _example("e3", ("model.layers.2.a",)),
        _example("e4", ("model.layers.3.a",)),
    )
    for mode in (GroupedSplitMode.COMPONENT, GroupedSplitMode.LAYER):
        manifest = create_grouped_split(examples, GroupedSplitConfig(mode, seed=99))
        partitions_by_key: dict[str, set[SplitPartition]] = {}
        partitions = _partition_map(manifest)
        for example in examples:
            for key in group_keys(example, mode):
                partitions_by_key.setdefault(key, set()).add(partitions[example.example_id])
        assert all(len(partitions) == 1 for partitions in partitions_by_key.values())


def test_invalid_or_duplicate_dataset_fails_before_split_assignment() -> None:
    duplicate = _example("duplicate", ("model.layers.0.a",))
    try:
        create_grouped_split(
            (duplicate, duplicate),
            GroupedSplitConfig(GroupedSplitMode.COMPONENT, seed=1),
        )
    except DatasetSplitError as error:
        assert "dataset validation failed" in str(error)
        assert "duplicate_example_id" in str(error)
    else:
        raise AssertionError("duplicate examples should fail dataset validation")


def test_split_config_validation() -> None:
    for ratios in (
        SplitRatios(0.5, 0.25, 0.25),
        SplitRatios(0.7, 0.2, 0.1),
    ):
        assert math_is_one(ratios.train + ratios.validation + ratios.test)

    for args in ((0.8, 0.2, 0.2), (1.0, 0.0, 0.0)):
        try:
            SplitRatios(*args)
        except DatasetSplitError:
            pass
        else:
            raise AssertionError("invalid split ratios should fail")

    try:
        GroupedSplitConfig(GroupedSplitMode.COMPONENT, seed=-1)
    except DatasetSplitError:
        pass
    else:
        raise AssertionError("negative split seeds should fail")


def math_is_one(value: float) -> bool:
    return abs(value - 1.0) <= 1e-12
