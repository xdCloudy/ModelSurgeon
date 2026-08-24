"""Deterministic leakage-safe splits by component, layer, or mutation family."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import MutationExampleRecord
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationPrimitive
from modelsurgeon.datasets.validation import validate_mutation_dataset

GROUPED_SPLIT_VERSION = "1"
GROUPED_SPLIT_ALGORITHM = "connected-groups-greedy-v1"


class DatasetSplitError(ValueError):
    """Raised when validated examples cannot satisfy a configured split contract."""


class SplitPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class GroupedSplitMode(StrEnum):
    COMPONENT = "component"
    LAYER = "layer"
    MUTATION_FAMILY = "mutation_family"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise DatasetSplitError("split ratios must be finite and positive")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise DatasetSplitError("split ratios must sum to one")

    def to_record(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True, slots=True)
class GroupedSplitConfig:
    mode: GroupedSplitMode
    seed: int
    ratios: SplitRatios = SplitRatios()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise DatasetSplitError("split seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class SplitGroup:
    group_id: str
    partition: SplitPartition
    keys: tuple[str, ...]
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.group_id or not self.keys or not self.example_ids:
            raise DatasetSplitError("split groups require identity, keys, and examples")
        if self.keys != tuple(sorted(set(self.keys))):
            raise DatasetSplitError("split group keys must be unique and canonical")
        if self.example_ids != tuple(sorted(set(self.example_ids))):
            raise DatasetSplitError("split group examples must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "partition": self.partition.value,
            "keys": list(self.keys),
            "example_ids": list(self.example_ids),
        }


@dataclass(frozen=True, slots=True)
class GroupedSplitManifest:
    mode: GroupedSplitMode
    seed: int
    ratios: SplitRatios
    groups: tuple[SplitGroup, ...]
    version: str = GROUPED_SPLIT_VERSION
    algorithm: str = GROUPED_SPLIT_ALGORITHM

    def __post_init__(self) -> None:
        if self.version != GROUPED_SPLIT_VERSION or self.algorithm != GROUPED_SPLIT_ALGORITHM:
            raise DatasetSplitError("unsupported grouped split manifest version or algorithm")
        group_ids = tuple(group.group_id for group in self.groups)
        if group_ids != tuple(sorted(set(group_ids))):
            raise DatasetSplitError("split manifest groups must be unique and canonical")
        example_ids = tuple(
            example_id for group in self.groups for example_id in group.example_ids
        )
        if len(example_ids) != len(set(example_ids)):
            raise DatasetSplitError("an example cannot appear in multiple split groups")

    @property
    def group_counts(self) -> dict[SplitPartition, int]:
        return {
            partition: sum(group.partition is partition for group in self.groups)
            for partition in SplitPartition
        }

    @property
    def example_counts(self) -> dict[SplitPartition, int]:
        return {
            partition: sum(
                len(group.example_ids)
                for group in self.groups
                if group.partition is partition
            )
            for partition in SplitPartition
        }

    def partition_for(self, example_id: str) -> SplitPartition:
        for group in self.groups:
            if example_id in group.example_ids:
                return group.partition
        raise DatasetSplitError(f"example {example_id!r} is absent from the split manifest")

    def example_ids(self, partition: SplitPartition) -> tuple[str, ...]:
        return tuple(
            sorted(
                example_id
                for group in self.groups
                if group.partition is partition
                for example_id in group.example_ids
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "mode": self.mode.value,
            "seed": self.seed,
            "ratios": self.ratios.to_record(),
            "group_counts": {
                partition.value: self.group_counts[partition]
                for partition in SplitPartition
            },
            "example_counts": {
                partition.value: self.example_counts[partition]
                for partition in SplitPartition
            },
            "groups": [group.to_record() for group in self.groups],
        }


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _parameters(example: MutationExampleRecord) -> dict[str, MutationPrimitive]:
    return dict(example.mutation.plan.request.parameters)


def _component_keys(example: MutationExampleRecord) -> tuple[str, ...]:
    return tuple(f"component:{component}" for component in example.components)


def _layer_path(component: ComponentId) -> str | None:
    values = tuple(segment.value for segment in component.segments)
    layer_names = frozenset({"layer", "layers", "block", "blocks", "h"})
    for position in range(len(values) - 1):
        if values[position] in layer_names and isinstance(values[position + 1], int):
            return str(ComponentId(component.segments[: position + 2]))
    return None


def _layer_keys(example: MutationExampleRecord) -> tuple[str, ...]:
    paths = {
        path
        for component in example.components
        if (path := _layer_path(component)) is not None
    }
    if not paths:
        layer_index = _parameters(example).get("layer_index")
        if isinstance(layer_index, int) and not isinstance(layer_index, bool) and layer_index >= 0:
            paths.add(f"model.layer.{layer_index}")
    if not paths:
        raise DatasetSplitError(
            f"example {example.example_id} has no canonical layer identity for layer grouping"
        )
    return tuple(f"layer:{path}" for path in sorted(paths))


def _mutation_family_keys(example: MutationExampleRecord) -> tuple[str, ...]:
    request = example.mutation.plan.request
    parameters = _parameters(example)
    explicit = parameters.get("mutation_family")
    if isinstance(explicit, str) and explicit:
        family = explicit
    else:
        scope = parameters.get("candidate_scope")
        family = request.kind.value
        if isinstance(scope, str) and scope:
            family = f"{family}:{scope}"
    return (f"mutation_family:{family}",)


def group_keys(
    example: MutationExampleRecord,
    mode: GroupedSplitMode,
) -> tuple[str, ...]:
    """Return canonical leakage keys contributed by one example."""

    if mode is GroupedSplitMode.COMPONENT:
        return _component_keys(example)
    if mode is GroupedSplitMode.LAYER:
        return _layer_keys(example)
    return _mutation_family_keys(example)


def _connected_groups(
    examples: tuple[MutationExampleRecord, ...],
    mode: GroupedSplitMode,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    union = _UnionFind(len(examples))
    keys_by_example = tuple(group_keys(example, mode) for example in examples)
    owner_by_key: dict[str, int] = {}
    for index, keys in enumerate(keys_by_example):
        for key in keys:
            previous = owner_by_key.setdefault(key, index)
            union.union(index, previous)

    members: dict[int, list[int]] = {}
    for index in range(len(examples)):
        members.setdefault(union.find(index), []).append(index)

    output: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for indexes in members.values():
        keys = tuple(sorted({key for index in indexes for key in keys_by_example[index]}))
        example_ids = tuple(sorted(examples[index].example_id for index in indexes))
        output.append((keys, example_ids))
    return tuple(sorted(output, key=lambda item: (item[0], item[1])))


def _group_id(
    mode: GroupedSplitMode,
    keys: tuple[str, ...],
    example_ids: tuple[str, ...],
) -> str:
    encoded = canonical_identity_json(
        {"mode": mode.value, "keys": list(keys), "example_ids": list(example_ids)}
    ).encode("utf-8")
    return f"group_{hashlib.sha256(encoded).hexdigest()}"


def _seeded_order(group_id: str, seed: int) -> bytes:
    encoded = canonical_identity_json({"seed": seed, "group_id": group_id}).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _assign_groups(
    groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    config: GroupedSplitConfig,
) -> tuple[SplitGroup, ...]:
    materialized = tuple(
        (_group_id(config.mode, keys, example_ids), keys, example_ids)
        for keys, example_ids in groups
    )
    ordered = sorted(
        materialized,
        key=lambda item: (_seeded_order(item[0], config.seed), item[0]),
    )
    total_examples = sum(len(item[2]) for item in ordered)
    ratios = {
        SplitPartition.TRAIN: config.ratios.train,
        SplitPartition.VALIDATION: config.ratios.validation,
        SplitPartition.TEST: config.ratios.test,
    }
    targets = {partition: ratios[partition] * total_examples for partition in SplitPartition}
    counts = {partition: 0 for partition in SplitPartition}
    assigned: list[SplitGroup] = []
    for group_id, keys, example_ids in ordered:
        size = len(example_ids)

        def priority(partition: SplitPartition) -> tuple[float, int]:
            target = targets[partition]
            normalized_deficit = (target - counts[partition]) / target
            return normalized_deficit, -tuple(SplitPartition).index(partition)

        partition = max(SplitPartition, key=priority)
        counts[partition] += size
        assigned.append(SplitGroup(group_id, partition, keys, example_ids))
    return tuple(sorted(assigned, key=lambda item: item.group_id))


def create_grouped_split(
    examples: tuple[MutationExampleRecord, ...],
    config: GroupedSplitConfig,
) -> GroupedSplitManifest:
    """Validate examples, form connected leakage groups, and assign whole groups."""

    if not examples:
        raise DatasetSplitError("grouped splitting requires at least one mutation example")
    validation = validate_mutation_dataset(examples)
    if not validation.valid:
        first = validation.issues[0]
        raise DatasetSplitError(
            f"dataset validation failed at record {first.record_index} "
            f"{first.path}: {first.code}: {first.detail}"
        )
    connected = _connected_groups(examples, config.mode)
    groups = _assign_groups(connected, config)
    return GroupedSplitManifest(config.mode, config.seed, config.ratios, groups)
