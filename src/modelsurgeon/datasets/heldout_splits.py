"""Deterministic held-out splits across complete models and architecture families."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.datasets.validation import validate_mutation_dataset
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import MutationExampleRecord

HELDOUT_SPLIT_VERSION = "1"
HELDOUT_SPLIT_ALGORITHM = "heldout-groups-v1"


class HeldOutSplitError(ValueError):
    """Raised when model/family holdout constraints cannot be satisfied safely."""


class HeldOutSplitMode(StrEnum):
    MODEL = "model"
    ARCHITECTURE_FAMILY = "architecture_family"


class HeldOutAssignmentStrategy(StrEnum):
    SEEDED_RATIOS = "seeded_ratios"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class HeldOutSplitConfig:
    mode: HeldOutSplitMode
    seed: int = 0
    ratios: SplitRatios = field(default_factory=SplitRatios)
    validation_holdouts: tuple[str, ...] = ()
    test_holdouts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise HeldOutSplitError("held-out split seed must be an unsigned 64-bit integer")
        for name, values in (
            ("validation_holdouts", self.validation_holdouts),
            ("test_holdouts", self.test_holdouts),
        ):
            if values != tuple(sorted(set(values))) or any(not value for value in values):
                raise HeldOutSplitError(f"{name} must contain unique canonical non-empty values")
        overlap = set(self.validation_holdouts) & set(self.test_holdouts)
        if overlap:
            raise HeldOutSplitError(
                f"validation and test holdouts must be disjoint; overlap={sorted(overlap)}"
            )

    @property
    def strategy(self) -> HeldOutAssignmentStrategy:
        if self.validation_holdouts or self.test_holdouts:
            return HeldOutAssignmentStrategy.EXPLICIT
        return HeldOutAssignmentStrategy.SEEDED_RATIOS


@dataclass(frozen=True, slots=True)
class HeldOutGroup:
    key: str
    partition: SplitPartition
    model_identifiers: tuple[str, ...]
    model_revisions: tuple[str, ...]
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.key
            or not self.model_identifiers
            or not self.model_revisions
            or not self.example_ids
        ):
            raise HeldOutSplitError("held-out groups require key, model provenance, and examples")
        for values in (self.model_identifiers, self.model_revisions, self.example_ids):
            if values != tuple(sorted(set(values))):
                raise HeldOutSplitError("held-out group values must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "key": self.key,
            "partition": self.partition.value,
            "model_identifiers": list(self.model_identifiers),
            "model_revisions": list(self.model_revisions),
            "example_ids": list(self.example_ids),
        }


@dataclass(frozen=True, slots=True)
class HeldOutSplitManifest:
    mode: HeldOutSplitMode
    strategy: HeldOutAssignmentStrategy
    seed: int
    ratios: SplitRatios
    validation_holdouts: tuple[str, ...]
    test_holdouts: tuple[str, ...]
    groups: tuple[HeldOutGroup, ...]
    version: str = HELDOUT_SPLIT_VERSION
    algorithm: str = HELDOUT_SPLIT_ALGORITHM

    def __post_init__(self) -> None:
        if self.version != HELDOUT_SPLIT_VERSION or self.algorithm != HELDOUT_SPLIT_ALGORITHM:
            raise HeldOutSplitError("unsupported held-out split manifest version or algorithm")
        keys = tuple(group.key for group in self.groups)
        if keys != tuple(sorted(set(keys))):
            raise HeldOutSplitError("held-out group keys must be unique and canonical")
        examples = tuple(example for group in self.groups for example in group.example_ids)
        if len(examples) != len(set(examples)):
            raise HeldOutSplitError("an example cannot appear in multiple held-out groups")

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
        raise HeldOutSplitError(f"example {example_id!r} is absent from the held-out manifest")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "mode": self.mode.value,
            "strategy": self.strategy.value,
            "seed": self.seed,
            "ratios": self.ratios.to_record(),
            "validation_holdouts": list(self.validation_holdouts),
            "test_holdouts": list(self.test_holdouts),
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


@dataclass(frozen=True, slots=True)
class _PendingGroup:
    key: str
    model_identifiers: tuple[str, ...]
    model_revisions: tuple[str, ...]
    example_ids: tuple[str, ...]


def _family(example: MutationExampleRecord) -> ModelFamily:
    try:
        return ModelFamily(example.model.family)
    except ValueError as error:
        supported = ", ".join(family.value for family in ModelFamily)
        raise HeldOutSplitError(
            f"model {example.model.identifier!r} has unsupported architecture family "
            f"{example.model.family!r}; expected canonical #11 family in {{{supported}}}"
        ) from error


def heldout_key(example: MutationExampleRecord, mode: HeldOutSplitMode) -> str:
    """Return the canonical complete-model or architecture-family holdout key."""

    if mode is HeldOutSplitMode.MODEL:
        return example.model.identifier
    return _family(example).value


def _materialize_groups(
    examples: tuple[MutationExampleRecord, ...],
    mode: HeldOutSplitMode,
) -> tuple[_PendingGroup, ...]:
    family_by_model: dict[str, ModelFamily] = {}
    indexes_by_key: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        family = _family(example)
        previous = family_by_model.setdefault(example.model.identifier, family)
        if previous is not family:
            raise HeldOutSplitError(
                f"model {example.model.identifier!r} has conflicting canonical families "
                f"{previous.value!r} and {family.value!r}"
            )
        key = heldout_key(example, mode)
        indexes_by_key.setdefault(key, []).append(index)

    groups: list[_PendingGroup] = []
    for key, indexes in indexes_by_key.items():
        groups.append(
            _PendingGroup(
                key,
                tuple(sorted({examples[index].model.identifier for index in indexes})),
                tuple(sorted({examples[index].model.revision for index in indexes})),
                tuple(sorted(examples[index].example_id for index in indexes)),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.key))


def _seeded_order(key: str, seed: int) -> bytes:
    encoded = canonical_identity_json({"seed": seed, "heldout_key": key}).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _seeded_assign(
    groups: tuple[_PendingGroup, ...],
    config: HeldOutSplitConfig,
) -> tuple[HeldOutGroup, ...]:
    if len(groups) < len(tuple(SplitPartition)):
        noun = (
            "architecture families"
            if config.mode is HeldOutSplitMode.ARCHITECTURE_FAMILY
            else "models"
        )
        raise HeldOutSplitError(
            f"seeded {config.mode.value} split needs at least 3 distinct {noun} for "
            f"train/validation/test; found {len(groups)}. Add more {noun} or use explicit holdouts."
        )
    ordered = sorted(groups, key=lambda group: (_seeded_order(group.key, config.seed), group.key))
    total = sum(len(group.example_ids) for group in ordered)
    ratios = {
        SplitPartition.TRAIN: config.ratios.train,
        SplitPartition.VALIDATION: config.ratios.validation,
        SplitPartition.TEST: config.ratios.test,
    }
    targets = {partition: ratios[partition] * total for partition in SplitPartition}
    counts = {partition: 0 for partition in SplitPartition}
    assigned: list[HeldOutGroup] = []
    for group in ordered:

        def priority(partition: SplitPartition) -> tuple[float, int]:
            deficit = (targets[partition] - counts[partition]) / targets[partition]
            return deficit, -tuple(SplitPartition).index(partition)

        partition = max(SplitPartition, key=priority)
        counts[partition] += len(group.example_ids)
        assigned.append(
            HeldOutGroup(
                group.key,
                partition,
                group.model_identifiers,
                group.model_revisions,
                group.example_ids,
            )
        )
    return tuple(sorted(assigned, key=lambda group: group.key))


def _explicit_assign(
    groups: tuple[_PendingGroup, ...],
    config: HeldOutSplitConfig,
) -> tuple[HeldOutGroup, ...]:
    available = {group.key for group in groups}
    requested = set(config.validation_holdouts) | set(config.test_holdouts)
    missing = sorted(requested - available)
    if missing:
        raise HeldOutSplitError(
            f"requested held-out {config.mode.value} keys are absent: {missing}; "
            f"available keys are {sorted(available)}"
        )
    remaining = available - requested
    if not remaining:
        raise HeldOutSplitError(
            "explicit held-out split leaves no training group; "
            "keep at least one model/family unheld"
        )
    assigned: list[HeldOutGroup] = []
    validation = set(config.validation_holdouts)
    test = set(config.test_holdouts)
    for group in groups:
        if group.key in test:
            partition = SplitPartition.TEST
        elif group.key in validation:
            partition = SplitPartition.VALIDATION
        else:
            partition = SplitPartition.TRAIN
        assigned.append(
            HeldOutGroup(
                group.key,
                partition,
                group.model_identifiers,
                group.model_revisions,
                group.example_ids,
            )
        )
    return tuple(assigned)


def create_heldout_split(
    examples: tuple[MutationExampleRecord, ...],
    config: HeldOutSplitConfig,
) -> HeldOutSplitManifest:
    """Create complete-model or complete-family holdouts after dataset validation."""

    if not examples:
        raise HeldOutSplitError("held-out splitting requires at least one mutation example")
    validation = validate_mutation_dataset(examples)
    if not validation.valid:
        first = validation.issues[0]
        raise HeldOutSplitError(
            f"dataset validation failed at record {first.record_index} "
            f"{first.path}: {first.code}: {first.detail}"
        )
    groups = _materialize_groups(examples, config.mode)
    if config.strategy is HeldOutAssignmentStrategy.EXPLICIT:
        assigned = _explicit_assign(groups, config)
    else:
        assigned = _seeded_assign(groups, config)
    return HeldOutSplitManifest(
        config.mode,
        config.strategy,
        config.seed,
        config.ratios,
        config.validation_holdouts,
        config.test_holdouts,
        assigned,
    )
