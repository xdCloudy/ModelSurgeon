"""Format-neutral compilation of mutations into validated physical tensor edits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modelsurgeon.graph import ComponentId, ComponentIdentityRemap
from modelsurgeon.surgery.contracts import MutationPlan, MutationPrimitive

PHYSICAL_MUTATION_PLAN_SCHEMA_VERSION: Literal[1] = 1


class PhysicalPlanError(ValueError):
    """Raised before allocation when a physical edit plan is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class PhysicalTensorDescriptor:
    """Format-neutral physical tensor facts supplied by an adapter."""

    component_id: ComponentId
    locator: str
    shape: tuple[int, ...]
    storage_bytes: int

    def __post_init__(self) -> None:
        if not self.locator:
            raise PhysicalPlanError("physical tensor locators must be non-empty")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise PhysicalPlanError("physical tensor shapes must be non-empty and positive")
        if self.storage_bytes <= 0:
            raise PhysicalPlanError("physical tensor storage must be positive")


@dataclass(frozen=True, slots=True)
class AxisRemoval:
    axis: int
    removed_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise PhysicalPlanError("removed tensor axes must be non-negative")
        if (
            not self.removed_indices
            or self.removed_indices != tuple(sorted(set(self.removed_indices)))
            or self.removed_indices[0] < 0
        ):
            raise PhysicalPlanError(
                "removed indices must be non-empty, unique, non-negative, and canonical"
            )


@dataclass(frozen=True, slots=True)
class TensorEditIntent:
    component_id: ComponentId
    removals: tuple[AxisRemoval, ...]
    new_storage_bytes: int

    def __post_init__(self) -> None:
        axes = tuple(removal.axis for removal in self.removals)
        if not axes or axes != tuple(sorted(set(axes))):
            raise PhysicalPlanError("tensor edit axes must be non-empty, unique, and canonical")
        if self.new_storage_bytes <= 0:
            raise PhysicalPlanError("edited tensor storage must remain positive")


@dataclass(frozen=True, slots=True)
class TensorAxisTransform:
    axis: int
    old_size: int
    removed_indices: tuple[int, ...]

    @property
    def new_size(self) -> int:
        return self.old_size - len(self.removed_indices)

    def map_index(self, old_index: int) -> int | None:
        if old_index < 0 or old_index >= self.old_size:
            raise PhysicalPlanError(
                f"axis {self.axis} index {old_index} is outside old size {self.old_size}"
            )
        if old_index in self.removed_indices:
            return None
        removed_before = sum(index < old_index for index in self.removed_indices)
        return old_index - removed_before

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "removed_indices": list(self.removed_indices),
        }


@dataclass(frozen=True, slots=True)
class PhysicalTensorEdit:
    component_id: ComponentId
    locator: str
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    transforms: tuple[TensorAxisTransform, ...]
    old_storage_bytes: int
    new_storage_bytes: int

    @property
    def storage_delta(self) -> int:
        return self.new_storage_bytes - self.old_storage_bytes

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "locator": self.locator,
            "old_shape": list(self.old_shape),
            "new_shape": list(self.new_shape),
            "transforms": [item.to_record() for item in self.transforms],
            "old_storage_bytes": self.old_storage_bytes,
            "new_storage_bytes": self.new_storage_bytes,
            "storage_delta": self.storage_delta,
        }


@dataclass(frozen=True, slots=True)
class PhysicalMetadataUpdate:
    key: str
    value: MutationPrimitive

    def __post_init__(self) -> None:
        if not self.key:
            raise PhysicalPlanError("metadata update keys must be non-empty")


@dataclass(frozen=True, slots=True)
class PhysicalMutationPlan:
    mutation_plan: MutationPlan
    tensor_edits: tuple[PhysicalTensorEdit, ...]
    metadata_updates: tuple[PhysicalMetadataUpdate, ...]
    identity_remap: ComponentIdentityRemap
    schema_version: Literal[1] = PHYSICAL_MUTATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_MUTATION_PLAN_SCHEMA_VERSION:
            raise PhysicalPlanError(
                f"unsupported physical mutation plan schema {self.schema_version}"
            )

    @property
    def expected_storage_delta(self) -> int:
        return sum(edit.storage_delta for edit in self.tensor_edits)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mutation_id": self.mutation_plan.request.mutation_id,
            "affected_components": [
                str(item) for item in self.mutation_plan.affected_components
            ],
            "tensor_edits": [edit.to_record() for edit in self.tensor_edits],
            "metadata_updates": {
                update.key: update.value for update in self.metadata_updates
            },
            "identity_remap": self.identity_remap.to_record(),
            "expected_storage_delta": self.expected_storage_delta,
        }


def _validate_descriptors(
    descriptors: tuple[PhysicalTensorDescriptor, ...],
) -> dict[ComponentId, PhysicalTensorDescriptor]:
    components = tuple(item.component_id for item in descriptors)
    locators = tuple(item.locator for item in descriptors)
    if components != tuple(sorted(components)) or len(components) != len(set(components)):
        raise PhysicalPlanError("physical descriptors must have unique canonical components")
    if len(locators) != len(set(locators)):
        raise PhysicalPlanError("physical tensor locators must be unique")
    return {item.component_id: item for item in descriptors}


def compile_physical_mutation_plan(
    mutation_plan: MutationPlan,
    *,
    descriptors: tuple[PhysicalTensorDescriptor, ...],
    edit_intents: tuple[TensorEditIntent, ...],
    metadata_updates: tuple[PhysicalMetadataUpdate, ...],
    identity_remap: ComponentIdentityRemap,
) -> PhysicalMutationPlan:
    """Validate the complete plan and derive transforms without allocating tensors."""

    descriptor_by_component = _validate_descriptors(descriptors)
    intent_components = tuple(intent.component_id for intent in edit_intents)
    if (
        not intent_components
        or intent_components != tuple(sorted(intent_components))
        or len(intent_components) != len(set(intent_components))
    ):
        raise PhysicalPlanError("edit intents must be non-empty, unique, and canonical")
    affected = set(mutation_plan.affected_components)
    unexpected = tuple(sorted(set(intent_components) - affected))
    if unexpected:
        raise PhysicalPlanError(
            "physical edits target components outside mutation closure: "
            + ", ".join(map(str, unexpected))
        )
    required = affected.intersection(descriptor_by_component)
    missing = tuple(sorted(required - set(intent_components)))
    if missing:
        raise PhysicalPlanError(
            "affected physical tensors have no edit intent: "
            + ", ".join(map(str, missing))
        )

    edits: list[PhysicalTensorEdit] = []
    for intent in edit_intents:
        try:
            descriptor = descriptor_by_component[intent.component_id]
        except KeyError as error:
            raise PhysicalPlanError(
                f"edit target {intent.component_id} has no physical descriptor"
            ) from error
        new_shape = list(descriptor.shape)
        transforms: list[TensorAxisTransform] = []
        for removal in intent.removals:
            if removal.axis >= len(descriptor.shape):
                raise PhysicalPlanError(
                    f"edit axis {removal.axis} escapes tensor {descriptor.locator!r} "
                    f"rank {len(descriptor.shape)}"
                )
            old_size = descriptor.shape[removal.axis]
            if removal.removed_indices[-1] >= old_size:
                raise PhysicalPlanError(
                    f"removed index {removal.removed_indices[-1]} escapes axis size {old_size}"
                )
            transform = TensorAxisTransform(
                removal.axis, old_size, removal.removed_indices
            )
            if transform.new_size <= 0:
                raise PhysicalPlanError("physical edits cannot remove an entire tensor axis")
            new_shape[removal.axis] = transform.new_size
            transforms.append(transform)
        edits.append(
            PhysicalTensorEdit(
                descriptor.component_id,
                descriptor.locator,
                descriptor.shape,
                tuple(new_shape),
                tuple(transforms),
                descriptor.storage_bytes,
                intent.new_storage_bytes,
            )
        )

    update_keys = tuple(update.key for update in metadata_updates)
    if update_keys != tuple(sorted(set(update_keys))):
        raise PhysicalPlanError("metadata updates must have unique canonical keys")
    remapped_sources = {mapping.source for mapping in identity_remap.mappings}
    missing_mappings = tuple(sorted(affected - remapped_sources))
    if missing_mappings:
        raise PhysicalPlanError(
            "affected components lack post-surgery identity mappings: "
            + ", ".join(map(str, missing_mappings))
        )
    compiled = PhysicalMutationPlan(
        mutation_plan,
        tuple(edits),
        metadata_updates,
        identity_remap,
    )
    if compiled.expected_storage_delta != mutation_plan.expected_delta.storage_bytes:
        raise PhysicalPlanError(
            f"compiled storage delta {compiled.expected_storage_delta} disagrees with "
            f"mutation plan {mutation_plan.expected_delta.storage_bytes}"
        )
    return compiled
