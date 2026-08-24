"""Coupled native GGUF MLP-channel physical mutation planning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    CouplingKind,
    GGUFDiscovery,
    GGUFTensorComponent,
    MetadataSemantic,
    build_llama_gguf_surgery_adapter,
    build_qwen_gguf_surgery_adapter,
    plan_storage_axis_edit,
    resolve_gguf_architecture,
)
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationPrecondition,
    MutationRequest,
)
from modelsurgeon.surgery.gguf_alignment import (
    GGUFQuantizationBinding,
    GGUFQuantizedMutationPlan,
    validate_gguf_quantized_plan,
)
from modelsurgeon.surgery.physical_plan import (
    AxisRemoval,
    PhysicalMetadataUpdate,
    PhysicalMutationPlan,
    PhysicalTensorDescriptor,
    TensorEditIntent,
    compile_physical_mutation_plan,
)


class NativeGGUFMLPPlanError(ValueError):
    """Raised before decode when coupled MLP channel removal is not representable."""


@dataclass(frozen=True, slots=True)
class NativeGGUFMLPRemovalPlan:
    family: ModelFamily
    architecture: str
    layer_indices: tuple[int, ...]
    removed_channels: tuple[int, ...]
    coupled_tensor_names: tuple[str, ...]
    physical_plan: PhysicalMutationPlan
    quantized_plan: GGUFQuantizedMutationPlan

    @property
    def expected_parameter_delta(self) -> int:
        return self.physical_plan.mutation_plan.expected_delta.parameters

    @property
    def expected_storage_delta(self) -> int:
        return self.physical_plan.expected_storage_delta

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "architecture": self.architecture,
            "layer_indices": list(self.layer_indices),
            "removed_channels": list(self.removed_channels),
            "coupled_tensor_names": list(self.coupled_tensor_names),
            "expected_parameter_delta": self.expected_parameter_delta,
            "expected_storage_delta": self.expected_storage_delta,
            "physical_plan": self.physical_plan.to_record(),
            "quantized_plan": self.quantized_plan.to_record(),
        }


def _channel_id(layer: int, channel: int) -> ComponentId:
    return ComponentId.parse(f"model.layers.{layer}.mlp.channel.{channel}")


def _validate_family_adapter(discovery: GGUFDiscovery, layer: int) -> None:
    if discovery.family is ModelFamily.LLAMA:
        build_llama_gguf_surgery_adapter(discovery).layer(layer)
        return
    if discovery.family is ModelFamily.QWEN:
        build_qwen_gguf_surgery_adapter(discovery).layer(layer)
        return
    raise NativeGGUFMLPPlanError(
        f"native GGUF MLP planning supports Llama and dense Qwen, not "
        f"{discovery.family.value}"
    )


def _plan_native_gguf_mlp_channel_removal(
    discovery: GGUFDiscovery,
    *,
    layer_indices: tuple[int, ...],
    removed_channels: tuple[int, ...],
) -> NativeGGUFMLPRemovalPlan:
    """Compile coupled gate/up/down edits without loading any tensor payload."""

    if (
        not layer_indices
        or layer_indices != tuple(sorted(set(layer_indices)))
        or layer_indices[0] < 0
        or layer_indices[-1] >= discovery.shape.layers
    ):
        raise NativeGGUFMLPPlanError(
            f"MLP layers must be non-empty, canonical, and inside block_count "
            f"{discovery.shape.layers}"
        )
    if (
        not removed_channels
        or removed_channels != tuple(sorted(set(removed_channels)))
        or removed_channels[0] < 0
        or removed_channels[-1] >= discovery.shape.feed_forward_length
    ):
        raise NativeGGUFMLPPlanError(
            "removed MLP channels must be non-empty, unique, in-range, and canonical"
        )
    new_feed_forward = discovery.shape.feed_forward_length - len(removed_channels)
    if new_feed_forward <= 0:
        raise NativeGGUFMLPPlanError("MLP channel removal cannot remove the entire width")
    for layer_index in layer_indices:
        _validate_family_adapter(discovery, layer_index)
    architecture = resolve_gguf_architecture(
        discovery.architecture, family=discovery.family
    )
    indexed = {tensor.descriptor.name: tensor for tensor in discovery.tensors}
    selected: list[tuple[GGUFTensorComponent, int]] = []
    for layer_index in layer_indices:
        group = next(
            item
            for item in architecture.coupling_groups(layer_index)
            if item.kind is CouplingKind.MLP_CHANNEL
        )
        for target in group.targets:
            try:
                selected.append((indexed[target.tensor_name], target.axis))
            except KeyError as error:
                raise NativeGGUFMLPPlanError(
                    f"MLP coupling requires tensor {target.tensor_name!r}"
                ) from error

    descriptors: list[PhysicalTensorDescriptor] = []
    intents: list[TensorEditIntent] = []
    bindings: list[GGUFQuantizationBinding] = []
    old_parameters = new_parameters = old_storage = new_storage = 0
    for tensor, axis in selected:
        descriptor = tensor.descriptor
        shape = descriptor.dimensions
        new_shape = list(shape)
        if shape[axis] != discovery.shape.feed_forward_length:
            raise NativeGGUFMLPPlanError(
                f"coupled tensor {descriptor.name!r} axis {axis} has size {shape[axis]}, "
                f"expected feed-forward width {discovery.shape.feed_forward_length}"
            )
        new_shape[axis] = new_feed_forward
        try:
            encoded_size = plan_storage_axis_edit(
                descriptor.quant_type, tuple(new_shape), 0
            ).tensor_bytes
        except ValueError as error:
            raise NativeGGUFMLPPlanError(
                f"new MLP width {new_feed_forward} is not block-representable for "
                f"{descriptor.name!r} ({descriptor.quant_type.value})"
            ) from error
        descriptors.append(
            PhysicalTensorDescriptor(
                tensor.component_id,
                descriptor.name,
                shape,
                descriptor.byte_size,
            )
        )
        intents.append(
            TensorEditIntent(
                tensor.component_id,
                (AxisRemoval(axis, removed_channels),),
                encoded_size,
            )
        )
        bindings.append(
            GGUFQuantizationBinding(tensor.component_id, descriptor.quant_type)
        )
        old_parameters += reduce(mul, shape, 1)
        new_parameters += reduce(mul, new_shape, 1)
        old_storage += descriptor.byte_size
        new_storage += encoded_size

    channel_ids = tuple(
        _channel_id(layer_index, index)
        for layer_index in layer_indices
        for index in range(discovery.shape.feed_forward_length)
    )
    removed_set = set(removed_channels)
    mappings: list[ComponentIdentityMapping] = []
    for layer_index in layer_indices:
        removed_before = 0
        for index in range(discovery.shape.feed_forward_length):
            source = _channel_id(layer_index, index)
            if index in removed_set:
                mappings.append(ComponentIdentityMapping(source, (), "MLP channel removed"))
                removed_before += 1
            else:
                channel_target = _channel_id(layer_index, index - removed_before)
                mappings.append(
                    ComponentIdentityMapping(
                        source,
                        (channel_target,),
                        "MLP channel retained/renumbered",
                    )
                )
    for tensor, _ in selected:
        mappings.append(
            ComponentIdentityMapping(
                tensor.component_id,
                (tensor.component_id,),
                "coupled physical tensor retained with new shape",
            )
        )
    identity_remap = ComponentIdentityRemap.build(tuple(mappings))
    request = MutationRequest(
        MutationKind.REMOVE,
        tuple(
            sorted(
                _channel_id(layer_index, index)
                for layer_index in layer_indices
                for index in removed_channels
            )
        ),
        (
            ("family", discovery.family.value),
            ("layer_indices", ",".join(str(item) for item in layer_indices)),
            ("removed_count", len(removed_channels)),
        ),
    )
    affected = tuple(
        sorted({*channel_ids, *(tensor.component_id for tensor, _ in selected)})
    )
    mutation_plan = MutationPlan(
        request,
        affected,
        (
            MutationPrecondition("architecture", discovery.architecture),
            MutationPrecondition(
                "feed_forward_length", discovery.shape.feed_forward_length
            ),
            MutationPrecondition(
                "layer_indices", ",".join(str(item) for item in layer_indices)
            ),
        ),
        MutationDelta(
            parameters=new_parameters - old_parameters,
            storage_bytes=new_storage - old_storage,
        ),
    )
    metadata_key = architecture.metadata_key(MetadataSemantic.FEED_FORWARD_LENGTH)
    physical = compile_physical_mutation_plan(
        mutation_plan,
        descriptors=tuple(sorted(descriptors, key=lambda item: item.component_id)),
        edit_intents=tuple(sorted(intents, key=lambda item: item.component_id)),
        metadata_updates=(PhysicalMetadataUpdate(metadata_key, new_feed_forward),),
        identity_remap=identity_remap,
    )
    quantized = validate_gguf_quantized_plan(
        physical,
        tuple(sorted(bindings, key=lambda item: item.component_id)),
    )
    tensor_names = tuple(sorted(tensor.descriptor.name for tensor, _ in selected))
    if sum(edit.storage_delta for edit in physical.tensor_edits) != new_storage - old_storage:
        raise NativeGGUFMLPPlanError("coupled MLP storage deltas do not reconcile")
    return NativeGGUFMLPRemovalPlan(
        discovery.family,
        discovery.architecture,
        layer_indices,
        removed_channels,
        tensor_names,
        physical,
        quantized,
    )


def plan_native_gguf_mlp_channel_removal(
    discovery: GGUFDiscovery,
    *,
    layer_index: int,
    removed_channels: tuple[int, ...],
) -> NativeGGUFMLPRemovalPlan:
    """Plan a single-layer edit only when GGUF metadata is genuinely layer-local."""

    if discovery.shape.layers != 1:
        raise NativeGGUFMLPPlanError(
            "GGUF feed-forward width metadata is model-wide; use coordinated model-wide "
            "MLP channel removal for multi-layer models"
        )
    return _plan_native_gguf_mlp_channel_removal(
        discovery,
        layer_indices=(layer_index,),
        removed_channels=removed_channels,
    )


def plan_native_gguf_model_mlp_channel_removal(
    discovery: GGUFDiscovery,
    *,
    removed_channels: tuple[int, ...],
) -> NativeGGUFMLPRemovalPlan:
    """Plan the same coupled channel removal across every transformer layer."""

    return _plan_native_gguf_mlp_channel_removal(
        discovery,
        layer_indices=tuple(range(discovery.shape.layers)),
        removed_channels=removed_channels,
    )
