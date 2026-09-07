from __future__ import annotations

import pytest

from modelsurgeon.search import (
    ArchitectureAxis,
    ArtifactContainerFormat,
    ArtifactLineage,
    AxisStatus,
    DeployableArchitectureState,
    DeployableStateError,
    DistanceStatus,
    LayerWidth,
    LowRankFactor,
    MaterializationStatus,
    PlacementState,
    QuantizationState,
    SparsityEntry,
    architecture_distance,
    deployable_state_from_record,
)


def _state(
    *, quant_codec: str = "Q8_0", placement_device: str = "cpu"
) -> DeployableArchitectureState:
    return DeployableArchitectureState(
        "llama",
        "revision-a",
        2,
        (LayerWidth(0, 64, 128), LayerWidth(1, 64, 128)),
        8,
        2,
        64,
        64,
        (LowRankFactor("model.layers.0.mlp", 16),),
        (SparsityEntry("model.layers.0.mlp", 0.25),),
        QuantizationState(quant_codec, (("model.layers.0", quant_codec),)),
        PlacementState((("model.layers.0", placement_device),)),
        ArtifactLineage(
            "a" * 64,
            ArtifactContainerFormat.GGUF,
            MaterializationStatus.COMPLETE,
            "b" * 64,
            4096,
            "outcome_" + "c" * 64,
        ),
        parameter_count=1_000,
        storage_bytes=4_096,
        mutation_order=(),
    )


def test_state_round_trip_identity_and_context_sensitive_distance() -> None:
    state = _state()
    restored = deployable_state_from_record(state.to_record())
    assert restored == state
    assert restored.state_id == state.state_id

    quantized = _state(quant_codec="Q4_K_M")
    placed = _state(placement_device="cuda:0")
    quant_distance = architecture_distance(state, quantized)
    placed_distance = architecture_distance(state, placed)
    assert quant_distance.status is DistanceStatus.COMPUTED
    assert placed_distance.status is DistanceStatus.COMPUTED
    assert quant_distance.total == architecture_distance(quantized, state).total
    assert quant_distance.total is not None and quant_distance.total > 0
    assert placed_distance.total is not None and placed_distance.total > 0
    assert state.state_id != quantized.state_id != placed.state_id
    assert quantized.state_id != placed.state_id


def test_unknown_axis_is_explicit_and_distance_is_not_claimed() -> None:
    state = _state()
    unknown = DeployableArchitectureState(
        state.model_family,
        state.model_revision,
        state.depth,
        state.layer_widths,
        state.query_heads,
        state.kv_heads,
        state.hidden_size,
        state.embedding_size,
        state.low_rank_factors,
        state.sparsity,
        None,
        state.placement,
        state.artifact,
        state.mutation_order,
        unknown_axes=(ArchitectureAxis.QUANTIZATION,),
    )
    assert unknown.axis_status(ArchitectureAxis.QUANTIZATION) is AxisStatus.UNKNOWN
    distance = architecture_distance(state, unknown)
    assert distance.status is DistanceStatus.UNKNOWN
    assert distance.total is None


def test_invalid_materialization_and_axis_status_fail_closed() -> None:
    with pytest.raises(DeployableStateError):
        ArtifactLineage(
            "a" * 64,
            ArtifactContainerFormat.GGUF,
            MaterializationStatus.COMPLETE,
            "b" * 64,
            4096,
        )
    with pytest.raises(DeployableStateError):
        DeployableArchitectureState(
            "llama",
            "revision-a",
            2,
            (LayerWidth(0, 64, 128), LayerWidth(1, 64, 128)),
            8,
            2,
            64,
            64,
            (),
            (),
            None,
            PlacementState((("model", "cpu"),)),
            ArtifactLineage(
                "a" * 64,
                ArtifactContainerFormat.GGUF,
                MaterializationStatus.PREDICTED,
                reason="not materialized",
            ),
            unknown_axes=(ArchitectureAxis.QUANTIZATION,),
            predicted_axes=(ArchitectureAxis.QUANTIZATION,),
        )
