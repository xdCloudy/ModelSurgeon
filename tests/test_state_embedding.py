"""Tests for fixed-width current-state representations."""

from __future__ import annotations

import pytest

from modelsurgeon.search import (
    ArtifactContainerFormat,
    ArtifactLineage,
    DeployableArchitectureState,
    LayerWidth,
    MaterializationStatus,
    PlacementState,
    QuantizationState,
)
from modelsurgeon.surgeon import (
    CurrentStateInput,
    StateConstraintObservation,
    StateEmbeddingConfig,
    StateEmbeddingError,
    StateEvidenceObservation,
    StateEvidenceOutcome,
    StateMetricObservation,
    StateObservationStatus,
    encode_current_state,
)


def _state(history: tuple[str, ...] = ()) -> DeployableArchitectureState:
    return DeployableArchitectureState(
        "llama",
        "revision-a",
        2,
        (LayerWidth(0, 64, 128), LayerWidth(1, 96, 192)),
        8,
        2,
        64,
        64,
        (),
        (),
        QuantizationState("Q8_0"),
        PlacementState((("model.layers", "cpu"),)),
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
        mutation_order=history,
    )


def _current(state: DeployableArchitectureState) -> CurrentStateInput:
    return CurrentStateInput(
        state,
        accepted_mutations=("B", "A"),
        cumulative_metrics=(
            StateMetricObservation("latency_seconds", StateObservationStatus.UNKNOWN),
            StateMetricObservation("quality_loss", StateObservationStatus.KNOWN, 0.2),
        ),
        remaining_constraints=(
            StateConstraintObservation("ram", StateObservationStatus.KNOWN, 0.5),
        ),
        evidence_outcomes=(
            StateEvidenceObservation("failed-cell", StateEvidenceOutcome.FAILED),
            StateEvidenceObservation("measured-cell", StateEvidenceOutcome.MEASURED),
        ),
    )


def test_encoding_is_fixed_width_and_set_inputs_are_order_invariant() -> None:
    first = encode_current_state(_current(_state(("A", "B"))))
    reordered = encode_current_state(_current(_state(("A", "B"))))

    assert len(first.values) == len(first.masks) == len(first.feature_names)
    assert first.values == reordered.values
    assert first.masks == reordered.masks
    assert first.embedding_id == reordered.embedding_id
    assert (
        dict(zip(first.feature_names, first.values, strict=True))["metric_latency_seconds_status"]
        == 0.25
    )
    assert (
        dict(zip(first.feature_names, first.masks, strict=True))["metric_latency_seconds_value"]
        == 0.0
    )


def test_order_and_long_history_are_explicitly_distinct_and_bounded() -> None:
    config = StateEmbeddingConfig(max_history=2, mutation_buckets=8, history_buckets=8)
    forward = encode_current_state(_current(_state(("A", "B", "C"))), config)
    reverse = encode_current_state(_current(_state(("A", "C", "B"))), config)

    assert forward.history_length == 3
    assert forward.history_truncated_count == 1
    assert forward.history_prefix_digest is not None
    assert forward.embedding_id != reverse.embedding_id
    assert len(forward.values) == len(reverse.values)


def test_invalid_or_unbounded_inputs_fail_closed() -> None:
    with pytest.raises(StateEmbeddingError, match="schema version"):
        StateEmbeddingConfig(schema_version=99)

    with pytest.raises(StateEmbeddingError, match="accepted mutation set"):
        encode_current_state(
            CurrentStateInput(_state(), accepted_mutations=tuple(f"m-{i}" for i in range(3))),
            StateEmbeddingConfig(max_set_items=2),
        )

    with pytest.raises(StateEmbeddingError, match="known state metrics"):
        StateMetricObservation("latency_seconds", StateObservationStatus.KNOWN)
