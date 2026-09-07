from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.surgery import (
    ArtifactState,
    PhysicalOutcomeError,
    PhysicalOutcomeSequence,
    build_example_physical_outcomes,
    import_physical_outcome,
)


def test_hf_and_native_gguf_examples_round_trip_through_typed_import() -> None:
    hf, gguf = build_example_physical_outcomes()

    assert import_physical_outcome(hf.to_record()) == hf
    assert import_physical_outcome(gguf.to_record()) == gguf
    assert hf.artifact.state is ArtifactState.COMPLETE
    assert gguf.artifact.format.value == "GGUF"


def test_cumulative_sequence_reconciles_without_double_counting() -> None:
    hf, _ = build_example_physical_outcomes()
    second_tensor = replace(
        hf.tensor_outcomes[0],
        old_storage_bytes=3600,
        new_storage_bytes=3200,
        old_shape=(90, 10),
        new_shape=(80, 10),
    )
    second = replace(
        hf,
        parent_outcome_id=hf.outcome_id,
        mutation_id="mutation_hf_stage_2",
        architecture=replace(hf.architecture, active_parameters=800),
        tensor_outcomes=(second_tensor,),
        stage_parameter_delta=-100,
        stage_storage_delta=-400,
        cumulative_parameter_delta=-200,
        cumulative_storage_delta=-800,
    )

    report = PhysicalOutcomeSequence(1000, 4000, (hf, second)).reconcile()
    assert report.valid
    assert report.cumulative_parameter_delta == -200
    assert report.cumulative_storage_delta == -800


def test_unknown_schema_and_incomplete_artifact_fail_closed() -> None:
    hf, _ = build_example_physical_outcomes()
    unknown = hf.to_record()
    unknown["schema_version"] = 99
    with pytest.raises(PhysicalOutcomeError, match="unknown"):
        import_physical_outcome(unknown)

    incomplete = hf.to_record()
    artifact = dict(incomplete["artifact"])  # type: ignore[arg-type]
    artifact["state"] = "incomplete"
    artifact["digest"] = None
    artifact["size_bytes"] = None
    artifact["reason"] = "publication was interrupted"
    incomplete["artifact"] = artifact
    with pytest.raises(PhysicalOutcomeError, match="outcome ID"):
        import_physical_outcome(incomplete)
