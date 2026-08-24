from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.active_learning import (
    CandidatePoolError,
    CandidatePoolProvenance,
    write_candidate_pool,
)
from modelsurgeon.experiments.candidates import (
    CandidateEnumerationError,
    CandidateEnumeratorConfig,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.identity import derive_run_identity
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode


def _candidates():
    nodes = tuple(
        GraphNode(
            ComponentId.parse(f"model.layers.{index}.mlp.channel.0"),
            "mlp_channel",
            (("channel_index", 0), ("layer_index", index)),
        )
        for index in range(5)
    )
    run_id = derive_run_identity("exp_" + "a" * 64).run_id
    return enumerate_mutation_candidates(
        ComponentGraph.build(nodes), run_id, CandidateEnumeratorConfig(seed=7)
    ).candidates


def _provenance() -> CandidatePoolProvenance:
    return CandidatePoolProvenance(
        derive_run_identity("exp_" + "a" * 64).run_id,
        "sha256:graph",
        "model-revision",
        "tool-revision",
        7,
    )


def test_pool_is_bounded_resumable_and_carries_features_and_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pool.jsonl"
    first = write_candidate_pool(
        _candidates(), output, _provenance(), max_candidates=4, max_new_candidates=2
    )
    resumed = write_candidate_pool(_candidates(), output, _provenance(), max_candidates=4)

    assert first.candidate_count == 2 and not first.complete
    assert resumed.candidate_count == 4 and resumed.complete
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(record["cheap_features"]["affected_component_count"] == 1 for record in records)
    assert all(record["provenance"]["tool_revision"] == "tool-revision" for record in records)
    assert resumed.content_sha256


def test_resume_fails_closed_when_provenance_changes(tmp_path: Path) -> None:
    output = tmp_path / "pool.jsonl"
    write_candidate_pool(_candidates(), output, _provenance(), max_new_candidates=1)
    changed = CandidatePoolProvenance(
        _provenance().run_id, "different", "model-revision", "tool-revision", 7
    )

    with pytest.raises(CandidatePoolError, match="provenance changed"):
        write_candidate_pool(_candidates(), output, changed)


def test_enumerator_and_pool_reject_more_than_100000() -> None:
    with pytest.raises(CandidateEnumerationError, match="100000"):
        CandidateEnumeratorConfig(seed=0, max_candidates=100_001)
    with pytest.raises(CandidatePoolError, match="100000"):
        write_candidate_pool(_candidates(), Path("unused"), _provenance(), max_candidates=100_001)
