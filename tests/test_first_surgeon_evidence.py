"""Focused contract tests for the First Surgeon empirical evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.cli.proof_evidence import (
    FirstSurgeonEvidenceConfig,
    FirstSurgeonEvidenceError,
    _rank_scores,
    load_grouped_proof_split,
)
from modelsurgeon.surgeon.ranking import RankedCandidate, RankingResult


def _split_record() -> dict[str, object]:
    return {
        "version": "1",
        "algorithm": "connected-groups-greedy-v1",
        "mode": "component",
        "seed": 7,
        "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        "groups": [
            {
                "group_id": "group-train",
                "partition": "train",
                "keys": ["component:a"],
                "example_ids": ["example-train"],
            },
            {
                "group_id": "group-validation",
                "partition": "validation",
                "keys": ["component:b"],
                "example_ids": ["example-validation"],
            },
            {
                "group_id": "group-test",
                "partition": "test",
                "keys": ["component:c"],
                "example_ids": ["example-test"],
            },
        ],
    }


def test_evidence_config_fails_closed_on_invalid_limits() -> None:
    with pytest.raises(FirstSurgeonEvidenceError, match="threads"):
        FirstSurgeonEvidenceConfig(threads=0)
    with pytest.raises(FirstSurgeonEvidenceError, match="safe_perplexity_delta"):
        FirstSurgeonEvidenceConfig(safe_perplexity_delta=-0.1)
    with pytest.raises(FirstSurgeonEvidenceError, match="bootstrap_confidence"):
        FirstSurgeonEvidenceConfig(bootstrap_confidence=1.0)


def test_grouped_proof_split_preserves_group_identity(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_split_record()), encoding="utf-8")
    manifest = load_grouped_proof_split(path)

    assert len(manifest.groups) == 3
    assert {group.group_id for group in manifest.groups} == {
        "group-train",
        "group-validation",
        "group-test",
    }
    assert manifest.example_counts


def test_grouped_proof_split_rejects_flat_assignments(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps({"version": "inline-1", "assignments": {"example": "test"}}),
        encoding="utf-8",
    )
    with pytest.raises(FirstSurgeonEvidenceError, match="version"):
        load_grouped_proof_split(path)


def test_rank_scores_requires_identical_held_out_candidate_set() -> None:
    ranking = RankingResult(
        "test",
        (
            RankedCandidate("a", 1, 0.1, 1.0),
            RankedCandidate("b", 2, 0.2, 1.0),
            RankedCandidate("c", 3, 0.3, 1.0),
        ),
    )
    assert _rank_scores(ranking, ("a", "b", "c")) == (1.0, 0.5, 0.0)

    with pytest.raises(FirstSurgeonEvidenceError, match="identical held-out"):
        _rank_scores(ranking, ("a", "b", "missing"))
