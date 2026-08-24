"""Dependency-free contracts for the empirical First Surgeon evidence report."""

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
from modelsurgeon.cli.surgeon import load_split_assignments
from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.surgeon.ranking import RankedCandidate, RankingResult


def _split_record() -> dict[str, object]:
    return {
        "version": "1",
        "algorithm": "connected-groups-greedy-v1",
        "mode": "component",
        "seed": 43,
        "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        "groups": [
            {
                "group_id": "group-a",
                "partition": "train",
                "keys": ["component:model.layers.0.mlp.channel.0"],
                "example_ids": ["example-a"],
            },
            {
                "group_id": "group-b",
                "partition": "validation",
                "keys": ["component:model.layers.0.mlp.channel.1"],
                "example_ids": ["example-b"],
            },
            {
                "group_id": "group-c",
                "partition": "test",
                "keys": ["component:model.layers.1.mlp.channel.0"],
                "example_ids": ["example-c"],
            },
            {
                "group_id": "group-d",
                "partition": "test",
                "keys": ["component:model.layers.1.mlp.channel.1"],
                "example_ids": ["example-d"],
            },
        ],
    }


def test_evidence_config_rejects_invalid_lightgbm_limits() -> None:
    with pytest.raises(FirstSurgeonEvidenceError, match="threads"):
        FirstSurgeonEvidenceConfig(threads=0)
    with pytest.raises(FirstSurgeonEvidenceError, match="seed"):
        FirstSurgeonEvidenceConfig(seed=-1)
    with pytest.raises(FirstSurgeonEvidenceError, match="bootstrap_confidence"):
        FirstSurgeonEvidenceConfig(bootstrap_confidence=1.0)


def test_grouped_proof_split_round_trips_group_identity(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_split_record()), encoding="utf-8")

    manifest = load_grouped_proof_split(path)

    assert manifest.seed == 43
    assert manifest.group_counts[SplitPartition.TEST] == 2
    assert manifest.example_counts[SplitPartition.TEST] == 2
    assert manifest.example_ids(SplitPartition.TEST) == ("example-c", "example-d")
    assert {group.group_id for group in manifest.groups} == {
        "group-a",
        "group-b",
        "group-c",
        "group-d",
    }


def test_train_cli_preserves_grouped_manifest_identity(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_split_record()), encoding="utf-8")

    loaded = load_split_assignments(path)

    assert isinstance(loaded, GroupedSplitManifest)
    assert loaded.group_counts[SplitPartition.TEST] == 2
    assert tuple(group.group_id for group in loaded.groups) == (
        "group-a",
        "group-b",
        "group-c",
        "group-d",
    )


def test_rank_scores_preserve_requested_held_out_order() -> None:
    ranking = RankingResult(
        "test",
        (
            RankedCandidate("example-b", 1, 0.1, 1.0),
            RankedCandidate("example-a", 2, 0.2, 1.0),
            RankedCandidate("example-c", 3, 0.3, 1.0),
        ),
    )

    scores = _rank_scores(ranking, ("example-c", "example-a", "example-b"))

    assert scores == pytest.approx((0.0, 0.5, 1.0))
