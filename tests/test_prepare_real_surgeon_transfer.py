"""Tests for preserving identity when building transfer examples from evidence."""

import copy
from pathlib import Path
from typing import Any

import pytest
from tools.prepare_real_surgeon_transfer import _iter_observations, _select_examples


def _observation() -> dict[str, Any]:
    return {
        "observation_id": "observation-1",
        "run_id": "run-1",
        "stage": "active_search",
        "outcome": "accepted",
        "model": {
            "identifier": "example/model",
            "revision": "revision-1",
            "family": "llama",
            "format": "huggingface",
        },
        "measurement": {
            "baseline_perplexity": 2.0,
            "candidate_perplexity": 2.1,
            "measurement_authority": "physical_model_evaluation",
        },
        "features": [{"name": "weight_l1_norm", "value": 1.0}],
        "lineage": {
            "mutation": {"kind": "mask", "targets": ["model.layers.0.mlp.channel.1"]}
        },
    }


def test_transfer_preparation_deduplicates_identical_source_records() -> None:
    observation = _observation()

    selected = _select_examples([observation, copy.deepcopy(observation)])

    assert list(selected) == ["observation-1"]
    assert selected["observation-1"]["post_metrics"][0]["value"] == 2.1


def test_transfer_preparation_rejects_conflicting_duplicate_ids() -> None:
    first = _observation()
    second = copy.deepcopy(first)
    second["measurement"]["candidate_perplexity"] = 1.2

    with pytest.raises(ValueError, match="conflicting duplicate transfer observation id"):
        _select_examples([first, second])


def test_transfer_preparation_rejects_conflicts_even_when_one_record_is_unusable() -> None:
    first = _observation()
    second = copy.deepcopy(first)
    del second["measurement"]

    with pytest.raises(ValueError, match="conflicting duplicate transfer observation id"):
        _select_examples([first, second])


def test_transfer_preparation_reports_corrupt_observation_path(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    observation_path = root / "optimization-evidence" / "observations" / "broken.json"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.json"):
        _iter_observations([root])


def test_transfer_preparation_rejects_missing_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="transfer evidence root does not exist"):
        _iter_observations([tmp_path / "missing"])
