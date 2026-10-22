"""Tests for preserving identity when building transfer examples from evidence."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from tools.prepare_real_surgeon_transfer import (
    _iter_observations,
    _publish_dataset,
    _select_examples,
    main,
)


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


def test_transfer_dataset_publishes_complete_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "transfer"

    _publish_dataset(
        output,
        {"observation-1": {"example_id": "observation-1"}},
        {"mode": "model_identity"},
        {"example_count": 1},
    )

    assert json.loads((output / "examples.jsonl").read_text(encoding="utf-8")) == {
        "example_id": "observation-1"
    }
    assert json.loads((output / "split.json").read_text(encoding="utf-8")) == {
        "mode": "model_identity"
    }
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == {
        "example_count": 1
    }
    assert not tuple(output.parent.glob(".transfer.partial-*"))


def test_transfer_dataset_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "transfer"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="output already exists"):
        _publish_dataset(output, {}, {}, {})

    assert marker.read_text(encoding="utf-8") == "existing"


def test_transfer_dataset_write_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "transfer"
    original_write_text = Path.write_text

    def fail_manifest(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == "manifest.json":
            raise OSError("simulated manifest write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        _publish_dataset(
            output,
            {"observation-1": {"example_id": "observation-1"}},
            {},
            {},
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".transfer.partial-*"))


def test_transfer_preparation_cli_publishes_manifest_and_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    observations = root / "campaign" / "optimization-evidence" / "observations"
    observations.mkdir(parents=True)
    (observations / "observation.json").write_text(
        json.dumps(_observation()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_real_surgeon_transfer", "--root", str(root), "--output", "transfer"],
    )

    assert main() == 0

    manifest = json.loads(Path("transfer/manifest.json").read_text(encoding="utf-8"))
    split = json.loads(Path("transfer/split.json").read_text(encoding="utf-8"))
    assert manifest["example_count"] == 1
    assert manifest["dataset"] == str(Path("transfer") / "examples.jsonl")
    assert split["assignments"] == {"observation-1": "train"}
