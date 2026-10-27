from __future__ import annotations

import copy
import hashlib
import json

import pytest
from tools.compare_reap_profile_aggregation import _load as load_aggregation_profile
from tools.compare_reap_profile_aggregation import compare_aggregations
from tools.compare_reap_profiles import _load as load_pairwise_profile


def _profile(name: str, scores: list[list[float]]) -> dict[str, object]:
    return {
        "name": name,
        "profile_sha256": name,
        "record": {
            "model_fingerprint": "model-a",
            "arch": "qwen35moe",
            "n_layers": len(scores),
            "n_experts": len(scores[0]),
            "top_k": 2,
            "tokens": 128,
            "dataset_bytes": 1024,
            "dataset_hash": name,
            "layers": [{"reap_mean": layer} for layer in scores],
        },
    }


def test_identical_profiles_have_perfect_leave_one_out_stability() -> None:
    profile = _profile("profile-a", [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    profiles = [profile, copy.deepcopy(profile), copy.deepcopy(profile)]
    result = compare_aggregations(profiles, prune_fraction=0.5)

    assert result["single_profile_pairwise_mean_jaccard"] == 1.0
    assert result["single_profile_worst_held_out_mean_jaccard"] == 1.0
    for aggregate in result["aggregation_results"].values():
        assert aggregate["leave_one_profile_out_mean_jaccard"] == 1.0
        assert aggregate["worst_held_out_profile_mean_jaccard"] == 1.0


def test_leave_one_out_emits_one_fold_per_calibration_profile() -> None:
    profiles = [
        _profile("a", [[0.1, 0.2, 0.3, 0.4]]),
        _profile("b", [[0.2, 0.1, 0.4, 0.3]]),
        _profile("c", [[0.3, 0.4, 0.1, 0.2]]),
    ]
    result = compare_aggregations(profiles, prune_fraction=0.5)

    assert result["experts_pruned_per_layer"] == 2
    assert len(result["aggregation_results"]["mean_minmax"]["folds"]) == 3
    assert [
        fold["held_out_profile"] for fold in result["aggregation_results"]["mean_rank"]["folds"]
    ] == ["a", "b", "c"]


def test_leave_one_out_rejects_incompatible_profiles() -> None:
    profiles = [
        _profile("a", [[0.1, 0.2, 0.3, 0.4]]),
        _profile("b", [[0.2, 0.1, 0.4, 0.3]]),
        _profile("c", [[0.3, 0.4, 0.1, 0.2]]),
    ]
    profiles[2]["record"]["model_fingerprint"] = "other-model"

    with pytest.raises(ValueError, match="model_fingerprint"):
        compare_aggregations(profiles)


def test_leave_one_out_requires_three_profiles() -> None:
    profiles = [
        _profile("a", [[0.1, 0.2, 0.3, 0.4]]),
        _profile("b", [[0.2, 0.1, 0.4, 0.3]]),
    ]

    with pytest.raises(ValueError, match="at least three profiles"):
        compare_aggregations(profiles)


def test_profile_hashes_are_stable_across_lf_and_crlf_checkouts(tmp_path) -> None:
    record = {
        "poeprofile": 0,
        "model_fingerprint": "model-a",
        "arch": "qwen35moe",
        "n_layers": 1,
        "n_experts": 4,
        "top_k": 2,
        "tokens": 128,
        "dataset_bytes": 4,
        "dataset_hash": "dataset-a",
        "layers": [{"reap_mean": [0.1, 0.2, 0.3, 0.4]}],
    }
    lf_bytes = (json.dumps(record, indent=2) + "\n").encode()
    lf_path = tmp_path / "profile-lf.poeprofile"
    crlf_path = tmp_path / "profile-crlf.poeprofile"
    dataset_path = tmp_path / "dataset.bin"
    lf_path.write_bytes(lf_bytes)
    crlf_path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    dataset_path.write_bytes(b"data")

    expected = hashlib.sha256(lf_bytes).hexdigest()
    assert load_aggregation_profile(lf_path)["profile_sha256"] == expected
    assert load_aggregation_profile(crlf_path)["profile_sha256"] == expected
    assert load_pairwise_profile(lf_path, dataset_path)["profile_sha256"] == expected
    assert load_pairwise_profile(crlf_path, dataset_path)["profile_sha256"] == expected


def test_profile_loader_rejects_non_object_json(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported POE profile"):
        load_aggregation_profile(profile_path)
