"""Tests for paired EvalPlus pruning-result analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tools.analyze_evalplus_pruning_pairs import (
    _exact_mcnemar_p_value,
    _paired_bootstrap_interval,
    _paired_outcomes,
    build_report,
)


def _run_report(
    candidate: list[str],
    *,
    source_hash: str = "same-source",
    implementation: dict[str, object] | None = None,
) -> dict[str, object]:
    task_ids = [f"task-{index}" for index in range(len(candidate))]
    source = {task_id: {"plus_status": "pass"} for task_id in task_ids[:2]}
    source.update({task_id: {"plus_status": "fail"} for task_id in task_ids[2:]})
    results = {
        task_id: {"plus_status": status}
        for task_id, status in zip(task_ids, candidate, strict=True)
    }
    quality = {
        "per_task": source,
        "evalplus_source_revision": "evalplus-revision",
        "docker_image_id": "sha256:docker-image",
    }
    report: dict[str, object] = {
        "model": {"revision": "model-revision"},
        "dataset": {"sha256": "dataset-sha256"},
        "method": {"removed_channels": [0, 1]},
        "arms": {
            "source": {
                "generation": {"sample_file_sha256": source_hash, "runtime": None},
                "task_quality": quality,
            },
            "pruned": {
                "generation": {"runtime": None},
                "task_quality": {"per_task": results},
            },
        },
    }
    if implementation is not None:
        report["implementation"] = implementation
    return report


def test_paired_outcomes_records_each_discordant_result() -> None:
    left = {"a": "pass", "b": "pass", "c": "fail", "d": "fail"}
    right = {"a": "pass", "b": "fail", "c": "pass", "d": "fail"}

    assert _paired_outcomes(left, right) == (1, 1, 1, 1, [0, 1, -1, 0])


def test_exact_mcnemar_uses_two_sided_exact_binomial_probability() -> None:
    assert _exact_mcnemar_p_value(1, 0) == 1.0
    assert _exact_mcnemar_p_value(3, 0) == 0.25
    assert _exact_mcnemar_p_value(2, 2) == 1.0


def test_paired_bootstrap_is_reproducible_and_exact_for_constant_differences() -> None:
    assert _paired_bootstrap_interval([1, 1, 1], samples=100) == [100.0, 100.0]
    assert _paired_bootstrap_interval([1, 0, -1], samples=100) == _paired_bootstrap_interval(
        [1, 0, -1], samples=100
    )


def test_build_report_checks_controls_and_hashes_sources(tmp_path: Path) -> None:
    paths = {
        "magnitude": tmp_path / "magnitude.json",
        "activation": tmp_path / "activation.json",
    }
    values = {
        "magnitude": _run_report(["pass", "fail", "pass", "fail"]),
        "activation": _run_report(["pass", "pass", "fail", "fail"]),
    }
    for name, path in paths.items():
        path.write_text(json.dumps(values[name]), encoding="utf-8")

    report = build_report(paths, bootstrap_samples=100)

    assert report["implementation_provenance"] == {
        "status": "legacy_not_recorded",
        "git_revision": None,
    }
    assert report["schema_version"] == 3
    assert report["controls"]["task_count"] == 4
    assert report["run_metadata"]["magnitude"]["candidate_generated_tokens_per_second"] is None
    assert report["method"]["comparison_scope"] == "reference candidate pairs"
    assert [item["left"] + "-" + item["right"] for item in report["comparisons"]] == [
        "source-magnitude",
        "source-activation",
        "magnitude-activation",
    ]
    activation_vs_magnitude = report["comparisons"][2]
    assert activation_vs_magnitude["left_minus_right_percentage_points"] == 0.0
    assert activation_vs_magnitude["discordant_tasks"] == {
        "left_only_pass": 1,
        "right_only_pass": 1,
    }
    for item in report["source_reports"]:
        normalized = paths[item["name"]].read_bytes().replace(b"\r\n", b"\n")
        assert item["sha256_lf_normalized"] == hashlib.sha256(normalized).hexdigest()


def test_build_report_records_shared_clean_implementation_revision(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("magnitude", "activation")}
    status_sha256 = hashlib.sha256(b"").hexdigest()
    for path in paths.values():
        path.write_text(
            json.dumps(
                _run_report(
                    ["pass", "fail", "pass", "fail"],
                    implementation={
                        "name": "ModelSurgeon",
                        "git_revision": "a" * 40,
                        "source_tree_clean": True,
                        "changed_source_path_count": 0,
                        "source_status_sha256": status_sha256,
                    },
                )
            ),
            encoding="utf-8",
        )

    report = build_report(paths, bootstrap_samples=100)

    assert report["implementation_provenance"] == {
        "status": "verified_clean",
        "git_revision": "a" * 40,
    }


def test_build_report_rejects_different_model_surgeon_revisions(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("magnitude", "activation")}
    status_sha256 = hashlib.sha256(b"").hexdigest()
    for name, path in paths.items():
        path.write_text(
            json.dumps(
                _run_report(
                    ["pass", "fail", "pass", "fail"],
                    implementation={
                        "name": "ModelSurgeon",
                        "git_revision": "a" * 40 if name == "magnitude" else "b" * 40,
                        "source_tree_clean": True,
                        "changed_source_path_count": 0,
                        "source_status_sha256": status_sha256,
                    },
                )
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="different ModelSurgeon Git revisions"):
        build_report(paths, bootstrap_samples=100)


def test_build_report_rejects_dirty_implementation_provenance(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("magnitude", "activation")}
    status_sha256 = hashlib.sha256(b" M src/module.py\0").hexdigest()
    for path in paths.values():
        path.write_text(
            json.dumps(
                _run_report(
                    ["pass", "fail", "pass", "fail"],
                    implementation={
                        "name": "ModelSurgeon",
                        "git_revision": "a" * 40,
                        "source_tree_clean": False,
                        "changed_source_path_count": 1,
                        "source_status_sha256": status_sha256,
                    },
                )
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="not produced from a verified clean source tree"):
        build_report(paths, bootstrap_samples=100)


def test_build_report_rejects_mixed_legacy_and_versioned_provenance(tmp_path: Path) -> None:
    magnitude = tmp_path / "magnitude.json"
    activation = tmp_path / "activation.json"
    magnitude.write_text(
        json.dumps(_run_report(["pass", "fail", "pass", "fail"])), encoding="utf-8"
    )
    activation.write_text(
        json.dumps(
            _run_report(
                ["pass", "fail", "pass", "fail"],
                implementation={
                    "name": "ModelSurgeon",
                    "git_revision": "a" * 40,
                    "source_tree_clean": True,
                    "changed_source_path_count": 0,
                    "source_status_sha256": hashlib.sha256(b"").hexdigest(),
                },
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mix reports with and without"):
        build_report({"magnitude": magnitude, "activation": activation}, bootstrap_samples=100)


def test_build_report_rejects_runs_with_different_source_samples(tmp_path: Path) -> None:
    magnitude = tmp_path / "magnitude.json"
    activation = tmp_path / "activation.json"
    magnitude.write_text(json.dumps(_run_report(["pass", "fail"])), encoding="utf-8")
    activation.write_text(
        json.dumps(_run_report(["pass", "fail"], source_hash="different-source")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different source-arm sample set"):
        build_report({"magnitude": magnitude, "activation": activation}, bootstrap_samples=100)


def test_build_report_supports_a_non_magnitude_reference(tmp_path: Path) -> None:
    magnitude = tmp_path / "magnitude.json"
    activation = tmp_path / "activation.json"
    magnitude.write_text(
        json.dumps(_run_report(["pass", "fail", "pass", "fail"])), encoding="utf-8"
    )
    activation.write_text(
        json.dumps(_run_report(["pass", "pass", "fail", "fail"])), encoding="utf-8"
    )

    report = build_report(
        {"magnitude": magnitude, "activation": activation},
        reference_name="activation",
        bootstrap_samples=100,
    )

    assert report["comparisons"][0]["right"] == "activation"
    assert report["comparisons"][2]["left"] == "activation"
    assert report["comparisons"][2]["removed_channel_selection"] == {
        "left_count": 2,
        "right_count": 2,
        "overlap_count": 2,
        "jaccard_similarity": 1.0,
    }


def test_build_report_can_compare_every_candidate_pair(tmp_path: Path) -> None:
    paths = {
        "magnitude": tmp_path / "magnitude.json",
        "activation": tmp_path / "activation.json",
        "energy": tmp_path / "energy.json",
    }
    outcomes = {
        "magnitude": ["pass", "fail", "pass", "fail"],
        "activation": ["pass", "pass", "fail", "fail"],
        "energy": ["fail", "pass", "pass", "fail"],
    }
    for name, path in paths.items():
        path.write_text(json.dumps(_run_report(outcomes[name])), encoding="utf-8")

    report = build_report(paths, all_pairs=True, bootstrap_samples=100)

    assert report["method"]["comparison_scope"] == "all candidate pairs"
    assert [item["left"] + "-" + item["right"] for item in report["comparisons"]] == [
        "source-magnitude",
        "source-activation",
        "source-energy",
        "magnitude-activation",
        "magnitude-energy",
        "activation-energy",
    ]


def test_build_report_supports_one_candidate_against_its_source(tmp_path: Path) -> None:
    path = tmp_path / "magnitude.json"
    path.write_text(
        json.dumps(_run_report(["pass", "fail", "fail", "pass"])), encoding="utf-8"
    )

    report = build_report(
        {"magnitude": path}, reference_name="magnitude", all_pairs=True, bootstrap_samples=100
    )

    assert [item["left"] + "-" + item["right"] for item in report["comparisons"]] == [
        "source-magnitude"
    ]
    assert report["comparisons"][0]["task_count"] == 4
