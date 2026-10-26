"""Invariant coverage for the first-party acceptance campaign tool."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from tools import run_first_party_hf_acceptance as acceptance
from tools.run_first_party_hf_acceptance import _model_reference, _stage_summary


def test_acceptance_model_reference_requires_immutable_revision() -> None:
    assert _model_reference("org/model@deadbeef") == ("org/model", "deadbeef")
    with pytest.raises(argparse.ArgumentTypeError):
        _model_reference("org/model")


def test_acceptance_stage_summary_retains_measured_artifact_evidence() -> None:
    summary = _stage_summary(
        {
            "stages": [
                {
                    "stage": "surgery",
                    "result": {
                        "outcome": "supported",
                        "measured": True,
                        "constraints_passed": True,
                        "artifact_digest": "sha256:" + "a" * 64,
                        "detail": '{"stages":[{"reloadable":true}],"failed_index":null}',
                    },
                }
            ]
        }
    )
    assert summary == [
        {
            "stage": "surgery",
            "outcome": "supported",
            "measured": True,
            "constraints_passed": True,
            "artifact_digest": "sha256:" + "a" * 64,
            "stages": [{"reloadable": True}],
            "failed_index": None,
        }
    ]


def test_acceptance_cell_passes_declared_hardware_profile_to_planner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_build_optimize_plan(_settings: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            executable=False,
            outcome=SimpleNamespace(value="unsupported"),
            uncertainties=("test uncertainty",),
            limitations=("test limitation",),
            to_record=lambda: {"hardware_profile": kwargs["hardware_profile"]},
        )

    monkeypatch.setattr(acceptance, "_resolve_model", lambda _identifier, _revision: tmp_path)
    monkeypatch.setattr(acceptance, "build_optimize_plan", fake_build_optimize_plan)

    cell = acceptance._run_cell(
        "org/model",
        "deadbeef",
        calibration_text=tmp_path / "calibration.txt",
        artifact_root=tmp_path / "artifacts",
        preset="balanced",
        hardware_profile="gpu-12gb",
        quality_profile="quality",
        evaluations=None,
        search_scope="mlp_channel",
        low_rank_rank=4,
        surgeon_registry=None,
        surgeon_card=None,
        surgeon_signing_env=None,
        task_quality_dataset=None,
        task_quality_dataset_revision=None,
        task_quality_split="test",
        task_quality_max_new_tokens=32,
        task_quality_max_samples=None,
    )

    assert captured["hardware_profile"] == "gpu-12gb"
    assert cell["plan"] == {"hardware_profile": "gpu-12gb"}
    assert cell["status"] == "not_executed"
