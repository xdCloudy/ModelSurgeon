"""Invariant coverage for the first-party acceptance campaign tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from tools import run_first_party_hf_acceptance as acceptance_tool
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


def test_single_model_campaign_records_explicit_hardware_calibration_and_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("small campaign fixture", encoding="utf-8")
    output = tmp_path / "campaign.json"
    artifact_root = tmp_path / "artifacts"
    calls: list[dict[str, object]] = []

    def run_cell(identifier: str, revision: str, **kwargs: object) -> dict[str, object]:
        calls.append({"model": identifier, "revision": revision, **kwargs})
        return {"model": identifier, "revision": revision, "status": "completed"}

    monkeypatch.setattr(acceptance_tool, "_run_cell", run_cell)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_first_party_hf_acceptance.py",
            "--model",
            "org/model@deadbeef",
            "--calibration-text",
            str(calibration),
            "--output",
            str(output),
            "--artifact-root",
            str(artifact_root),
            "--hardware-profile",
            "gpu-12gb",
            "--calibration-samples",
            "16",
            "--max-sequence-length",
            "128",
            "--min-quality-retention-ratio",
            "0.9",
            "--repair-method",
            "distillation",
            "--repair-max-steps",
            "2",
            "--repair-target-module",
            "model.layers.0.mlp.down_proj",
        ],
    )

    assert acceptance_tool.main() == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    contract = record["command_contract"]
    assert record["tool_revision"] == "first-party-hf-acceptance-v4"
    assert contract["hardware_profile"] == "gpu-12gb"
    assert contract["calibration_samples"] == 16
    assert contract["max_sequence_length"] == 128
    assert contract["min_quality_retention_ratio"] == 0.9
    assert contract["repair_method"] == "distillation"
    assert contract["repair_max_steps"] == 2
    assert contract["repair_target_modules"] == ["model.layers.0.mlp.down_proj"]
    assert calls[0]["hardware_profile"] == "gpu-12gb"
    assert calls[0]["repair_target_modules"] == ("model.layers.0.mlp.down_proj",)
