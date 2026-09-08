"""Invariant coverage for the first-party acceptance campaign tool."""

from __future__ import annotations

import argparse

import pytest
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
