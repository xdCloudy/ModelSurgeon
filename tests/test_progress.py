"""Crash-consistent progress and privacy-safe diagnostics tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.progress import (
    ProgressOutcome,
    ProgressStage,
    ProgressStatus,
    ProgressStore,
    redact_diagnostics,
)


def test_restart_resumes_completed_stages_exactly_once(tmp_path: Path) -> None:
    root = tmp_path / "progress"
    store = ProgressStore(root)
    started = store.start(
        "campaign-1",
        (ProgressStage.DOWNLOAD, ProgressStage.SEARCH, ProgressStage.PUBLICATION),
        source_artifact_digest="sha256:" + "1" * 64,
        total=10,
    )
    assert started.status is ProgressStatus.RUNNING
    first = store.record(
        "campaign-1",
        event_type="stage_completed",
        stage=ProgressStage.DOWNLOAD,
        status=ProgressStatus.COMPLETED,
        completed=3,
        total=10,
        elapsed_seconds=6,
        next_work=(ProgressStage.SEARCH.value,),
    )
    assert first.status is ProgressStatus.RUNNING
    assert first.completed_stages == (ProgressStage.DOWNLOAD,)
    assert first.eta_seconds == 14

    restarted = ProgressStore(root)
    resumed = restarted.resume("campaign-1")
    duplicate = restarted.record(
        "campaign-1",
        event_type="stage_completed",
        stage=ProgressStage.DOWNLOAD,
        status=ProgressStatus.COMPLETED,
        completed=3,
        total=10,
        elapsed_seconds=6,
        next_work=(ProgressStage.SEARCH.value,),
    )
    assert resumed.next_work == (ProgressStage.SEARCH.value,)
    assert duplicate.event_sequence == resumed.event_sequence
    assert len(restarted.events("campaign-1")) == 3


def test_cancel_pause_and_resume_preserve_last_accepted_artifact(tmp_path: Path) -> None:
    store = ProgressStore(tmp_path / "progress")
    store.start("campaign", (ProgressStage.SURGERY,), source_artifact_digest="sha256:" + "2" * 64)
    store.record(
        "campaign",
        event_type="stage_completed",
        stage=ProgressStage.SURGERY,
        status=ProgressStatus.RUNNING,
        completed=1,
        total=2,
        elapsed_seconds=3,
        accepted_artifact_digest="sha256:" + "3" * 64,
    )
    paused = store.pause("campaign")
    cancelled = store.cancel("campaign")
    resumed = store.resume("campaign")

    assert paused.status is ProgressStatus.PAUSED
    assert cancelled.status is ProgressStatus.CANCELLED
    assert resumed.status is ProgressStatus.RUNNING
    assert resumed.accepted_artifact_digest == "sha256:" + "3" * 64


def test_diagnostics_redact_secrets_and_local_paths(tmp_path: Path) -> None:
    redacted, paths = redact_diagnostics(
        {
            "api_token": "secret",
            "checkpoint": "C:/Users/Cloudy/model.safetensors",
            "nested": {"password": "pw"},
        }
    )
    assert redacted == {
        "api_token": "<redacted-secret>",
        "checkpoint": "<redacted-path>",
        "nested": {"password": "<redacted-secret>"},
    }
    assert paths == ("root.api_token", "root.checkpoint", "root.nested.password")

    store = ProgressStore(tmp_path / "progress")
    store.start("campaign", (ProgressStage.DOWNLOAD,))
    bundle = store.diagnostics(
        "campaign", environment={"secret": "value", "path": "/private/model"}
    )
    record = bundle.to_record()
    assert record["environment"] == {
        "path": "<redacted-path>",
        "secret": "<redacted-secret>",
    }
    assert "value" not in json.dumps(record)


def test_completed_campaign_is_terminal_and_unknown_outcome_is_explicit(tmp_path: Path) -> None:
    store = ProgressStore(tmp_path / "progress")
    store.start("campaign", (ProgressStage.PUBLICATION,), total=1)
    completed = store.record(
        "campaign",
        event_type="stage_completed",
        stage=ProgressStage.PUBLICATION,
        status=ProgressStatus.COMPLETED,
        completed=1,
        total=1,
        elapsed_seconds=1,
        outcome=ProgressOutcome.UNSUPPORTED,
    )
    assert completed.status is ProgressStatus.COMPLETED
    assert completed.outcome is ProgressOutcome.UNSUPPORTED
    assert store.resume("campaign") == completed


def test_progress_cli_json_is_noninteractive(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "progress",
            "init",
            "campaign",
            "--stages",
            "download,search",
            "--total",
            "2",
            "--root",
            str(tmp_path / "progress"),
            "--json",
        ],
        color=False,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["campaign_id"] == "campaign"
