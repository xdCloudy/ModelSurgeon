"""Tests for the revision-pinned calibration CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import modelsurgeon.cli.calibration as calibration_module
from modelsurgeon.cli.app import app
from modelsurgeon.datasets.calibration import CalibrationSample
from modelsurgeon.datasets.huggingface import CalibrationManifest, TokenizedCalibrationSample


def _config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "dataset": "org/data",
                    "revision": "d" * 40,
                    "split": "train",
                    "license": "mit",
                    "trust": "trusted",
                    "trust_reason": "reviewed",
                    "metadata": {},
                },
                "preprocessing": {
                    "name": "plain-text",
                    "version": "1",
                    "configuration": {"normalization": "none"},
                },
                "tokenizer": {
                    "tokenizer": "org/tokenizer",
                    "revision": "t" * 40,
                    "configuration": {"trust_remote_code": False},
                },
                "selection": {"seed": 19, "sample_count": 2, "algorithm": "sha256-rank-v1"},
                "text_field": "text",
                "batch_size": 2,
                "max_tokens": 8,
            }
        ),
        encoding="utf-8",
    )


def _manifest(plan: calibration_module.CalibrationPlan) -> CalibrationManifest:
    contract = plan.request.contract
    identities = tuple(
        CalibrationSample(
            f"row-{index}", f"{index + 1:064x}"
        )
        for index in range(2)
    )
    return CalibrationManifest(
        contract.to_record(identities),
        tuple(
            TokenizedCalibrationSample(identity, (index + 1, index + 2))
            for index, identity in enumerate(identities)
        ),
    )


def test_calibrate_dry_run_does_not_create_cache_or_import_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "calibration.json"
    cache = tmp_path / "not-created" / "manifest.json"
    _config(config)
    monkeypatch.setattr(
        calibration_module,
        "stream_huggingface_calibration",
        lambda _: (_ for _ in ()).throw(AssertionError("dry-run accessed dataset")),
    )

    result = CliRunner().invoke(app, ["calibrate", str(config), "--cache", str(cache), "--dry-run"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["state"] == "dry_run"
    assert payload["dataset_revision"] == "d" * 40
    assert payload["tokenizer_revision"] == "t" * 40
    assert payload["cache_identity"] is None
    assert not cache.exists()
    assert not cache.parent.exists()


def test_calibrate_creates_and_reuses_a_validated_cache(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "calibration.json"
    cache = tmp_path / "cache" / "manifest.json"
    _config(config)
    calls = 0

    def stream(request) -> CalibrationManifest:
        nonlocal calls
        calls += 1
        return _manifest(calibration_module.CalibrationPlan(request))

    monkeypatch.setattr(calibration_module, "stream_huggingface_calibration", stream)
    runner = CliRunner()
    first = runner.invoke(app, ["calibrate", str(config), "--cache", str(cache)])
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.stdout)
    assert first_payload["state"] == "created"
    assert first_payload["sample_count"] == 2
    assert first_payload["token_count"] == 4
    first_bytes = cache.read_bytes()

    def unexpected(_: calibration_module.CalibrationPlan) -> CalibrationManifest:
        raise AssertionError("reused cache streamed the dataset")

    monkeypatch.setattr(calibration_module, "stream_huggingface_calibration", unexpected)
    second = runner.invoke(app, ["calibrate", str(config), "--cache", str(cache)])
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.stdout)
    assert second_payload["state"] == "reused"
    assert second_payload["cache_identity"] == first_payload["cache_identity"]
    assert cache.read_bytes() == first_bytes
    assert calls == 1


def test_calibrate_refresh_interrupt_preserves_completed_cache(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "calibration.json"
    cache = tmp_path / "cache" / "manifest.json"
    _config(config)
    monkeypatch.setattr(
        calibration_module,
        "stream_huggingface_calibration",
        lambda request: _manifest(calibration_module.CalibrationPlan(request)),
    )
    runner = CliRunner()
    created = runner.invoke(app, ["calibrate", str(config), "--cache", str(cache)])
    assert created.exit_code == 0, created.output
    before = cache.read_bytes()

    def interrupted(_: calibration_module.CalibrationPlan) -> CalibrationManifest:
        raise KeyboardInterrupt

    monkeypatch.setattr(calibration_module, "stream_huggingface_calibration", interrupted)
    result = runner.invoke(app, ["calibrate", str(config), "--cache", str(cache), "--refresh"])

    assert result.exit_code == 130
    assert "existing cache preserved" in result.output
    assert cache.read_bytes() == before
    assert not cache.with_name(f".{cache.name}.tmp").exists()


def test_calibrate_rejects_tampered_cache_without_streaming(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "calibration.json"
    cache = tmp_path / "cache" / "manifest.json"
    _config(config)
    monkeypatch.setattr(
        calibration_module,
        "stream_huggingface_calibration",
        lambda request: _manifest(calibration_module.CalibrationPlan(request)),
    )
    runner = CliRunner()
    assert runner.invoke(app, ["calibrate", str(config), "--cache", str(cache)]).exit_code == 0
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["token_count"] = 999
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        calibration_module,
        "stream_huggingface_calibration",
        lambda _: (_ for _ in ()).throw(AssertionError("tampered cache streamed")),
    )

    result = runner.invoke(app, ["calibrate", str(config), "--cache", str(cache)])

    assert result.exit_code == 2
    assert "cache token count is invalid" in result.output
