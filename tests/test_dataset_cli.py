from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import modelsurgeon.cli.dataset as dataset_module
from modelsurgeon.cli.app import app
from modelsurgeon.experiments.campaign import CampaignProgress
from test_dataset_builder import _join


class _Runtime:
    def __init__(self) -> None:
        self.resumes: list[bool] = []

    def run(self, *, resume: bool) -> dataset_module.DatasetCampaignResult:
        self.resumes.append(resume)
        return dataset_module.DatasetCampaignResult(
            CampaignProgress("run_fixture", 1, 0, 0, 0, 1, 0, 0),
            (_join(),),
        )


def test_generate_dataset_writes_split_files_and_reports_progress(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(dataset_module, "load_dataset_runtime", lambda _: runtime)
    output = tmp_path / "dataset"

    result = CliRunner().invoke(
        app,
        ["generate-dataset", "--runtime", "fixture:runtime", "--output", str(output), "--resume"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["campaign_progress"]["succeeded"] == 1
    assert payload["validation"]["valid"] is True
    assert runtime.resumes == [True]
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == payload
    assert any((output / f"{name}.jsonl").exists() for name in ("train", "validation", "test"))


def test_generate_dataset_never_overwrites_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_module, "load_dataset_runtime", lambda _: _Runtime())
    output = tmp_path / "dataset"
    output.mkdir()

    result = CliRunner().invoke(
        app,
        ["generate-dataset", "--runtime", "fixture:runtime", "--output", str(output)],
    )

    assert result.exit_code == 2
    assert "output already exists" in result.output
