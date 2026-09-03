from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import modelsurgeon.cli.app as app_module
from modelsurgeon.cli.report import generate_persisted_report
from test_reproduce_cli import _persisted_recipe


def test_report_resolves_a_run_and_writes_deterministic_offline_outputs(tmp_path: Path) -> None:
    run_id, metadata, _, _ = _persisted_recipe(tmp_path)
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"

    result = CliRunner().invoke(
        app_module.app,
        [
            "report",
            run_id,
            "--metadata",
            str(metadata),
            "--output",
            str(html_path),
            "--json-output",
            str(json_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["subject"] == {"id": run_id, "kind": "run"}
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert "ModelSurgeon run report" in html_path.read_text(encoding="utf-8")
    assert "https://" not in html_path.read_text(encoding="utf-8")

    repeated = generate_persisted_report(run_id, metadata_path=metadata)
    assert repeated.json_text == json_path.read_text(encoding="utf-8")


def test_report_explains_unknown_and_incomplete_ids_without_writing(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app_module.app,
        ["report", "search_unknown", "--metadata", str(tmp_path / "missing.sqlite3")],
    )

    assert result.exit_code == 2
    assert "existing regular SQLite" in result.stderr
