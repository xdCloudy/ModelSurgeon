from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.cli.benchmark import (
    DeterministicFakeBenchmarkExecutor,
    build_benchmark_plan,
    initialize_state,
    run_benchmark_matrix,
)


def test_plan_is_content_addressed_and_has_explicit_skips() -> None:
    first = build_benchmark_plan()
    second = build_benchmark_plan()

    assert first.to_record() == second.to_record()
    assert first.plan_id == second.plan_id
    assert len(first.cells) == 98
    assert sum(cell.status == "pending" for cell in first.cells) == 14
    assert sum(cell.status == "unsupported" for cell in first.cells) == 84


def test_resume_does_not_rerun_completed_cells() -> None:
    plan = build_benchmark_plan()
    first = run_benchmark_matrix(plan, initialize_state(plan), DeterministicFakeBenchmarkExecutor())
    second = run_benchmark_matrix(plan, first, DeterministicFakeBenchmarkExecutor())

    assert second.to_record() == first.to_record()
    assert {item.status for item in second.results} == {"success", "unsupported"}


def test_cli_plan_run_and_audit_have_stable_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    state_path = tmp_path / "state.json"
    runner = CliRunner()

    planned = runner.invoke(app, ["benchmark", "plan", "--output", str(plan_path)])
    assert planned.exit_code == 0, planned.output
    assert json.loads(planned.stdout)["plan_id"] == json.loads(plan_path.read_text())["plan_id"]

    ran = runner.invoke(
        app,
        ["benchmark", "run", "--plan", str(plan_path), "--state", str(state_path)],
    )
    assert ran.exit_code == 0, ran.output

    audited = runner.invoke(
        app,
        ["benchmark", "audit", "--plan", str(plan_path), "--state", str(state_path)],
    )
    assert audited.exit_code == 0, audited.output
    payload = json.loads(audited.stdout)
    assert payload["valid"] is True
    assert payload["failed_cells"] == []


def test_import_refuses_to_replace_an_immutable_cell(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    state_path = tmp_path / "state.json"
    result_path = tmp_path / "result.json"
    runner = CliRunner()
    assert runner.invoke(app, ["benchmark", "plan", "--output", str(plan_path)]).exit_code == 0
    assert runner.invoke(
        app, ["benchmark", "run", "--plan", str(plan_path), "--state", str(state_path)]
    ).exit_code == 0
    first_cell = json.loads(plan_path.read_text())["cells"][0]["cell_id"]
    result_path.write_text(
        json.dumps({"cell_id": first_cell, "status": "success", "metrics": {}}),
        encoding="utf-8",
    )
    imported = runner.invoke(
        app,
        [
            "benchmark",
            "import",
            "--plan",
            str(plan_path),
            "--state",
            str(state_path),
            "--result",
            str(result_path),
        ],
    )
    assert imported.exit_code == 1
    assert "immutable" in imported.stderr
