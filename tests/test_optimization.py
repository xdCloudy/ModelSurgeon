"""Contracts for the unified optimize planner and CLI boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import (
    CalibrationConfig,
    ConstraintConfig,
    ModelConfig,
    RuntimeConfig,
    Settings,
)
from modelsurgeon.optimization import (
    OptimizeOutcome,
    OptimizePlanError,
    build_optimize_plan,
    execute_first_party_optimize,
    write_optimize_plan,
)
from modelsurgeon.optimization_orchestrator import WorkflowStatus


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model": ModelConfig(path="models/tiny", revision="abc123"),
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_pinned_plan_is_supported_and_retains_auditable_contract() -> None:
    plan = build_optimize_plan(_settings())

    assert plan.outcome is OptimizeOutcome.SUPPORTED
    assert plan.plan_id.startswith("optimize_plan_")
    assert plan.config_digest
    assert plan.resume_token
    assert plan.executable
    record = plan.to_record()
    assert record["record_type"] == "optimize_plan"
    assert record["artifact_lineage"]["resolved_config_digest"] == plan.config_digest  # type: ignore[index]
    assert any(item.code == "artifact_write" and not item.required for item in plan.approvals)
    assert plan.source_overwrite is False


def test_equivalent_resolved_settings_have_the_same_plan_identity() -> None:
    first = build_optimize_plan(_settings())
    second = build_optimize_plan(
        Settings.model_validate(
            {
                "model": {"revision": "abc123", "path": "models/tiny"},
            }
        )
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.plan_id == second.plan_id


def test_missing_source_identity_is_unknown_but_still_plannable() -> None:
    plan = build_optimize_plan(Settings())

    assert plan.outcome is OptimizeOutcome.UNKNOWN
    assert not plan.executable
    assert len(plan.uncertainties) == 2
    assert plan.cost.download_bytes is None


def test_invalid_hardware_profile_has_actionable_alternatives() -> None:
    with pytest.raises(OptimizePlanError, match="cpu-small"):
        build_optimize_plan(_settings(), hardware_profile="tpu")


def test_infeasible_resource_limit_is_failed() -> None:
    plan = build_optimize_plan(
        _settings(constraints=ConstraintConfig(max_ram_bytes=1 * 1024**3))
    )

    assert plan.outcome is OptimizeOutcome.FAILED
    assert any("RAM" in message or "ram" in message for message in plan.uncertainties)


def test_gguf_plan_is_explicitly_unsupported() -> None:
    from modelsurgeon.adapters import ModelFamily, ModelFormat

    plan = build_optimize_plan(
        _settings(model=ModelConfig(path="models/model.gguf", revision="sha")),
    )
    assert plan.outcome is OptimizeOutcome.SUPPORTED

    plan = build_optimize_plan(
        Settings(
            model=ModelConfig(path="models/model.gguf", revision="sha", format=ModelFormat.GGUF)
        )
    )
    assert plan.outcome is OptimizeOutcome.UNSUPPORTED

    executable = RuntimeConfig(
        llama_cli="llama-cli",
        llama_perplexity="llama-perplexity",
        llama_bench="llama-bench",
        expected_revision="de8656bd9",
    )
    configured = Settings(
        model=ModelConfig(
            path="models/model.gguf",
            revision="sha",
            format=ModelFormat.GGUF,
            family=ModelFamily.LLAMA,
        ),
        calibration=CalibrationConfig(dataset="calibration.txt", dataset_revision="dataset-v1"),
        runtime=executable,
    )
    assert build_optimize_plan(configured).outcome is OptimizeOutcome.SUPPORTED


def test_plan_artifact_is_not_overwritten_by_default(tmp_path: Path) -> None:
    plan = build_optimize_plan(_settings())
    path = tmp_path / "plan.json"
    write_optimize_plan(path, plan)
    assert json.loads(path.read_text(encoding="utf-8"))["plan_id"] == plan.plan_id
    with pytest.raises(OptimizePlanError, match="overwrite"):
        write_optimize_plan(path, plan)


def test_python_execution_boundary_is_approval_gated_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-model"
    source.mkdir()
    state = tmp_path / "campaign.json"

    run = execute_first_party_optimize(
        Settings(model=ModelConfig(path=str(source), revision="sha256:source")),
        state,
    )

    assert run.status is WorkflowStatus.PAUSED
    assert run.outcome.value == OptimizeOutcome.UNKNOWN.value
    assert state.is_file()
    assert tuple(source.iterdir()) == ()


def test_cli_dry_run_emits_json_without_creating_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "--model",
            "models/tiny",
            "--revision",
            "abc123",
            "--json",
            "--output",
            str(output),
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["outcome"] == "supported"
    assert record["dry_run"] is True
    assert output.is_file()


def test_cli_no_llm_path_is_supported_without_provider_configuration() -> None:
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "--model",
            "models/tiny",
            "--revision",
            "abc123",
            "--no-llm",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["outcome"] == "supported"
    assert record["resolved_config"]["provider"]["kind"] == "none"
