import json
import subprocess
import sys

import pytest
import structlog

from modelsurgeon.logging import (
    LogFormat,
    bind_log_context,
    clear_log_context,
    configure_logging,
    log_context,
)


@pytest.fixture(autouse=True)
def clear_context_after_test() -> None:
    clear_log_context()


def test_import_does_not_configure_global_logging() -> None:
    script = (
        "import logging; "
        "before=list(logging.getLogger().handlers); "
        "import modelsurgeon.logging; "
        "after=list(logging.getLogger().handlers); "
        "raise SystemExit(0 if before == after else 1)"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_json_logging_contains_bound_run_model_and_component_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(output_format=LogFormat.JSON)
    bind_log_context(run_id="run-1", model_id="model-1", component_id="model.layers.2")

    structlog.get_logger("test").info("feature_collected", count=3)

    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "feature_collected"
    assert payload["run_id"] == "run-1"
    assert payload["model_id"] == "model-1"
    assert payload["component_id"] == "model.layers.2"
    assert payload["count"] == 3
    assert payload["level"] == "info"


def test_human_logging_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(output_format=LogFormat.HUMAN)

    structlog.get_logger("test").warning("oom_recovered", batch_size=2)

    output = capsys.readouterr().err
    assert "oom_recovered" in output
    assert "batch_size=2" in output


def test_context_manager_restores_previous_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(output_format=LogFormat.JSON)
    bind_log_context(run_id="outer")

    with log_context(run_id="inner", component_id="model.layers.0"):
        structlog.get_logger("test").info("inside")
    structlog.get_logger("test").info("outside")

    inside, outside = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert inside["run_id"] == "inner"
    assert inside["component_id"] == "model.layers.0"
    assert outside["run_id"] == "outer"
    assert "component_id" not in outside


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="verbose")

