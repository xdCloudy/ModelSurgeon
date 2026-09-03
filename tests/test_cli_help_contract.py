"""Smoke-test the public CLI help and non-interactive terminal contract."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app


_PUBLIC_COMMANDS = (
    "inspect",
    "experiment",
    "first-surgeon-proof",
    "first-surgeon-hf-proof",
    "first-surgeon-evidence",
    "train-surgeon",
    "predict-surgeon",
    "search",
    "features",
    "generate-dataset",
    "reproduce",
    "report",
)


def test_root_help_exposes_completion_without_terminal_control_sequences() -> None:
    result = CliRunner().invoke(app, ["--help"], color=False)

    assert result.exit_code == 0
    assert "--show-completion" in result.output
    assert "\x1b" not in result.output


@pytest.mark.parametrize("command", _PUBLIC_COMMANDS)
def test_every_public_command_has_plain_noninteractive_help(command: str) -> None:
    result = CliRunner().invoke(app, [command, "--help"], color=False)

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "\x1b" not in result.output
