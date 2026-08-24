"""Tests for pinned llama.cpp GGUF load and bounded generation validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest

from modelsurgeon.evaluation import llama_cpp
from modelsurgeon.evaluation.llama_cpp import (
    LLAMA_CPP_VALIDATION_COMMIT,
    LlamaCppValidationConfig,
    LlamaCppValidationError,
    validate_generated_gguf,
)


def _install_fake_llama(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_returncode: int = 0,
    model_stdout: bytes = b"ok",
    model_stderr: bytes = b"",
    revision: str = LLAMA_CPP_VALIDATION_COMMIT[:7],
    model_timeout: bool = False,
) -> list[tuple[str, ...]]:
    executable = "/opt/llama.cpp/llama-cli"
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: executable)

    def fake_run(command: object, **options: object) -> object:
        argv = tuple(cast(tuple[str, ...], command))
        calls.append(argv)
        stdout = cast(BinaryIO, options["stdout"])
        stderr = cast(BinaryIO, options["stderr"])
        assert options["check"] is False
        assert options["shell"] is False
        if "--version" in argv:
            stdout.write(f"version: 1 (`{revision}`)\nbuilt with test\n".encode())
            return SimpleNamespace(returncode=0)
        stdout.write(model_stdout)
        stderr.write(model_stderr)
        if model_timeout:
            raise subprocess.TimeoutExpired(argv, timeout=cast(float, options["timeout"]))
        return SimpleNamespace(returncode=model_returncode)

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    return calls


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.gguf"
    path.write_bytes(b"GGUF")
    return path


def test_generated_gguf_passes_only_after_pinned_llama_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_fake_llama(monkeypatch)
    model = _model(tmp_path)

    report = validate_generated_gguf(model)

    assert report.successful is True
    assert report.failure_reason is None
    assert report.tool.expected_revision == LLAMA_CPP_VALIDATION_COMMIT
    assert report.tool.reported_revision == LLAMA_CPP_VALIDATION_COMMIT[:7]
    assert calls[0] == (str(Path("/opt/llama.cpp/llama-cli").resolve()), "--version")
    assert report.command == calls[1]
    assert report.command[1:5] == (
        "-m",
        str(model.resolve()),
        "-p",
        "ModelSurgeon external GGUF validation.",
    )
    assert report.command[5:9] == ("-n", "1", "-c", "128")
    assert report.command[9:13] == ("-t", "1", "-ngl", "0")
    assert "--seed" in report.command
    assert "--temp" in report.command
    assert report.command[-2:] == ("--single-turn", "--simple-io")
    assert json.loads(json.dumps(report.to_record()))["successful"] is True


def test_current_official_version_format_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = "/opt/llama.cpp/llama-cli"
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: executable)

    def fake_run(command: object, **options: object) -> object:
        argv = tuple(cast(tuple[str, ...], command))
        stdout = cast(BinaryIO, options["stdout"])
        if "--version" in argv:
            stdout.write(
                b"version: 0.2.0-dev (build 10597, commit 95b8e33e1)\n"
                b"built with Clang 22.1.8 for Windows AMD64\n"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    report = validate_generated_gguf(_model(tmp_path))

    assert report.successful
    assert report.tool.reported_revision == "95b8e33e1"


def test_mismatched_llama_revision_fails_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_fake_llama(monkeypatch, revision="deadbee")

    with pytest.raises(LlamaCppValidationError, match="pinned to"):
        validate_generated_gguf(_model(tmp_path))

    assert len(calls) == 1
    assert calls[0][-1] == "--version"


def test_nonzero_llama_exit_returns_unsuccessful_report_with_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama(
        monkeypatch,
        model_returncode=2,
        model_stdout=b"partial output",
        model_stderr=b"failed to load tensor",
    )

    report = validate_generated_gguf(_model(tmp_path))

    assert report.successful is False
    assert report.returncode == 2
    assert report.timed_out is False
    assert report.failure_reason == "llama-cli exited with status 2"
    assert report.stdout == "partial output"
    assert report.stderr == "failed to load tensor"


def test_timeout_never_reports_generated_gguf_as_successful(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama(monkeypatch, model_stdout=b"started", model_timeout=True)

    report = validate_generated_gguf(
        _model(tmp_path),
        config=LlamaCppValidationConfig(timeout_seconds=0.25),
    )

    assert report.successful is False
    assert report.returncode is None
    assert report.timed_out is True
    assert report.failure_reason == "llama-cli validation timed out"
    assert report.stdout == "started"


def test_logs_are_capped_in_memory_and_marked_truncated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama(
        monkeypatch,
        model_stdout=b"x" * 64,
        model_stderr=b"y" * 64,
    )

    report = validate_generated_gguf(
        _model(tmp_path),
        config=LlamaCppValidationConfig(max_log_bytes=8),
    )

    assert report.successful is True
    assert report.stdout == "x" * 8
    assert report.stderr == "y" * 8
    assert report.stdout_truncated is True
    assert report.stderr_truncated is True


def test_missing_model_and_missing_tool_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.gguf"
    with pytest.raises(LlamaCppValidationError, match="does not exist"):
        validate_generated_gguf(missing)

    model = _model(tmp_path)
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: None)
    with pytest.raises(LlamaCppValidationError, match="not found on PATH"):
        validate_generated_gguf(model)
