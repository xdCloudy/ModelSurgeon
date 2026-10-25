"""Regression tests for native GGUF task-quality measurement and gating."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modelsurgeon.native_gguf_optimize_runtime import (
    NativeGGUFOptimizeRuntime,
    NativeGGUFOptimizeRuntimeError,
)


def _runtime(dataset: Path, *, enabled: bool = True) -> NativeGGUFOptimizeRuntime:
    runtime = object.__new__(NativeGGUFOptimizeRuntime)
    runtime.plan = SimpleNamespace(
        resolved_config={
            "constraints": {"min_quality_retention_ratio": 0.99},
            "task_quality": {
                "method": "code_exact_match" if enabled else "none",
                "dataset": str(dataset),
                "dataset_revision": "dataset-r1",
                "split": "test",
                "max_samples": 1,
                "max_new_tokens": 8,
            },
            "runtime": {
                "llama_cli": "llama-cli",
                "llama_perplexity": "llama-perplexity",
                "llama_bench": "llama-bench",
                "expected_revision": "llama-r1",
                "context_size": 1024,
            },
        },
        quality_profile=SimpleNamespace(max_perplexity_delta=10.0),
    )
    runtime.task_quality_dataset = None
    return runtime


def _dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"id": "case-1", "prompt": "Return one.", "reference": "return 1"}
        )
        + "\n",
        encoding="utf-8",
    )


def _task_record(digest: str, score: float) -> dict[str, Any]:
    return {
        "dataset": {"digest": digest},
        "exact_match_accuracy": score,
    }


def test_native_task_quality_uses_pinned_cli_and_records_content_addressed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "coding.jsonl"
    _dataset(dataset)
    runtime = _runtime(dataset)
    validation = SimpleNamespace(timeout_seconds=17, threads=3, gpu_layers=0)
    monkeypatch.setattr(runtime, "_tool_config", lambda: (validation, None, None))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="return 1\n", stderr="")

    monkeypatch.setattr("modelsurgeon.native_gguf_optimize_runtime.subprocess.run", run)

    result = runtime._task_quality(tmp_path / "candidate.gguf")

    assert result is not None
    assert result["exact_match_accuracy"] == 1.0
    assert result["total"] == 1
    assert result["dataset"]["revision"] == "dataset-r1"
    assert result["dataset"]["split"] == "test"
    assert result["dataset"]["digest"].startswith("sha256:")
    assert len(result["output_digests"]) == 1
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ["llama-cli", "-m", str(tmp_path / "candidate.gguf")]
    assert command[command.index("-p") + 1] == "Return one."
    assert command[command.index("-n") + 1] == "8"
    assert command[command.index("--temp") + 1] == "0"
    assert command[command.index("-t") + 1] == "3"
    assert "-ngl" not in command
    assert kwargs["timeout"] == 17
    assert kwargs["capture_output"] is True


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("timeout", "generation timed out"),
        ("nonzero", "failed with status 3"),
        ("empty", "returned no text"),
    ),
)
def test_native_task_quality_generation_failures_abort_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    dataset = tmp_path / "coding.jsonl"
    _dataset(dataset)
    runtime = _runtime(dataset)
    validation = SimpleNamespace(timeout_seconds=17, threads=3, gpu_layers=0)
    monkeypatch.setattr(runtime, "_tool_config", lambda: (validation, None, None))

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            return SimpleNamespace(returncode=3, stdout="", stderr="runtime error")
        return SimpleNamespace(returncode=0, stdout=" \n", stderr="")

    monkeypatch.setattr("modelsurgeon.native_gguf_optimize_runtime.subprocess.run", run)

    with pytest.raises(NativeGGUFOptimizeRuntimeError, match=message):
        runtime._task_quality(tmp_path / "candidate.gguf")


def test_native_task_quality_gate_combines_perplexity_and_exact_match() -> None:
    runtime = _runtime(Path("unused.jsonl"))
    digest = "sha256:dataset"

    accepted = runtime._quality_gate(
        10.0,
        10.0,
        _task_record(digest, 1.0),
        _task_record(digest, 1.0),
    )
    rejected = runtime._quality_gate(
        10.0,
        10.0,
        _task_record(digest, 1.0),
        _task_record(digest, 0.98),
    )

    assert accepted["accepted"] is True
    assert accepted["task_quality_gate"]["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["task_quality_gate"]["accepted"] is False


def test_native_task_quality_gate_rejects_missing_or_mismatched_evidence() -> None:
    runtime = _runtime(Path("unused.jsonl"))

    with pytest.raises(NativeGGUFOptimizeRuntimeError, match="incomplete evidence"):
        runtime._quality_gate(10.0, 10.0, _task_record("sha256:data", 1.0), None)

    with pytest.raises(NativeGGUFOptimizeRuntimeError, match="different digests"):
        runtime._quality_gate(
            10.0,
            10.0,
            _task_record("sha256:baseline", 1.0),
            _task_record("sha256:candidate", 1.0),
        )
