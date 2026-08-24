"""Tests for paired pinned llama.cpp GGUF perplexity benchmarking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from modelsurgeon.evaluation import llama_cpp_perplexity as perplexity
from modelsurgeon.evaluation.llama_cpp import LlamaCppToolProvenance, _BoundedCommandResult
from modelsurgeon.evaluation.llama_cpp_perplexity import (
    LlamaCppPerplexityConfig,
    LlamaCppPerplexityError,
    LlamaCppPerplexityManifest,
    benchmark_gguf_perplexity,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest(tmp_path: Path) -> LlamaCppPerplexityManifest:
    text = tmp_path / "samples.txt"
    payload = b"A deterministic calibration sample.\n"
    text.write_bytes(payload)
    return LlamaCppPerplexityManifest(
        text,
        _sha256(payload),
        1,
        "dataset/name",
        "dataset-revision",
        "test",
        "tokenizer/name",
        "tokenizer-revision",
    )


def _models(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.gguf"
    candidate = tmp_path / "candidate.gguf"
    baseline.write_bytes(b"GGUF-baseline")
    candidate.write_bytes(b"GGUF-candidate")
    return baseline, candidate


def _install_tool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: list[_BoundedCommandResult] | None = None,
) -> list[tuple[str, ...]]:
    monkeypatch.setattr(perplexity, "_tokenizer_digest", lambda path: "a" * 64)
    executable = Path("/opt/llama.cpp/llama-perplexity").resolve()
    tool = LlamaCppToolProvenance(
        executable,
        "95b8e33e16bb9a60de780a70930ebf729db6a90a",
        "95b8e33e1",
        (str(executable), "--version"),
        "version: 0.2.0-dev (build 10597, commit 95b8e33e1)",
    )
    monkeypatch.setattr(perplexity, "_probe_tool", lambda config: tool)
    calls: list[tuple[str, ...]] = []
    queued = results or [
        _BoundedCommandResult(0, False, "", "Final estimate: PPL = 2.5 +/- 0.1", False, False),
        _BoundedCommandResult(0, False, "", "Final estimate: PPL = 2.75 +/- 0.2", False, False),
    ]

    def fake_run(
        command: tuple[str, ...], *, timeout_seconds: float, max_log_bytes: int
    ) -> _BoundedCommandResult:
        assert timeout_seconds == 12
        assert max_log_bytes == 1024
        calls.append(command)
        return queued.pop(0)

    monkeypatch.setattr(perplexity, "_run_bounded", fake_run)
    return calls


def test_pair_uses_one_manifest_tokenizer_and_context_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_tool(monkeypatch)
    baseline, candidate = _models(tmp_path)
    manifest = _manifest(tmp_path)
    config = LlamaCppPerplexityConfig(
        context_size=128,
        batch_size=64,
        microbatch_size=32,
        threads=3,
        chunks=2,
        timeout_seconds=12,
        max_log_bytes=1024,
    )

    report = benchmark_gguf_perplexity(baseline, candidate, manifest, config=config)

    assert report.successful
    assert report.baseline.perplexity == 2.5
    assert report.candidate.perplexity == 2.75
    assert report.perplexity_delta == pytest.approx(0.25)
    assert report.baseline.tokenizer_sha256 == report.candidate.tokenizer_sha256
    assert len(calls) == 2
    baseline_command, candidate_command = calls
    assert baseline_command[0] == candidate_command[0]
    assert baseline_command[1:3] != candidate_command[1:3]
    assert baseline_command[3:] == candidate_command[3:]
    assert baseline_command[3:7] == (
        "-f",
        str(manifest.text_path.resolve()),
        "-c",
        "128",
    )
    assert "--no-warmup" in baseline_command
    record = json.loads(json.dumps(report.to_record()))
    assert record["manifest_id"] == manifest.manifest_id
    assert record["perplexity_delta"] == pytest.approx(0.25)


def test_runtime_and_parse_failures_preserve_raw_bounded_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_failure = _BoundedCommandResult(
        2,
        False,
        "partial baseline",
        "failed to load tensor",
        True,
        False,
    )
    raw_parse = _BoundedCommandResult(
        0,
        False,
        "candidate output",
        "no final estimate",
        False,
        False,
    )
    _install_tool(monkeypatch, results=[raw_failure, raw_parse])
    baseline, candidate = _models(tmp_path)

    report = benchmark_gguf_perplexity(
        baseline,
        candidate,
        _manifest(tmp_path),
        config=LlamaCppPerplexityConfig(timeout_seconds=12, max_log_bytes=1024),
    )

    assert not report.successful
    assert report.perplexity_delta is None
    assert report.baseline.failure_reason == "llama-perplexity exited with status 2"
    assert report.baseline.stdout == "partial baseline"
    assert report.baseline.stderr == "failed to load tensor"
    assert report.baseline.stdout_truncated
    assert report.candidate.failure_reason is not None
    assert report.candidate.stdout == "candidate output"
    assert report.candidate.stderr == "no final estimate"


def test_tokenizer_mismatch_fails_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline, candidate = _models(tmp_path)
    tokenizer_digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(perplexity, "_tokenizer_digest", lambda path: next(tokenizer_digests))
    monkeypatch.setattr(
        perplexity,
        "_probe_tool",
        lambda config: pytest.fail("tool must not run after tokenizer mismatch"),
    )

    with pytest.raises(LlamaCppPerplexityError, match="tokenizer metadata do not match"):
        benchmark_gguf_perplexity(baseline, candidate, _manifest(tmp_path))


def test_changed_or_non_utf8_manifest_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline, candidate = _models(tmp_path)
    manifest = _manifest(tmp_path)
    manifest.text_path.write_text("changed", encoding="utf-8")
    with pytest.raises(LlamaCppPerplexityError, match="manifest digest"):
        benchmark_gguf_perplexity(baseline, candidate, manifest)

    payload = b"\xff\xfe"
    manifest.text_path.write_bytes(payload)
    invalid_utf8 = LlamaCppPerplexityManifest(
        manifest.text_path,
        _sha256(payload),
        1,
        "dataset/name",
        "revision",
        "test",
        "tokenizer/name",
        "revision",
    )
    with pytest.raises(LlamaCppPerplexityError, match="valid UTF-8"):
        benchmark_gguf_perplexity(baseline, candidate, invalid_utf8)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(LlamaCppPerplexityError, match="microbatch <= batch <= context"):
        LlamaCppPerplexityConfig(context_size=64, batch_size=128)
    with pytest.raises(LlamaCppPerplexityError, match="lowercase SHA-256"):
        LlamaCppPerplexityManifest(
            Path("samples.txt"),
            "bad",
            1,
            "dataset",
            "revision",
            "test",
            "tokenizer",
            "revision",
        )
