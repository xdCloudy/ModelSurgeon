"""Tests for pinned llama-bench throughput and memory reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.evaluation import llama_cpp_throughput as throughput
from modelsurgeon.evaluation.llama_cpp_throughput import (
    LlamaCppThroughputConfig,
    LlamaCppThroughputError,
    _SampledCommandResult,
    benchmark_gguf_throughput,
    compare_gguf_throughput,
)


def _model(tmp_path: Path, name: str = "model.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"GGUF-model")
    return path


def _row(*, prompt: int, generation: int, rate: float) -> dict[str, object]:
    return {
        "build_commit": "95b8e33e1",
        "build_number": 10597,
        "cpu_info": "test cpu",
        "gpu_info": "",
        "backends": "CPU",
        "model_filename": "model.gguf",
        "model_type": "tiny F16",
        "model_size": 1024,
        "model_n_params": 100,
        "n_batch": 64,
        "n_ubatch": 32,
        "n_threads": 3,
        "n_gpu_layers": 0,
        "n_prompt": prompt,
        "n_gen": generation,
        "avg_ns": 1000,
        "stddev_ns": 100,
        "avg_ts": rate,
        "stddev_ts": 1.5,
        "samples_ns": [900, 1100],
        "samples_ts": [rate - 1, rate + 1],
    }


def _output(prompt_rate: float = 100.0, generation_rate: float = 20.0) -> str:
    return json.dumps(
        [
            _row(prompt=48, generation=0, rate=prompt_rate),
            _row(prompt=0, generation=12, rate=generation_rate),
        ]
    )


def _config(**changes: object) -> LlamaCppThroughputConfig:
    values: dict[str, object] = {
        "executable": "/opt/llama.cpp/llama-bench",
        "prompt_tokens": 48,
        "generation_tokens": 12,
        "batch_size": 64,
        "microbatch_size": 32,
        "threads": 3,
        "gpu_layers": 0,
        "repetitions": 2,
        "warmup": False,
        "timeout_seconds": 12,
        "sample_interval_seconds": 0.01,
        "max_log_bytes": 1024,
    }
    values.update(changes)
    return LlamaCppThroughputConfig(**values)


def _install_run(
    monkeypatch: pytest.MonkeyPatch,
    result: _SampledCommandResult,
) -> list[tuple[str, ...]]:
    executable = Path("/opt/llama.cpp/llama-bench").resolve()
    monkeypatch.setattr(throughput, "_resolve_executable", lambda value, revision: executable)
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **options: object) -> _SampledCommandResult:
        calls.append(command)
        assert options["timeout_seconds"] == 12
        assert options["sample_interval_seconds"] == 0.01
        assert options["max_log_bytes"] == 1024
        return result

    monkeypatch.setattr(throughput, "_run_sampled", fake_run)
    return calls


def test_benchmark_records_prompt_generation_latency_memory_and_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _SampledCommandResult(
        0, False, _output(), "cpu warning", False, False, 300_000_000, 0, 11
    )
    calls = _install_run(monkeypatch, result)

    report = benchmark_gguf_throughput(_model(tmp_path), config=_config())

    assert report.successful
    assert report.prompt is not None and report.generation is not None
    assert report.prompt.average_tokens_per_second == 100
    assert report.generation.average_tokens_per_second == 20
    assert report.prompt.average_latency_ns == 1000
    assert report.peak_rss_bytes == 300_000_000
    assert report.peak_vram_bytes == 0
    assert report.memory_sample_count == 11
    assert report.config.context_tokens == 48
    assert report.config.warmup is False
    assert report.config.threads == 3
    assert report.config.gpu_layers == 0
    assert calls[0][-1] == "--no-warmup"
    assert json.loads(json.dumps(report.to_record()))["successful"] is True


def test_nonzero_and_parse_failures_preserve_raw_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _SampledCommandResult(
        2, False, "partial", "load failed", True, False, 1234, 0, 2
    )
    _install_run(monkeypatch, raw)
    report = benchmark_gguf_throughput(_model(tmp_path), config=_config())
    assert not report.successful
    assert report.stdout == "partial"
    assert report.stderr == "load failed"
    assert report.stdout_truncated
    assert report.failure_reason == "llama-bench exited with status 2"

    invalid = _SampledCommandResult(
        0, False, "not-json", "diagnostic", False, False, 1234, 0, 2
    )
    _install_run(monkeypatch, invalid)
    parsed = benchmark_gguf_throughput(
        _model(tmp_path, "second.gguf"), config=_config()
    )
    assert not parsed.successful
    assert parsed.stdout == "not-json"
    assert parsed.stderr == "diagnostic"
    assert parsed.parse_error == "llama-bench stdout is not valid JSON"


def test_comparison_flags_configuration_drift_before_computing_ratios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_run(
        monkeypatch,
        _SampledCommandResult(0, False, _output(), "", False, False, 300, 0, 3),
    )
    baseline = benchmark_gguf_throughput(_model(tmp_path), config=_config())
    _install_run(
        monkeypatch,
        _SampledCommandResult(0, False, _output(), "", False, False, 250, 0, 3),
    )
    candidate = benchmark_gguf_throughput(
        _model(tmp_path, "candidate.gguf"), config=_config()
    )

    comparison = compare_gguf_throughput(baseline, candidate)
    assert comparison.comparable
    assert comparison.prompt_speedup_ratio == 1
    assert comparison.generation_speedup_ratio == 1
    assert comparison.peak_rss_delta_bytes == -50

    drifted = replace(candidate, config=replace(candidate.config, threads=4))
    rejected = compare_gguf_throughput(baseline, drifted)
    assert not rejected.comparable
    assert rejected.drift_fields == ("config.threads",)
    assert rejected.prompt_speedup_ratio is None


def test_pinned_revision_and_echoed_runtime_configuration_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = json.loads(_output())
    value[0]["build_commit"] = "deadbee"
    value[1]["build_commit"] = "deadbee"
    _install_run(
        monkeypatch,
        _SampledCommandResult(0, False, json.dumps(value), "", False, False, 1, 0, 1),
    )
    wrong_revision = benchmark_gguf_throughput(_model(tmp_path), config=_config())
    assert not wrong_revision.successful
    assert wrong_revision.parse_error is not None
    assert "unsupported llama.cpp revision" in wrong_revision.parse_error

    value = json.loads(_output())
    value[1]["n_threads"] = 4
    _install_run(
        monkeypatch,
        _SampledCommandResult(0, False, json.dumps(value), "", False, False, 1, 0, 1),
    )
    drift = benchmark_gguf_throughput(
        _model(tmp_path, "drift.gguf"), config=_config()
    )
    assert drift.parse_error is not None
    assert "drifted" in drift.parse_error


def test_gpu_offload_requires_vram_probe(tmp_path: Path) -> None:
    with pytest.raises(LlamaCppThroughputError, match="VRAM probe"):
        benchmark_gguf_throughput(
            _model(tmp_path),
            config=_config(gpu_layers=1),
        )


def test_invalid_configuration_fails_early() -> None:
    with pytest.raises(LlamaCppThroughputError, match="microbatch"):
        _config(batch_size=16, microbatch_size=32)
    with pytest.raises(LlamaCppThroughputError, match="positive"):
        _config(prompt_tokens=0)
