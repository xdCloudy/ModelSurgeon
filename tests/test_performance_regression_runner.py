"""Tests for the bounded performance fixture runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tools.run_performance_regressions import (
    PerformanceRunnerError,
    _case_operation,
    _cpu_hash,
    _gguf_stream,
    _io_delta,
    run_profile,
)

from modelsurgeon.adapters.gguf import GGUFWriteError
from modelsurgeon.evaluation.performance_regression import (
    PerformanceBudgetManifest,
    RegressionLane,
    load_performance_budget_manifest,
)
from modelsurgeon.experiments.runtime_telemetry import ProcessIOCounters


def _profile(name: str) -> PerformanceBudgetManifest:
    payload = Path("tests/fixtures/performance_budgets_v1.json").read_text(
        encoding="utf-8"
    )
    return load_performance_budget_manifest(payload, name)


def test_io_delta_requires_monotonic_complete_counters() -> None:
    before = ProcessIOCounters(10, 20)
    after = ProcessIOCounters(15, 32)

    assert _io_delta(before, after) == (5, 12)
    assert _io_delta(None, after) == (None, None)
    assert _io_delta(before, None) == (None, None)
    assert _io_delta(after, before) == (None, None)


def test_cpu_hash_fixture_is_bounded_and_repeatable(tmp_path: Path) -> None:
    operation = _cpu_hash(1024 * 1024)

    assert operation(tmp_path) == 1024 * 1024
    assert operation(tmp_path) == 1024 * 1024
    assert not tuple(tmp_path.iterdir())


def test_gguf_fixture_streams_a_real_non_overwriting_artifact(tmp_path: Path) -> None:
    operation = _gguf_stream(1024 * 1024)

    size = operation(tmp_path)

    output = tmp_path / "streaming-fixture.gguf"
    assert output.stat().st_size == size
    assert size > 1024 * 1024
    with pytest.raises(GGUFWriteError, match="already exists"):
        operation(tmp_path)


def test_fixture_cases_are_profile_bound() -> None:
    pr = _profile("pr-cpu")

    assert callable(_case_operation(pr, "cpu_canonical_hash"))
    assert callable(_case_operation(pr, "gguf_stream_write"))
    with pytest.raises(PerformanceRunnerError, match="does not define case"):
        _case_operation(pr, "cuda_matmul")


def test_run_profile_rejects_lane_fixture_mismatch(tmp_path: Path) -> None:
    manifest = replace(_profile("pr-cpu"), lane=RegressionLane.GPU)

    with pytest.raises(PerformanceRunnerError, match="not valid for lane"):
        run_profile(manifest, scratch_root=tmp_path)


def test_run_profile_rejects_noncanonical_metric_units(tmp_path: Path) -> None:
    manifest = _profile("pr-cpu")
    budgets = list(manifest.budgets)
    budgets[0] = replace(budgets[0], unit="milliseconds")
    invalid = replace(manifest, budgets=tuple(budgets))

    with pytest.raises(PerformanceRunnerError, match="requires canonical unit"):
        run_profile(invalid, scratch_root=tmp_path)
