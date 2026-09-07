"""Tests for bounded fault matrices and crash-consistency invariants."""

from __future__ import annotations

import sys

import pytest

from modelsurgeon.resilience import (
    FaultPoint,
    RecoveryOutcome,
    ResilienceError,
    bounded_subprocess,
    generate_fault_matrix,
    verify_recovery_invariants,
)


def test_fault_matrix_is_deterministic_and_documents_recovery() -> None:
    first = generate_fault_matrix(seed=408)
    second = generate_fault_matrix(seed=408)
    assert [item.scenario_id for item in first] == [item.scenario_id for item in second]
    assert {item.point for item in first} == set(FaultPoint)
    assert all(item.to_record()["recovery_action"] for item in first)


def test_recovery_preserves_source_accepted_and_only_owned_staging() -> None:
    before = {
        "source.bin": "source-v1",
        "accepted.bin": "accepted-v1",
        ".modelsurgeon.x.staging": "partial",
        "neighbor.bin": "neighbor",
    }
    after = {
        "source.bin": "source-v1",
        "accepted.bin": "accepted-v1",
        "neighbor.bin": "neighbor",
    }
    source, accepted, removed = verify_recovery_invariants(
        before,
        after,
        source_paths=("source.bin",),
        accepted_paths=("accepted.bin",),
        owned_staging_prefix=".modelsurgeon.",
    )
    assert source and accepted
    assert removed == (".modelsurgeon.x.staging",)
    with pytest.raises(ResilienceError, match="outside owned"):
        verify_recovery_invariants(
            before,
            {"source.bin": "source-v1", "accepted.bin": "accepted-v1"},
            source_paths=("source.bin",),
            accepted_paths=("accepted.bin",),
            owned_staging_prefix=".modelsurgeon.",
        )


def test_timeout_terminates_child_and_retains_bounded_log() -> None:
    evidence = bounded_subprocess(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(1)"],
        timeout_seconds=0.05,
        grace_seconds=0.05,
        max_log_bytes=32,
    )
    assert evidence.outcome is RecoveryOutcome.UNKNOWN
    assert "timed out" in evidence.detail
    assert len(evidence.logs.encode()) <= 32
