from __future__ import annotations

import pytest

from modelsurgeon.experiments import (
    ReplayCursor,
    ReplayEnvironmentError,
    ReplayEnvironmentKind,
    ReplayEnvironmentLock,
    ReplayEnvironmentObservation,
    ReplayMetricTolerance,
    ReplayMigrationChain,
    ReplayMigrationCopy,
    ReplayRecipe,
    ReplayStatus,
    compare_replay_metrics,
    diagnose_replay_environment,
    prepare_replay_plan,
)


def _lock() -> ReplayEnvironmentLock:
    return ReplayEnvironmentLock(
        ReplayEnvironmentKind.DOCKER,
        "linux-cuda",
        "a" * 64,
        "toolchain-v1",
        "x86_64",
        "low_vram_12gb",
        "nvidia",
        ("cuda", "python", "torch"),
    )


def _recipe(lock: ReplayEnvironmentLock) -> ReplayRecipe:
    return ReplayRecipe(
        "evidence_bundle_1",
        lock.lock_id,
        ("modelsurgeon", "replay", "--seed", "7"),
        (7, 8, 9),
        ("b" * 64,),
        (("benchmark", 1), ("manifest", 2)),
        (ReplayMetricTolerance("quality", 0.01, 0.01),),
    )


def test_environment_mismatch_blocks_before_execution() -> None:
    expected = _lock()
    actual = ReplayEnvironmentObservation(
        expected.kind,
        expected.identifier,
        "c" * 64,
        expected.toolchain_revision,
        expected.architecture,
        expected.hardware_class,
        expected.gpu_passthrough,
        ("python",),
    )
    mismatches = diagnose_replay_environment(expected, actual)
    plan = prepare_replay_plan(_recipe(expected), expected, actual, ReplayCursor(("load", "run")))

    assert {item.code for item in mismatches} == {"environment_mismatch", "missing_capability"}
    assert plan.status is ReplayStatus.BLOCKED
    assert not plan.executable


def test_cursor_and_ordered_schema_migrations_are_deterministic() -> None:
    cursor = ReplayCursor(("load", "execute", "verify"))
    cursor = cursor.advance("load")
    cursor = cursor.advance("execute")
    assert cursor.next_step == "verify"
    assert cursor.cursor_id.startswith("replay_cursor_")

    original = "d" * 64
    first = ReplayMigrationCopy(original, 1, 2, original, "e" * 64, "migrate-1-2")
    second = ReplayMigrationCopy(original, 2, 3, "e" * 64, "f" * 64, "migrate-2-3")
    chain = ReplayMigrationChain((first, second))

    assert chain.final_digest == "f" * 64
    with pytest.raises(ReplayEnvironmentError, match="skip or repeat"):
        ReplayCursor(("load", "run")).advance("run")


def test_metric_comparison_requires_declared_tolerance() -> None:
    comparisons = compare_replay_metrics(
        {"quality": 1.0, "latency": 10.0},
        {"quality": 1.005, "latency": 10.0, "extra": 2.0},
        (ReplayMetricTolerance("quality", 0.001, 0.01),),
    )
    by_name = {item.metric: item for item in comparisons}

    assert by_name["quality"].passed
    assert by_name["latency"].passed
    assert not by_name["extra"].passed


def test_docker_lock_requires_explicit_gpu_boundary() -> None:
    with pytest.raises(ReplayEnvironmentError, match="GPU passthrough"):
        ReplayEnvironmentLock(
            ReplayEnvironmentKind.DOCKER,
            "docker",
            "a" * 64,
            "toolchain",
            "x86_64",
            "cpu",
            "none",
            ("python",),
        )
