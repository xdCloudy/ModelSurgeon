"""Tests for cross-platform metric and artifact stability evidence."""

from __future__ import annotations

import hashlib

import pytest

from modelsurgeon.evaluation.platform_stability import (
    ArtifactIdentity,
    DriftOutcome,
    MetricObservation,
    MetricTolerance,
    ObservationState,
    PlatformIdentity,
    PlatformStabilityError,
    StabilityStatus,
    ToleranceRegistry,
    compare_platform_stability,
    migrate_platform_stability_record,
)


def _registry() -> ToleranceRegistry:
    return ToleranceRegistry((MetricTolerance("loss", "nats", 1e-5, 1e-5, 1e-3, 1e-3),))


def _identity(supported: bool = True) -> PlatformIdentity:
    return PlatformIdentity("linux", "python-3.12", "float32", 1, "cpu", "rev", "config", supported)


def test_within_tolerance_and_expected_variation_are_distinct() -> None:
    registry = _registry()
    baseline = (MetricObservation("loss", "nats", 1.0),)
    close = compare_platform_stability(
        _identity(), _identity(), baseline, (MetricObservation("loss", "nats", 1.000001),), registry
    )
    expected = compare_platform_stability(
        _identity(), _identity(), baseline, (MetricObservation("loss", "nats", 1.0001),), registry
    )
    assert close.status is StabilityStatus.SUPPORTED
    assert close.drift is DriftOutcome.WITHIN_TOLERANCE
    assert expected.drift is DriftOutcome.EXPECTED_VARIATION


def test_semantic_and_artifact_drift_fail_closed() -> None:
    registry = _registry()
    baseline = (MetricObservation("loss", "nats", 1.0),)
    changed = compare_platform_stability(
        _identity(),
        _identity(),
        baseline,
        (MetricObservation("loss", "nats", 1.1),),
        registry,
        reference_artifact=ArtifactIdentity(hashlib.sha256(b"a").hexdigest(), 1),
        candidate_artifact=ArtifactIdentity(hashlib.sha256(b"b").hexdigest(), 1),
    )
    assert changed.status is StabilityStatus.FAILED
    assert changed.drift is DriftOutcome.ARTIFACT_DRIFT


def test_unsupported_and_unavailable_results_are_retained() -> None:
    registry = _registry()
    unsupported = compare_platform_stability(
        _identity(),
        _identity(False),
        (MetricObservation("loss", "nats", 1.0),),
        (MetricObservation("loss", "nats", 1.0),),
        registry,
    )
    unknown = compare_platform_stability(
        _identity(),
        _identity(),
        (MetricObservation("loss", "nats", None, ObservationState.UNAVAILABLE, "no probe"),),
        (MetricObservation("loss", "nats", 1.0),),
        registry,
    )
    assert unsupported.status is StabilityStatus.UNSUPPORTED
    assert unknown.status is StabilityStatus.UNKNOWN


def test_registry_requires_explicit_metric_definition_provenance() -> None:
    with pytest.raises(PlatformStabilityError, match="changelog"):
        MetricTolerance("loss", "nats", 0.1, 0.1, 0.2, 0.2, changelog_marker="")
    with pytest.raises(PlatformStabilityError, match="semantic"):
        MetricTolerance("loss", "nats", 0.2, 0.1, 0.1, 0.2)


def test_record_ids_and_migration_are_deterministic() -> None:
    registry = _registry()
    record = compare_platform_stability(
        _identity(),
        _identity(),
        (MetricObservation("loss", "nats", 1.0),),
        (MetricObservation("loss", "nats", 1.0),),
        registry,
    )
    assert record.record_id == record.record_id
    migrated = migrate_platform_stability_record({"schema_version": 0, "status": "unknown"})
    assert migrated["schema_version"] == 1
    assert migrated["migration"] == {
        "from_schema_version": 0,
        "to_schema_version": 1,
        "lossless": True,
    }
    with pytest.raises(PlatformStabilityError, match="unsupported"):
        migrate_platform_stability_record({"schema_version": 99})
