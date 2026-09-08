from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.experiments.optimization_evidence import (
    OptimizationEvidenceError,
    OptimizationEvidenceOutcome,
    OptimizationEvidenceRecord,
    OptimizationEvidenceStore,
)


def _record(
    outcome: OptimizationEvidenceOutcome = OptimizationEvidenceOutcome.ACCEPTED,
    *,
    measurement: dict[str, object] | None = None,
    failure: dict[str, object] | None = None,
) -> OptimizationEvidenceRecord:
    return OptimizationEvidenceRecord(
        observation_id="obs-test",
        run_id="run-test",
        stage="active_search",
        state_id="source",
        outcome=outcome,
        model={"identifier": "model"},
        dataset={"identifier": "dataset"},
        hardware={"cpu": "test"},
        versions={"runtime": "test"},
        lineage={"parent": "source"},
        candidate_id="candidate-test",
        mutation_id="mutation-test",
        features=({"name": "weight_l1_norm", "value": 1.0},),
        prediction={"value": 0.25, "authority": "prediction_only"},
        measurement=measurement,
        failure=failure,
    )


def test_optimization_evidence_is_immutable_and_manifested(tmp_path: Path) -> None:
    store = OptimizationEvidenceStore(tmp_path / "evidence")
    record = _record(measurement={"accepted": True, "quality_delta": 0.0})

    first = store.publish(record)
    second = store.publish(record)

    assert first == second
    assert first.path.is_file()
    assert json.loads(first.path.read_text(encoding="utf-8"))["outcome"] == "accepted"
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"][record.observation_id] == first.digest

    conflicting = _record(
        measurement={"accepted": True, "quality_delta": 1.0}
    )
    with pytest.raises(OptimizationEvidenceError, match="conflicts"):
        store.publish(conflicting)


def test_outcome_validation_retains_negative_and_rollback_evidence() -> None:
    rejected = _record(
        OptimizationEvidenceOutcome.REJECTED,
        measurement={"accepted": False, "quality_delta": 2.0},
    )
    rolled_back = _record(
        OptimizationEvidenceOutcome.ROLLED_BACK,
        failure={"reason": "quality gate"},
    )

    assert rejected.to_record()["outcome"] == "rejected"
    assert rolled_back.to_record()["outcome"] == "rolled_back"

    with pytest.raises(OptimizationEvidenceError, match="real measurement"):
        _record(OptimizationEvidenceOutcome.ACCEPTED)
    with pytest.raises(OptimizationEvidenceError, match="failure detail"):
        _record(OptimizationEvidenceOutcome.ROLLED_BACK)
