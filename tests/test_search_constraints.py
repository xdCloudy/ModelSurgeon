import json

import pytest

from modelsurgeon.config import ConstraintConfig, Settings
from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintError,
    ConstraintMetric,
    ConstraintObservation,
    ConstraintSet,
    OptimizationConstraint,
    constraints_from_config,
)


def _constraint(
    metric: ConstraintMetric,
    threshold: float,
    baseline: BaselineReference = BaselineReference.IMMUTABLE_SOURCE,
) -> OptimizationConstraint:
    return OptimizationConstraint(metric, threshold, baseline)


def test_constraints_compose_canonically_and_fail_closed() -> None:
    constraints = ConstraintSet(
        (
            _constraint(ConstraintMetric.LATENCY_GAIN, 0.1),
            _constraint(ConstraintMetric.QUALITY_RETENTION, 0.98),
            _constraint(ConstraintMetric.PEAK_RAM, 8 << 30, BaselineReference.ABSOLUTE),
        )
    )
    assert [item.metric for item in constraints.constraints] == [
        ConstraintMetric.LATENCY_GAIN,
        ConstraintMetric.PEAK_RAM,
        ConstraintMetric.QUALITY_RETENTION,
    ]
    evaluation = constraints.evaluate(
        (
            ConstraintObservation(
                ConstraintMetric.QUALITY_RETENTION,
                0.99,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
            ConstraintObservation(
                ConstraintMetric.LATENCY_GAIN,
                0.2,
                BaselineReference.PARENT_CANDIDATE,
            ),
        )
    )
    assert evaluation.passed is False
    assert [result.reason for result in evaluation.results] == [
        "baseline_mismatch",
        "missing_observation",
        None,
    ]
    assert (
        constraints.constraint_set_id
        == ConstraintSet(tuple(reversed(constraints.constraints))).constraint_set_id
    )
    json.dumps(constraints.to_record(), sort_keys=True)


def test_config_materializes_explicit_units_baselines_and_limits() -> None:
    config = ConstraintConfig(
        min_quality_retention_ratio=0.97,
        max_perplexity_delta=0.02,
        min_latency_gain_ratio=0.1,
        max_ram_bytes=8 << 30,
        max_vram_bytes=6 << 30,
        max_disk_bytes=20 << 30,
    )
    constraints = constraints_from_config(config)
    records = constraints.to_record()["constraints"]
    assert isinstance(records, list)
    assert {record["unit"] for record in records} == {"ratio", "perplexity_points", "bytes"}
    assert Settings(constraints=config).canonical_dict()["constraints"]["max_ram_bytes"] == 8 << 30


def test_ambiguous_or_invalid_constraints_are_rejected() -> None:
    with pytest.raises(ConstraintError, match="absolute baselines"):
        _constraint(ConstraintMetric.PEAK_VRAM, 1, BaselineReference.IMMUTABLE_SOURCE)
    with pytest.raises(ConstraintError, match="model baseline"):
        _constraint(ConstraintMetric.LATENCY_GAIN, 0.1, BaselineReference.ABSOLUTE)
    with pytest.raises(ConstraintError, match="within"):
        _constraint(ConstraintMetric.QUALITY_RETENTION, 1.1)
    duplicate = _constraint(ConstraintMetric.QUALITY_RETENTION, 0.9)
    with pytest.raises(ConstraintError, match="unique"):
        ConstraintSet((duplicate, duplicate))
