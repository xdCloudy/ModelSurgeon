"""Golden compatibility contract for persisted metrics and schema identities."""

from __future__ import annotations

import json
from pathlib import Path

from modelsurgeon.evaluation.tiered import _ALLOWED_METRICS
from modelsurgeon.experiments.identity import IDENTITY_SCHEMA_VERSION
from modelsurgeon.experiments.migrations import EXPERIMENT_DB_SCHEMA_VERSION, MIGRATIONS
from modelsurgeon.experiments.reproducibility import REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION
from modelsurgeon.experiments.schema import (
    EXPERIMENT_SCHEMA_VERSION,
    MUTATION_EXAMPLE_SCHEMA_VERSION,
    MetricObservation,
    MetricState,
)
from modelsurgeon.surgery.contracts import MUTATION_SCHEMA_VERSION
from modelsurgeon.surgery.serialization import MUTATION_RECORD_SCHEMA_VERSION

_FIXTURE = Path(__file__).parent / "fixtures" / "metric_schema_compatibility_v1.json"


def test_metric_and_schema_contract_matches_the_versioned_golden_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["fixture_version"] == 1
    assert fixture["schema_versions"] == {
        "experiment": EXPERIMENT_SCHEMA_VERSION,
        "experiment_database": EXPERIMENT_DB_SCHEMA_VERSION,
        "identity": IDENTITY_SCHEMA_VERSION,
        "mutation": MUTATION_SCHEMA_VERSION,
        "mutation_record": MUTATION_RECORD_SCHEMA_VERSION,
        "mutation_example": MUTATION_EXAMPLE_SCHEMA_VERSION,
        "reproducibility_manifest": REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
    }
    assert fixture["migration_names"] == [item.name for item in MIGRATIONS]
    assert fixture["tier_metrics"] == {
        str(int(tier)): sorted(metrics) for tier, metrics in _ALLOWED_METRICS.items()
    }

    restored = tuple(
        MetricObservation(
            item["name"],
            MetricState(item["state"]),
            item["value"],
            item["unit"],
            item["reason"],
        ).to_record()
        for item in fixture["metric_records"]
    )
    assert list(restored) == fixture["metric_records"]


def test_metric_schema_fixture_changes_require_a_versioned_changelog_marker() -> None:
    changelog = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "metric-schema-v1" in changelog
