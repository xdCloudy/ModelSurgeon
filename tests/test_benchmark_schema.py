"""Tests for provenance-complete deployable benchmark evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.evaluation.benchmark_schema import (
    BenchmarkArtifact,
    BenchmarkArtifactState,
    BenchmarkBudget,
    BenchmarkBudgetLimit,
    BenchmarkConfidenceInterval,
    BenchmarkCorpus,
    BenchmarkEvidenceRecord,
    BenchmarkHardware,
    BenchmarkMethod,
    BenchmarkMetric,
    BenchmarkMetricState,
    BenchmarkModel,
    BenchmarkOutcome,
    BenchmarkProvenance,
    BenchmarkReliability,
    BenchmarkSchemaError,
    migrate_benchmark_record,
    render_benchmark_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _record(*, outcome: BenchmarkOutcome = BenchmarkOutcome.MEASURED) -> BenchmarkEvidenceRecord:
    measured = outcome is BenchmarkOutcome.MEASURED
    return BenchmarkEvidenceRecord(
        model=BenchmarkModel("tiny/model", "model-revision", "llama", "safetensors"),
        corpus=BenchmarkCorpus(
            "tiny-corpus", "corpus-revision", "validation", "perplexity", "Apache-2.0", "manifest"
        ),
        method=BenchmarkMethod("reference", "method-revision", "dense"),
        hardware=BenchmarkHardware(
            "linux", "reference-cpu", "cpu", "torch", {"python": "3.12", "threads": 2}
        ),
        budget=BenchmarkBudget(
            (
                BenchmarkBudgetLimit("evaluation_tokens", 128, "tokens"),
                BenchmarkBudgetLimit("optimization_seconds", 30, "seconds"),
            ),
            "matched",
        ),
        seeds=(7, 11),
        provenance=BenchmarkProvenance(
            "tool-revision",
            "evaluator-revision",
            "config-revision",
            "source-revision",
            "lock-revision",
            "benchmark --fixture tiny",
        ),
        artifact=BenchmarkArtifact(
            "candidate",
            "artifact-revision",
            "safetensors",
            BenchmarkArtifactState.AVAILABLE if measured else BenchmarkArtifactState.NOT_APPLICABLE,
            "a" * 64 if measured else None,
            123 if measured else None,
        ),
        outcome=outcome,
        quality_metrics=(
            BenchmarkMetric(
                "perplexity",
                "nats/token",
                value=1.2 if measured else None,
                lower_bound=1.1 if measured else None,
                upper_bound=1.3 if measured else None,
                state=BenchmarkMetricState.MEASURED
                if measured
                else BenchmarkMetricState.UNAVAILABLE,
                reason=None if measured else "method does not support this architecture",
            ),
        ),
        runtime_metrics=(
            BenchmarkMetric("latency", "milliseconds/token", value=4.5)
            if measured
            else BenchmarkMetric(
                "latency",
                "milliseconds/token",
                state=BenchmarkMetricState.UNAVAILABLE,
                reason="runtime was not reached",
            ),
        ),
        optimization_cost_metrics=(
            BenchmarkMetric("optimization_time", "seconds", value=12.0)
            if measured
            else BenchmarkMetric(
                "optimization_time",
                "seconds",
                state=BenchmarkMetricState.UNAVAILABLE,
                reason="optimization was not applicable",
            ),
        ),
        reliability=BenchmarkReliability(
            repetitions=3,
            successful_repetitions=3 if measured else 0,
            intervals=(BenchmarkConfidenceInterval("perplexity", 0.95, 1.1, 1.3),)
            if measured
            else (),
        ),
        outcome_reason=None if measured else "method does not support this architecture",
    )


def test_measured_record_has_deterministic_identity_and_complete_groups() -> None:
    record = _record()
    payload = record.to_record()

    assert payload["cell_id"] == record.cell_id
    assert payload["outcome"] == "measured"
    assert payload["artifact"]["digest"] == "a" * 64  # type: ignore[index]
    assert payload["quality_metrics"][0]["unit"] == "nats/token"  # type: ignore[index]
    assert payload["reliability"]["intervals"][0]["level"] == 0.95  # type: ignore[index]


@pytest.mark.parametrize(
    "change",
    (
        lambda record: replace(record, model=replace(record.model, revision="other-model")),
        lambda record: replace(record, corpus=replace(record.corpus, revision="other-corpus")),
        lambda record: replace(record, method=replace(record.method, revision="other-method")),
        lambda record: replace(record, hardware=replace(record.hardware, accelerator="cuda")),
        lambda record: replace(
            record,
            budget=replace(
                record.budget,
                limits=(BenchmarkBudgetLimit("evaluation_tokens", 256, "tokens"),
                        BenchmarkBudgetLimit("optimization_seconds", 30, "seconds")),
            ),
        ),
        lambda record: replace(record, seeds=(7, 13)),
        lambda record: replace(
            record, provenance=replace(record.provenance, tool_revision="other-tool")
        ),
        lambda record: replace(
            record, artifact=replace(record.artifact, revision="other-artifact")
        ),
    ),
)
def test_cell_identity_changes_when_any_comparison_input_changes(change) -> None:
    record = _record()
    assert change(record).cell_id != record.cell_id


@pytest.mark.parametrize(
    "outcome",
    (BenchmarkOutcome.UNSUPPORTED, BenchmarkOutcome.FAILED, BenchmarkOutcome.INCOMPLETE),
)
def test_terminal_outcomes_are_explicit_and_do_not_use_numeric_sentinels(outcome) -> None:
    record = _record(outcome=outcome)
    assert record.to_record()["outcome"] == outcome.value
    assert record.to_record()["outcome_reason"]
    assert record.to_record()["quality_metrics"][0]["value"] is None  # type: ignore[index]


def test_invalid_metric_and_artifact_states_fail_closed() -> None:
    with pytest.raises(BenchmarkSchemaError, match="unavailable metrics"):
        BenchmarkMetric(
            "latency",
            "milliseconds/token",
            state=BenchmarkMetricState.UNAVAILABLE,
            value=0,
            reason="not run",
        )
    with pytest.raises(BenchmarkSchemaError, match="lowercase SHA-256"):
        BenchmarkArtifact(
            "candidate", "revision", "GGUF", BenchmarkArtifactState.AVAILABLE, "A" * 64, 1
        )


def test_migration_preserves_metric_units_and_explicit_outcome() -> None:
    legacy = json.loads((ROOT / "tests" / "fixtures" / "benchmark_evidence_v0.json").read_text())
    migrated = migrate_benchmark_record(legacy)

    assert migrated["schema_version"] == 1
    assert migrated["outcome"] == "unsupported"
    assert migrated["quality_metrics"][0]["unit"] == "nats/token"  # type: ignore[index]
    assert migrated["quality_metrics"][0]["value"] is None  # type: ignore[index]
    assert migrated["runtime_metrics"] == []


def test_schema_can_bind_pinned_hf_and_gguf_evidence_inputs() -> None:
    hf_manifest = json.loads(
        (ROOT / "tests" / "fixtures" / "tiny_hf_models_v1.json").read_text()
    )
    gguf_fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "qwen_gguf_surgery_v1.json").read_text()
    )
    hf_record = _record()
    gguf_record = replace(
        hf_record,
        model=replace(hf_record.model, identifier="tiny-gguf/model", format="GGUF"),
        artifact=replace(hf_record.artifact, format="GGUF"),
    )

    assert hf_manifest["models"][0]["revision"]
    assert gguf_fixture["fixture_version"] == 1
    assert hf_record.model.format == "safetensors"
    assert gguf_record.model.format == "GGUF"
    assert hf_record.cell_id != gguf_record.cell_id


def test_renderers_preserve_units_uncertainty_and_lineage() -> None:
    record = _record()
    rendered_json = json.loads(render_benchmark_report(record))
    rendered_markdown = render_benchmark_report(record, format="markdown")

    assert rendered_json["provenance"]["tool_revision"] == "tool-revision"
    assert "nats/token" in rendered_markdown
    assert "[1.1, 1.3]" in rendered_markdown
    assert "artifact-revision" in rendered_markdown
    with pytest.raises(BenchmarkSchemaError, match="unsupported benchmark report format"):
        render_benchmark_report(record, format="html")
