"""Acceptance tests for the versioned v2.9 resistance corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_adversarial_resistance import (
    AdversarialResistanceError,
    audit_corpus,
    load_corpus,
    run_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "adversarial_resistance_v1.json"
MANIFEST = ROOT / "docs" / "research" / "v2.9-adversarial-resistance-v1.json"


def test_resistance_corpus_is_versioned_and_complete() -> None:
    corpus = load_corpus(CORPUS)
    assert corpus["schema_version"] == 1
    assert corpus["corpus_revision"] == "v2.9-adversarial-resistance-v1"
    assert corpus["seed"] == 482
    assert corpus["authority_policy"]["allow_arbitrary_execution"] is False
    assert corpus["authority_policy"]["allow_silent_constraint_change"] is False
    assert corpus["authority_policy"]["allow_unvalidated_promotion"] is False
    assert len({case["case_id"] for case in corpus["cases"]}) == len(corpus["cases"])


def test_resistance_manifest_records_dependencies_scope_and_gate() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["record_type"] == "adversarial_resistance_release"
    assert manifest["issue"] == 482
    assert manifest["corpus"]["fixture"] == "tests/fixtures/adversarial_resistance_v1.json"
    assert {item["issue"] for item in manifest["dependency_evidence"]} == {467, 476}
    assert manifest["authority_contract"]["trusted_structured_policy_wins"] is True
    assert manifest["evidence_policy"]["hostile_process_containment"] == "not_claimed"
    assert "git diff --check" in manifest["quality_gate"]["commands"]
    assert "uv run --locked --extra dev pytest -q" in manifest["quality_gate"]["commands"]


def test_every_deterministic_variant_passes_and_retains_negative_results() -> None:
    report = audit_corpus(CORPUS)
    assert report["failed"] == 0
    assert report["all_failures_retained"] is True
    assert report["trusted_policy_wins"] is True
    assert report["variant_count"] == 69
    assert report["unresolved"] == 1
    assert {item["status"] for item in report["observations"]} == {"passed"}
    assert all(item["provenance"].startswith("sha256:") for item in report["observations"])


def test_provider_and_tool_routes_cover_fail_closed_outcomes() -> None:
    report = run_audit(CORPUS)
    observations = report["observations"]
    outcomes = {item["observed"]["outcome"] for item in observations}
    failures = {item["observed"]["failure_code"] for item in observations}
    assert {"supported", "refused", "failed", "malformed_output", "not_claimed"} <= outcomes
    assert {"invalid_input", "budget_exceeded", "approval_invalid", "output_invalid"} <= failures
    assert all(item["retained"] for item in observations)
    assert all(item["observed"]["budget"] for item in observations)


def test_resistance_audit_is_deterministic() -> None:
    first = json.dumps(run_audit(CORPUS), sort_keys=True)
    second = json.dumps(run_audit(CORPUS), sort_keys=True)
    assert first == second


def test_corpus_rejects_a_policy_that_allows_unvalidated_promotion(tmp_path: Path) -> None:
    record = json.loads(CORPUS.read_text(encoding="utf-8"))
    record["authority_policy"]["allow_unvalidated_promotion"] = True
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AdversarialResistanceError, match="unvalidated promotion"):
        load_corpus(path)


def test_corpus_rejects_missing_provenance_or_variant_budget(tmp_path: Path) -> None:
    record = json.loads(CORPUS.read_text(encoding="utf-8"))
    del record["cases"][0]["provenance"]
    record["cases"][1]["budget"]["max_variants"] = 0
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AdversarialResistanceError, match="provenance"):
        load_corpus(path)
