"""Regression checks for the frozen v2.1 conversational intent contract."""

from __future__ import annotations

import json
from pathlib import Path

from modelsurgeon.conversation import (
    CONVERSATIONAL_INTENT_SCHEMA_VERSION,
)
from modelsurgeon.search import (
    INTENT_COMPILER_SCHEMA_VERSION,
    INTENT_CORPUS_SCHEMA_VERSION,
    INTENT_POLICY_SCHEMA_VERSION,
    OBJECTIVE_CONTRACT_SCHEMA_VERSION,
    compile_intent,
    compile_intent_record,
    load_intent_corpus,
    run_intent_corpus,
)

ROOT = Path(__file__).parents[1]
FREEZE_PATH = ROOT / "docs" / "research" / "v2.1-conversational-intent-contract-v1.json"


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_freeze_record_is_strictly_versioned_and_points_at_existing_artifacts() -> None:
    freeze = _freeze()

    assert freeze["schema_version"] == 1
    assert freeze["record_type"] == "v2.1_conversational_intent_contract_freeze"
    assert freeze["status"] == "frozen"
    assert freeze["outcomes"] == [
        "executable",
        "clarification_required",
        "unsupported",
        "refused",
    ]

    stable = freeze["stable_optimization_schema"]
    assert isinstance(stable, dict)
    assert stable["schema_version"] == OBJECTIVE_CONTRACT_SCHEMA_VERSION
    assert (ROOT / stable["definition"]).is_file()

    components = freeze["versioned_components"]
    assert isinstance(components, list)
    expected_versions = {
        "intent_records": CONVERSATIONAL_INTENT_SCHEMA_VERSION,
        "intent_compiler": INTENT_COMPILER_SCHEMA_VERSION,
        "intent_policy": INTENT_POLICY_SCHEMA_VERSION,
        "equivalence_refusal_corpus": INTENT_CORPUS_SCHEMA_VERSION,
    }
    assert {component["name"] for component in components} == set(expected_versions)
    for component in components:
        assert component["schema_version"] == expected_versions[component["name"]]
        assert (ROOT / component["definition"]).is_file()
        if "fixtures" in component:
            assert (ROOT / component["fixtures"]).is_file()


def test_frozen_contract_replays_both_available_compiler_entrypoints() -> None:
    corpus = load_intent_corpus(ROOT / "tests" / "fixtures" / "intent_compiler_corpus_v1.json")
    run = run_intent_corpus(
        corpus,
        {"canonical": compile_intent_record, "public": compile_intent},
    )

    assert run.passed
    assert run.implementations == ("canonical", "public")
    assert all(result.retained for result in run.results)

    freeze = _freeze()
    protocol = freeze["replay_protocol"]
    assert isinstance(protocol, dict)
    assert "tests/test_v21_intent_contract.py" in protocol["focused_command"]
    assert "uv run ruff check ." in protocol["quality_gate"]
    assert "uv run mypy src" in protocol["quality_gate"]


def test_frozen_contract_keeps_non_executable_evidence_spec_free() -> None:
    corpus = load_intent_corpus(ROOT / "tests" / "fixtures" / "intent_compiler_corpus_v1.json")
    run = run_intent_corpus(corpus)

    assert run.passed
    for result in run.results:
        if result.policy_outcome is not None and result.policy_outcome.value != "executable":
            assert result.policy_spec is None
        if result.compiler_outcome is not None and result.compiler_outcome.value != "executable":
            assert result.compiler_spec is None
