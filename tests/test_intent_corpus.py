"""Regression corpus tests for equivalence, refusal, and retention boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.conversation import IntentRecord
from modelsurgeon.search import (
    CorpusCaseClass,
    IntentCompilation,
    IntentCorpusError,
    compile_intent,
    compile_intent_record,
    load_intent_corpus,
    run_intent_corpus,
)

CORPUS_PATH = Path(__file__).parent / "fixtures" / "intent_compiler_corpus_v1.json"


def test_versioned_corpus_passes_all_supported_compiler_entrypoints() -> None:
    corpus = load_intent_corpus(CORPUS_PATH)
    run = run_intent_corpus(
        corpus,
        {
            "canonical": compile_intent_record,
            "public": compile_intent,
        },
    )

    assert run.passed
    assert run.corpus_revision == "v2.1-intent-corpus-v1"
    assert len(run.results) == len(corpus.cases) * 2
    assert all(result.retained for result in run.results)


def test_equivalence_group_has_one_stable_hard_constraint_preserving_spec() -> None:
    corpus = load_intent_corpus(CORPUS_PATH)
    run = run_intent_corpus(corpus)
    equivalent = [
        result
        for result in run.results
        if result.equivalence_group == "equivalence-quality-latency"
    ]

    assert len(equivalent) == 3
    assert len({json.dumps(result.compiler_spec, sort_keys=True) for result in equivalent}) == 1
    assert all(
        result.compiler_spec is not None
        and result.compiler_spec["constraints"] == [
            {
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.95,
                "unit": "ratio",
                "baseline": "absolute",
            }
        ]
        for result in equivalent
    )


def test_negative_unsupported_and_inconclusive_cases_are_visible_and_spec_free() -> None:
    corpus = load_intent_corpus(CORPUS_PATH)
    run = run_intent_corpus(corpus)
    by_case = {result.case_id: result for result in run.results}
    assert {case.expected.classification for case in corpus.cases} == {
        CorpusCaseClass.SUPPORTED,
        CorpusCaseClass.CLARIFICATION,
        CorpusCaseClass.UNSUPPORTED,
        CorpusCaseClass.REFUSED,
        CorpusCaseClass.INCONCLUSIVE,
    }

    for case_id in (
        "contradictory-quality-constraints",
        "input-refusal-retained",
        "missing-hard-constraint",
        "required-vague-constraint",
        "unsupported-budget-declaration",
        "unsupported-operation-declaration",
    ):
        result = by_case[case_id]
        assert result.retained
        assert result.policy_spec is None
        assert result.compiler_spec is None
        assert result.passed

    low_confidence = by_case["low-confidence-required-constraint"]
    assert low_confidence.compiler_spec is not None
    assert low_confidence.policy_spec is None
    assert low_confidence.passed


def test_implementation_failure_is_retained_as_visible_negative_evidence() -> None:
    corpus = load_intent_corpus(CORPUS_PATH)

    def broken(_intent: IntentRecord) -> IntentCompilation:
        raise RuntimeError("provider/compiler fixture failure")

    run = run_intent_corpus(corpus, {"broken": broken})  # type: ignore[dict-item]

    assert not run.passed
    assert len(run.results) == len(corpus.cases)
    assert all(not result.passed for result in run.results)
    assert all(result.retained and result.error for result in run.results)


def test_corpus_rejects_provenance_drift(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["expected"]["provenance"]["tool_revision"] = "wrong-tool"
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IntentCorpusError, match="expected provenance"):
        load_intent_corpus(path)


def test_corpus_rejects_unknown_root_fields(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IntentCorpusError, match="unknown fields"):
        load_intent_corpus(path)
