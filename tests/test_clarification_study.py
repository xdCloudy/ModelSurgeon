"""Reproducible v2.4 clarification measurement tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    ClarificationStudyError,
    ClarificationStudyPolicy,
    load_and_run_clarification_study,
    load_clarification_study,
)
from modelsurgeon.search.intent_corpus import load_intent_corpus

ROOT = Path(__file__).resolve().parents[1]
INTENT_CORPUS = ROOT / "tests" / "fixtures" / "intent_compiler_corpus_v1.json"
STUDY = ROOT / "tests" / "fixtures" / "clarification_measurement_v1.json"


def test_bounded_study_retains_splits_negative_and_inconclusive_cases() -> None:
    corpus = load_intent_corpus(INTENT_CORPUS)
    study = load_clarification_study(STUDY, corpus)

    assert study.protocol_id == "v24-clarification-measurement-v1"
    assert study.limits.provider_replay_seeds == ()
    assert {case.split.value for case in study.cases} == {
        "fixture",
        "held_out",
        "adversarial",
    }
    assert {case.classification.value for case in study.cases} >= {
        "refused",
        "unsupported",
        "inconclusive",
    }
    assert all(case.retained for case in study.cases)


def test_schema_driven_policy_meets_preregistered_safety_targets() -> None:
    run = load_and_run_clarification_study(STUDY, INTENT_CORPUS)
    metrics = next(
        item for item in run.metrics if item.policy is ClarificationStudyPolicy.SCHEMA_DRIVEN
    )

    assert run.passed
    assert run.policy_passed(ClarificationStudyPolicy.SCHEMA_DRIVEN)
    assert metrics.necessary_question_recall == 1.0
    assert metrics.unnecessary_question_rate == 0.0
    assert metrics.silent_constraint_invention_rate == 0.0
    assert metrics.executable_spec_precision == 1.0
    assert metrics.executable_spec_recall == 1.0
    assert metrics.exact_spec_equivalence_rate == 1.0
    assert metrics.refusal_correctness_rate == 1.0

    results = [
        item for item in run.results if item.policy is ClarificationStudyPolicy.SCHEMA_DRIVEN
    ]
    assert len(results) == len(
        load_clarification_study(STUDY, load_intent_corpus(INTENT_CORPUS)).cases
    )
    assert all(item.retained and item.error is None for item in results)
    assert all(item.spec_equivalent for item in results)


def test_baselines_make_the_declared_tradeoffs_visible() -> None:
    run = load_and_run_clarification_study(STUDY, INTENT_CORPUS)
    metrics = {item.policy: item for item in run.metrics}

    assert metrics[ClarificationStudyPolicy.SCHEMA_ONLY].necessary_question_recall == 0.0
    assert metrics[ClarificationStudyPolicy.MINIMAL_QUESTION].necessary_question_recall == 1.0
    assert metrics[ClarificationStudyPolicy.MINIMAL_QUESTION].unnecessary_question_rate == 0.0
    assert metrics[ClarificationStudyPolicy.OVER_QUESTIONING].unnecessary_question_rate == 1.0
    assert all(
        item.silent_constraint_invention_rate == 0.0
        for item in metrics.values()
    )


def test_run_identity_and_question_identifiers_are_byte_stable() -> None:
    first = load_and_run_clarification_study(STUDY, INTENT_CORPUS)
    second = load_and_run_clarification_study(STUDY, INTENT_CORPUS)

    assert first.run_id == second.run_id
    assert first.canonical_json() == second.canonical_json()
    assert all(item.result_id for item in first.results)
    assert len({item.result_id for item in first.results}) == len(first.results)


def test_study_rejects_a_different_source_corpus_revision(tmp_path: Path) -> None:
    payload = STUDY.read_text(encoding="utf-8").replace(
        "v2.1-intent-corpus-v1", "different-corpus-revision"
    )
    path = tmp_path / "study.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ClarificationStudyError, match="different intent corpus revision"):
        load_clarification_study(path, load_intent_corpus(INTENT_CORPUS))
