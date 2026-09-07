"""Reproducible, bounded measurement for clarification policy.

The study is deliberately outside the clarification authority.  It loads a
versioned set of already-typed intent records, observes the existing
fail-closed state machine and compares it with intentionally simple baselines.
No provider is called and no answer is synthesized.  Every observation,
including a baseline failure or an inconclusive source case, is retained.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .clarification import (
    ClarificationMachine,
    ClarificationQuestion,
    ClarificationState,
)
from .intent import IntentOutcome, IntentProvenance, IntentRecord

if TYPE_CHECKING:
    from modelsurgeon.search.intent_corpus import CorpusCaseClass, IntentCorpus, IntentCorpusCase

CLARIFICATION_STUDY_SCHEMA_VERSION = 1
MAX_CLARIFICATION_STUDY_BYTES = 256_000
MAX_CLARIFICATION_STUDY_CASES = 64
MAX_CLARIFICATION_STUDY_QUESTIONS = 16


class ClarificationStudyError(ValueError):
    """Raised when a study definition exceeds its reproducibility boundary."""


class ClarificationStudySplit(StrEnum):
    FIXTURE = "fixture"
    HELD_OUT = "held_out"
    ADVERSARIAL = "adversarial"


class ClarificationStudyPolicy(StrEnum):
    SCHEMA_ONLY = "schema_only"
    MINIMAL_QUESTION = "minimal_question"
    SCHEMA_DRIVEN = "schema_driven"
    OVER_QUESTIONING = "over_questioning"


@dataclass(frozen=True, slots=True)
class ClarificationStudyLimits:
    """Hard CPU and output bounds for one local study run."""

    max_cases: int = MAX_CLARIFICATION_STUDY_CASES
    max_questions_per_case: int = MAX_CLARIFICATION_STUDY_QUESTIONS
    provider_replay_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_cases, bool) or not isinstance(self.max_cases, int):
            raise ClarificationStudyError("study max_cases must be an integer")
        if self.max_cases <= 0 or self.max_cases > MAX_CLARIFICATION_STUDY_CASES:
            raise ClarificationStudyError("study max_cases exceeds its bound")
        if (
            isinstance(self.max_questions_per_case, bool)
            or not isinstance(self.max_questions_per_case, int)
            or self.max_questions_per_case <= 0
            or self.max_questions_per_case > MAX_CLARIFICATION_STUDY_QUESTIONS
        ):
            raise ClarificationStudyError("study question budget exceeds its bound")
        if self.provider_replay_seeds != tuple(sorted(set(self.provider_replay_seeds))):
            raise ClarificationStudyError("provider replay seeds must be sorted and unique")
        if len(self.provider_replay_seeds) > 3:
            raise ClarificationStudyError("study allows at most three provider replay seeds")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.provider_replay_seeds
        ):
            raise ClarificationStudyError("provider replay seeds must be non-negative integers")

    def to_record(self) -> dict[str, object]:
        return {
            "max_cases": self.max_cases,
            "max_questions_per_case": self.max_questions_per_case,
            "provider_replay_seeds": list(self.provider_replay_seeds),
        }


@dataclass(frozen=True, slots=True)
class ClarificationStudyCase:
    """Gold labels for one source-corpus intent, kept separate from policy output."""

    case_id: str
    intent_case_id: str
    split: ClarificationStudySplit
    expected_outcome: IntentOutcome
    necessary_question: bool
    expected_question_categories: tuple[str, ...]
    classification: CorpusCaseClass
    retained: bool = True

    def __post_init__(self) -> None:
        _identifier(self.case_id, "study case ID")
        _identifier(self.intent_case_id, "intent case ID")
        if self.expected_question_categories != tuple(
            sorted(set(self.expected_question_categories))
        ):
            raise ClarificationStudyError("expected question categories must be sorted and unique")
        if not self.retained:
            raise ClarificationStudyError("negative and inconclusive cases must be retained")

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "intent_case_id": self.intent_case_id,
            "split": self.split.value,
            "expected_outcome": self.expected_outcome.value,
            "necessary_question": self.necessary_question,
            "expected_question_categories": list(self.expected_question_categories),
            "classification": self.classification.value,
            "retained": self.retained,
        }


@dataclass(frozen=True, slots=True)
class ClarificationStudyCorpus:
    """Versioned study metadata joined to one exact intent corpus revision."""

    corpus_id: str
    corpus_revision: str
    source_corpus_id: str
    source_corpus_revision: str
    limits: ClarificationStudyLimits
    cases: tuple[ClarificationStudyCase, ...]
    protocol_id: str = "v24-clarification-measurement-v1"
    schema_version: int = CLARIFICATION_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CLARIFICATION_STUDY_SCHEMA_VERSION:
            raise ClarificationStudyError("unsupported clarification study schema version")
        _identifier(self.corpus_id, "study corpus ID")
        if not self.corpus_revision.strip() or not self.source_corpus_revision.strip():
            raise ClarificationStudyError("study corpus revisions are required")
        if not self.cases or len(self.cases) > self.limits.max_cases:
            raise ClarificationStudyError("study case count exceeds its declared bound")
        ids = tuple(item.case_id for item in self.cases)
        if ids != tuple(sorted(set(ids))):
            raise ClarificationStudyError("study case IDs must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "source_corpus_id": self.source_corpus_id,
            "source_corpus_revision": self.source_corpus_revision,
            "limits": self.limits.to_record(),
            "cases": [item.to_record() for item in self.cases],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class ClarificationStudyResult:
    """One retained policy/case observation, including visible failures."""

    policy: ClarificationStudyPolicy
    case_id: str
    split: ClarificationStudySplit
    intent_id: str
    provenance: IntentProvenance
    actual_outcome: IntentOutcome | None
    executable: bool
    question_count: int
    question_categories: tuple[str, ...]
    question_ids: tuple[str, ...]
    expected_outcome: IntentOutcome
    necessary_question: bool
    expected_question_categories: tuple[str, ...]
    spec_equivalent: bool
    silent_constraint_invention: bool
    retained: bool
    mismatches: tuple[str, ...] = ()
    error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.error is None

    @property
    def result_id(self) -> str:
        return "clarification_study_result_" + _digest(
            self.to_record(include_identity=False)
        )[len("sha256:") :]

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "policy": self.policy.value,
            "case_id": self.case_id,
            "split": self.split.value,
            "intent_id": self.intent_id,
            "provenance": self.provenance.to_record(),
            "actual_outcome": None if self.actual_outcome is None else self.actual_outcome.value,
            "executable": self.executable,
            "question_count": self.question_count,
            "question_categories": list(self.question_categories),
            "question_ids": list(self.question_ids),
            "expected_outcome": self.expected_outcome.value,
            "necessary_question": self.necessary_question,
            "expected_question_categories": list(self.expected_question_categories),
            "spec_equivalent": self.spec_equivalent,
            "silent_constraint_invention": self.silent_constraint_invention,
            "retained": self.retained,
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "error": self.error,
        }
        if include_identity:
            record["result_id"] = self.result_id
        return record


@dataclass(frozen=True, slots=True)
class ClarificationStudyMetrics:
    """Rates computed from all retained results for one policy."""

    policy: ClarificationStudyPolicy
    case_count: int
    necessary_case_count: int
    necessary_question_hits: int
    no_question_case_count: int
    unnecessary_question_cases: int
    silent_constraint_inventions: int
    executable_true_positives: int
    executable_predictions: int
    executable_gold: int
    exact_spec_matches: int
    refusal_true_positives: int
    refusal_gold: int
    clarification_cost: float

    @property
    def necessary_question_recall(self) -> float:
        return _rate(self.necessary_question_hits, self.necessary_case_count)

    @property
    def unnecessary_question_rate(self) -> float:
        return _rate(self.unnecessary_question_cases, self.no_question_case_count)

    @property
    def silent_constraint_invention_rate(self) -> float:
        return _rate(self.silent_constraint_inventions, self.case_count)

    @property
    def executable_spec_precision(self) -> float:
        return _rate(self.executable_true_positives, self.executable_predictions)

    @property
    def executable_spec_recall(self) -> float:
        return _rate(self.executable_true_positives, self.executable_gold)

    @property
    def exact_spec_equivalence_rate(self) -> float:
        return _rate(self.exact_spec_matches, self.case_count)

    @property
    def refusal_correctness_rate(self) -> float:
        return _rate(self.refusal_true_positives, self.refusal_gold)

    def to_record(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "case_count": self.case_count,
            "necessary_case_count": self.necessary_case_count,
            "necessary_question_hits": self.necessary_question_hits,
            "necessary_question_recall": self.necessary_question_recall,
            "no_question_case_count": self.no_question_case_count,
            "unnecessary_question_cases": self.unnecessary_question_cases,
            "unnecessary_question_rate": self.unnecessary_question_rate,
            "silent_constraint_inventions": self.silent_constraint_inventions,
            "silent_constraint_invention_rate": self.silent_constraint_invention_rate,
            "executable_spec_precision": self.executable_spec_precision,
            "executable_spec_recall": self.executable_spec_recall,
            "exact_spec_equivalence_rate": self.exact_spec_equivalence_rate,
            "refusal_correctness_rate": self.refusal_correctness_rate,
            "clarification_cost": self.clarification_cost,
        }


@dataclass(frozen=True, slots=True)
class ClarificationStudyRun:
    """Complete retained study run and its policy-specific measurements."""

    protocol_id: str
    corpus_id: str
    corpus_revision: str
    source_corpus_id: str
    source_corpus_revision: str
    limits: ClarificationStudyLimits
    policies: tuple[ClarificationStudyPolicy, ...]
    results: tuple[ClarificationStudyResult, ...]
    metrics: tuple[ClarificationStudyMetrics, ...]
    schema_version: int = CLARIFICATION_STUDY_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.policy_passed(ClarificationStudyPolicy.SCHEMA_DRIVEN)

    def policy_passed(self, policy: ClarificationStudyPolicy) -> bool:
        return all(
            result.passed for result in self.results if result.policy is policy
        )

    @property
    def run_id(self) -> str:
        return "clarification_study_run_" + _digest(
            self.to_record(include_identity=False)
        )[len("sha256:") :]

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "source_corpus_id": self.source_corpus_id,
            "source_corpus_revision": self.source_corpus_revision,
            "limits": self.limits.to_record(),
            "policies": [item.value for item in self.policies],
            "passed": self.passed,
            "results": [item.to_record() for item in self.results],
            "metrics": [item.to_record() for item in self.metrics],
        }
        if include_identity:
            record["run_id"] = self.run_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


def load_clarification_study(
    path: Path, intent_corpus: IntentCorpus
) -> ClarificationStudyCorpus:
    """Load study labels and verify that they bind to one source corpus revision."""

    try:
        if path.stat().st_size > MAX_CLARIFICATION_STUDY_BYTES:
            raise ClarificationStudyError("clarification study exceeds its byte budget")
        root = json.loads(path.read_text(encoding="utf-8"))
    except ClarificationStudyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClarificationStudyError("clarification study is not valid UTF-8 JSON") from error
    record = _object(root, "clarification study")
    expected = {
        "schema_version",
        "protocol_id",
        "corpus_id",
        "corpus_revision",
        "source_corpus_id",
        "source_corpus_revision",
        "limits",
        "cases",
    }
    if set(record) != expected:
        raise ClarificationStudyError("clarification study has missing or unknown fields")
    if record["schema_version"] != CLARIFICATION_STUDY_SCHEMA_VERSION:
        raise ClarificationStudyError("unsupported clarification study schema version")
    source_id = _text(record["source_corpus_id"], "source corpus ID")
    source_revision = _text(record["source_corpus_revision"], "source corpus revision")
    if source_id != intent_corpus.corpus_id or source_revision != intent_corpus.corpus_revision:
        raise ClarificationStudyError("study is bound to a different intent corpus revision")
    limits = _limits(record["limits"])
    raw_cases = record["cases"]
    if not isinstance(raw_cases, list):
        raise ClarificationStudyError("study cases must be an array")
    cases = tuple(_case(item) for item in raw_cases)
    source_cases = {item.case_id for item in intent_corpus.cases}
    if any(item.intent_case_id not in source_cases for item in cases):
        raise ClarificationStudyError("study references an unknown intent corpus case")
    return ClarificationStudyCorpus(
        _text(record["corpus_id"], "study corpus ID"),
        _text(record["corpus_revision"], "study corpus revision"),
        source_id,
        source_revision,
        limits,
        cases,
        _text(record["protocol_id"], "protocol ID"),
    )


def run_clarification_study(
    corpus: ClarificationStudyCorpus,
    intent_corpus: IntentCorpus,
    policies: Sequence[ClarificationStudyPolicy] | None = None,
) -> ClarificationStudyRun:
    """Run every bounded policy/case cell and retain all observations."""

    selected_values = tuple(ClarificationStudyPolicy) if policies is None else tuple(policies)
    selected = tuple(sorted(set(selected_values), key=lambda item: item.value))
    if not selected:
        raise ClarificationStudyError("study requires at least one policy")
    if any(not isinstance(item, ClarificationStudyPolicy) for item in selected):
        raise ClarificationStudyError("study policy is unknown")
    source = {item.case_id: item for item in intent_corpus.cases}
    results: list[ClarificationStudyResult] = []
    for policy in selected:
        for study_case in corpus.cases:
            results.append(
                _run_case(policy, study_case, source[study_case.intent_case_id], corpus.limits)
            )
    retained = tuple(results)
    metrics = tuple(_metrics(policy, retained) for policy in selected)
    return ClarificationStudyRun(
        corpus.protocol_id,
        corpus.corpus_id,
        corpus.corpus_revision,
        corpus.source_corpus_id,
        corpus.source_corpus_revision,
        corpus.limits,
        selected,
        retained,
        metrics,
    )


def load_and_run_clarification_study(
    study_path: Path, intent_corpus_path: Path
) -> ClarificationStudyRun:
    """Convenience entrypoint used by the reproducibility script."""

    from modelsurgeon.search.intent_corpus import load_intent_corpus

    intent_corpus = load_intent_corpus(intent_corpus_path)
    study = load_clarification_study(study_path, intent_corpus)
    return run_clarification_study(study, intent_corpus)


def _run_case(
    policy: ClarificationStudyPolicy,
    study_case: ClarificationStudyCase,
    source_case: IntentCorpusCase,
    limits: ClarificationStudyLimits,
) -> ClarificationStudyResult:
    try:
        observation = _observe(policy, source_case.intent)
        if len(observation.questions) > limits.max_questions_per_case:
            raise ClarificationStudyError("question budget exceeded")
        actual_spec = observation.state.policy.contract if observation.state is not None else None
        expected_spec = source_case.expected.policy_spec
        spec_equivalent = (
            None if actual_spec is None else actual_spec.to_record()
        ) == expected_spec
        mismatches = []
        if observation.outcome is not study_case.expected_outcome:
            mismatches.append("outcome")
        if not spec_equivalent:
            mismatches.append("exact-spec-equivalence")
        if policy is ClarificationStudyPolicy.SCHEMA_DRIVEN and (
            tuple(item.category for item in observation.questions)
            != study_case.expected_question_categories
        ):
            mismatches.append("question-categories")
        silent = (
            study_case.expected_outcome is not IntentOutcome.EXECUTABLE
            and observation.executable
        )
        if silent:
            mismatches.append("silent-constraint-invention")
        return ClarificationStudyResult(
            policy,
            study_case.case_id,
            study_case.split,
            source_case.intent.intent_id,
            source_case.intent.provenance,
            observation.outcome,
            observation.executable,
            len(observation.questions),
            tuple(item.category for item in observation.questions),
            tuple(item.question_id for item in observation.questions),
            study_case.expected_outcome,
            study_case.necessary_question,
            study_case.expected_question_categories,
            spec_equivalent,
            silent,
            study_case.retained,
            tuple(sorted(mismatches)),
        )
    except Exception as error:  # retained as visible negative evidence
        return ClarificationStudyResult(
            policy,
            study_case.case_id,
            study_case.split,
            source_case.intent.intent_id,
            source_case.intent.provenance,
            None,
            False,
            0,
            (),
            (),
            study_case.expected_outcome,
            study_case.necessary_question,
            study_case.expected_question_categories,
            False,
            False,
            study_case.retained,
            ("implementation-error",),
            f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True, slots=True)
class _Observation:
    outcome: IntentOutcome
    executable: bool
    questions: tuple[ClarificationQuestion, ...]
    state: ClarificationState | None


def _observe(policy: ClarificationStudyPolicy, intent: IntentRecord) -> _Observation:
    from modelsurgeon.search.intent_policy import evaluate_intent_policy

    evaluated = evaluate_intent_policy(intent)
    if policy is ClarificationStudyPolicy.SCHEMA_ONLY:
        return _Observation(evaluated.outcome, evaluated.executable, (), None)
    machine = ClarificationMachine()
    state = machine.start(intent, evaluated)
    if policy is ClarificationStudyPolicy.SCHEMA_DRIVEN:
        return _Observation(state.policy.outcome, state.executable, state.questions, state)
    if policy is ClarificationStudyPolicy.MINIMAL_QUESTION:
        questions = (
            ()
            if not state.questions
            and state.policy.outcome is not IntentOutcome.CLARIFICATION_REQUIRED
            else (
                state.questions[:1]
                if state.questions
                else (
                    _synthetic_question(intent, "minimal-question", "clarification.generic"),
                )
            )
        )
        return _Observation(state.policy.outcome, state.executable, questions, state)
    confirmations = tuple(
        _synthetic_question(intent, "over-questioning", f"confirmation.{field.field_id}")
        for field in intent.fields
    )
    if not confirmations:
        confirmations = (_synthetic_question(intent, "over-questioning", "confirmation.generic"),)
    questions = tuple(sorted((*state.questions, *confirmations), key=lambda item: item.question_id))
    return _Observation(state.policy.outcome, state.executable, questions, state)


def _synthetic_question(
    intent: IntentRecord, category: str, field_id: str
) -> ClarificationQuestion:
    identity = {"intent_id": intent.intent_id, "category": category, "field_id": field_id}
    question_id = "study_question_" + _digest(identity)[len("sha256:") :]
    return ClarificationQuestion(
        question_id,
        field_id,
        category,
        "Study baseline question; no objective value is inferred.",
        (),
        False,
        (),
        ("clarification-study-baseline",),
    )


def _metrics(
    policy: ClarificationStudyPolicy, results: tuple[ClarificationStudyResult, ...]
) -> ClarificationStudyMetrics:
    selected = tuple(item for item in results if item.policy is policy)
    necessary = tuple(item for item in selected if item.necessary_question)
    no_question = tuple(item for item in selected if not item.necessary_question)
    actual_executable = tuple(item for item in selected if item.executable)
    gold_executable = tuple(
        item for item in selected if item.expected_outcome is IntentOutcome.EXECUTABLE
    )
    true_positive = tuple(
        item
        for item in actual_executable
        if item.expected_outcome is IntentOutcome.EXECUTABLE and item.spec_equivalent
    )
    refused = tuple(item for item in selected if item.expected_outcome is IntentOutcome.REFUSED)
    return ClarificationStudyMetrics(
        policy,
        len(selected),
        len(necessary),
        sum(item.question_count > 0 for item in necessary),
        len(no_question),
        sum(item.question_count > 0 for item in no_question),
        sum(item.silent_constraint_invention for item in selected),
        len(true_positive),
        len(actual_executable),
        len(gold_executable),
        sum(item.spec_equivalent for item in selected),
        sum(item.actual_outcome is IntentOutcome.REFUSED for item in refused),
        len(refused),
        sum(item.question_count for item in selected) / len(selected) if selected else 0.0,
    )


def _limits(value: object) -> ClarificationStudyLimits:
    record = _object(value, "study limits")
    expected = {"max_cases", "max_questions_per_case", "provider_replay_seeds"}
    if set(record) != expected:
        raise ClarificationStudyError("study limits have missing or unknown fields")
    seeds = record["provider_replay_seeds"]
    if not isinstance(seeds, list):
        raise ClarificationStudyError("provider replay seeds must be an array")
    return ClarificationStudyLimits(
        _positive_int(record["max_cases"], "max_cases"),
        _positive_int(record["max_questions_per_case"], "max_questions_per_case"),
        tuple(_non_negative_int(item, "provider replay seed") for item in seeds),
    )


def _case(value: object) -> ClarificationStudyCase:
    from modelsurgeon.search.intent_corpus import CorpusCaseClass

    record = _object(value, "study case")
    expected = {
        "case_id",
        "intent_case_id",
        "split",
        "expected_outcome",
        "necessary_question",
        "expected_question_categories",
        "classification",
        "retained",
    }
    if set(record) != expected:
        raise ClarificationStudyError("study case has missing or unknown fields")
    categories = record["expected_question_categories"]
    if not isinstance(categories, list):
        raise ClarificationStudyError("expected question categories must be an array")
    try:
        split = ClarificationStudySplit(cast(str, record["split"]))
        outcome = IntentOutcome(cast(str, record["expected_outcome"]))
        classification = CorpusCaseClass(cast(str, record["classification"]))
    except ValueError as error:
        raise ClarificationStudyError("study case enum is unknown") from error
    if not isinstance(record["necessary_question"], bool) or not isinstance(
        record["retained"], bool
    ):
        raise ClarificationStudyError("study case booleans are invalid")
    return ClarificationStudyCase(
        _text(record["case_id"], "study case ID"),
        _text(record["intent_case_id"], "intent case ID"),
        split,
        outcome,
        record["necessary_question"],
        tuple(_text(item, "question category") for item in categories),
        classification,
        record["retained"],
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ClarificationStudyError(f"{label} must be an object")
    return dict(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClarificationStudyError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    text = _text(value, label)
    valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if any(character not in valid_characters for character in text):
        raise ClarificationStudyError(f"{label} must be a canonical identifier")
    return text


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClarificationStudyError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClarificationStudyError(f"{label} must be a non-negative integer")
    return value


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ClarificationStudyError("study value is not canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "CLARIFICATION_STUDY_SCHEMA_VERSION",
    "ClarificationStudyCase",
    "ClarificationStudyCorpus",
    "ClarificationStudyError",
    "ClarificationStudyLimits",
    "ClarificationStudyMetrics",
    "ClarificationStudyPolicy",
    "ClarificationStudyResult",
    "ClarificationStudyRun",
    "ClarificationStudySplit",
    "load_and_run_clarification_study",
    "load_clarification_study",
    "run_clarification_study",
]
