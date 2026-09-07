"""Versioned equivalence and refusal corpus for the intent compiler.

The corpus is an evidence boundary, not another compiler.  It loads canonical
``IntentRecord`` values, runs one or more already-supported compiler
implementations, and retains every case result, including failures and
non-executable outcomes.  Exact expected specs, outcomes, diagnostics and
provenance make silent contract drift visible.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from modelsurgeon.conversation import IntentOutcome, IntentProvenance, IntentRecord

from .intent_compiler import IntentCompilation, compile_intent_record
from .intent_policy import IntentPolicyDecision, evaluate_intent_policy

INTENT_CORPUS_SCHEMA_VERSION = 1
MAX_INTENT_CORPUS_BYTES = 2_000_000
MAX_CORPUS_IMPLEMENTATIONS = 16


class IntentCorpusError(ValueError):
    """Raised when a corpus is malformed or exceeds its declared bounds."""


class CorpusCaseClass(StrEnum):
    """Evidence class retained for each corpus case."""

    SUPPORTED = "supported"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"
    REFUSED = "refused"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class IntentCorpusLimits:
    """Hard limits applied before corpus execution begins."""

    max_cases: int = 256
    max_request_chars: int = 8_192
    max_fields: int = 32
    max_ambiguities: int = 16

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cases", self.max_cases),
            ("max_request_chars", self.max_request_chars),
            ("max_fields", self.max_fields),
            ("max_ambiguities", self.max_ambiguities),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IntentCorpusError(f"{name} must be a positive integer")

    def to_record(self) -> dict[str, int]:
        return {
            "max_cases": self.max_cases,
            "max_request_chars": self.max_request_chars,
            "max_fields": self.max_fields,
            "max_ambiguities": self.max_ambiguities,
        }


@dataclass(frozen=True, slots=True)
class IntentCorpusExpected:
    """Expected compiler and policy evidence for one case."""

    compiler_outcome: IntentOutcome
    policy_outcome: IntentOutcome
    compiler_spec: Mapping[str, object] | None
    policy_spec: Mapping[str, object] | None
    compiler_diagnostic_codes: tuple[str, ...]
    policy_diagnostic_codes: tuple[str, ...]
    ambiguity_categories: tuple[str, ...]
    provenance: IntentProvenance
    classification: CorpusCaseClass
    retained: bool = True

    def __post_init__(self) -> None:
        for label, values in (
            ("compiler diagnostic codes", self.compiler_diagnostic_codes),
            ("policy diagnostic codes", self.policy_diagnostic_codes),
            ("ambiguity categories", self.ambiguity_categories),
        ):
            if values != tuple(sorted(set(values))):
                raise IntentCorpusError(f"{label} must be sorted and unique")
        if not self.retained:
            raise IntentCorpusError("corpus cases must retain negative and inconclusive evidence")
        for label, spec in (
            ("compiler spec", self.compiler_spec),
            ("policy spec", self.policy_spec),
        ):
            if spec is not None:
                _canonical(spec, label)
        if self.compiler_outcome is IntentOutcome.EXECUTABLE and self.compiler_spec is None:
            raise IntentCorpusError("executable compiler expectations require a spec")
        if self.compiler_outcome is not IntentOutcome.EXECUTABLE and self.compiler_spec is not None:
            raise IntentCorpusError("non-executable compiler expectations cannot contain a spec")
        if self.policy_outcome is not IntentOutcome.EXECUTABLE and self.policy_spec is not None:
            raise IntentCorpusError("non-executable policy expectations cannot contain a spec")

    def to_record(self) -> dict[str, object]:
        return {
            "compiler_outcome": self.compiler_outcome.value,
            "policy_outcome": self.policy_outcome.value,
            "compiler_spec": None if self.compiler_spec is None else dict(self.compiler_spec),
            "policy_spec": None if self.policy_spec is None else dict(self.policy_spec),
            "compiler_diagnostic_codes": list(self.compiler_diagnostic_codes),
            "policy_diagnostic_codes": list(self.policy_diagnostic_codes),
            "ambiguity_categories": list(self.ambiguity_categories),
            "provenance": self.provenance.to_record(),
            "classification": self.classification.value,
            "retained": self.retained,
        }


@dataclass(frozen=True, slots=True)
class IntentCorpusCase:
    """One canonical intent and its expected evidence."""

    case_id: str
    equivalence_group: str | None
    intent: IntentRecord
    expected: IntentCorpusExpected

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case ID")
        if self.equivalence_group is not None:
            _identifier(self.equivalence_group, "equivalence group")
        if self.expected.provenance.to_record() != self.intent.provenance.to_record():
            raise IntentCorpusError(f"{self.case_id} expected provenance does not match intent")


@dataclass(frozen=True, slots=True)
class IntentCorpus:
    """Loaded, bounded corpus ready for one or more compiler implementations."""

    corpus_id: str
    corpus_revision: str
    limits: IntentCorpusLimits
    cases: tuple[IntentCorpusCase, ...]
    schema_version: int = INTENT_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTENT_CORPUS_SCHEMA_VERSION:
            raise IntentCorpusError("unsupported intent corpus schema version")
        _identifier(self.corpus_id, "corpus ID")
        if not self.corpus_revision.strip():
            raise IntentCorpusError("corpus revision is required")
        if not self.cases or len(self.cases) > self.limits.max_cases:
            raise IntentCorpusError("corpus case count exceeds its declared bound")
        ids = tuple(item.case_id for item in self.cases)
        if ids != tuple(sorted(set(ids))):
            raise IntentCorpusError("corpus case IDs must be sorted and unique")
        for case in self.cases:
            if len(case.intent.original_request) > self.limits.max_request_chars:
                raise IntentCorpusError(f"{case.case_id} exceeds the request-size budget")
            if len(case.intent.fields) > self.limits.max_fields:
                raise IntentCorpusError(f"{case.case_id} exceeds the field-count budget")
            if len(case.intent.ambiguities) > self.limits.max_ambiguities:
                raise IntentCorpusError(f"{case.case_id} exceeds the ambiguity-count budget")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "limits": self.limits.to_record(),
            "cases": [
                {
                    "case_id": case.case_id,
                    "equivalence_group": case.equivalence_group,
                    "intent": case.intent.to_record(),
                    "expected": case.expected.to_record(),
                }
                for case in self.cases
            ],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record(), "corpus")


@dataclass(frozen=True, slots=True)
class IntentCorpusCaseResult:
    """Retained result for one implementation/case pair."""

    implementation: str
    case_id: str
    equivalence_group: str | None
    intent_id: str
    compiler_outcome: IntentOutcome | None
    policy_outcome: IntentOutcome | None
    compiler_diagnostic_codes: tuple[str, ...]
    policy_diagnostic_codes: tuple[str, ...]
    compiler_spec: Mapping[str, object] | None
    policy_spec: Mapping[str, object] | None
    retained: bool
    mismatches: tuple[str, ...] = ()
    error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.error is None

    def to_record(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "case_id": self.case_id,
            "equivalence_group": self.equivalence_group,
            "intent_id": self.intent_id,
            "compiler_outcome": (
                None if self.compiler_outcome is None else self.compiler_outcome.value
            ),
            "policy_outcome": None if self.policy_outcome is None else self.policy_outcome.value,
            "compiler_diagnostic_codes": list(self.compiler_diagnostic_codes),
            "policy_diagnostic_codes": list(self.policy_diagnostic_codes),
            "compiler_spec": None if self.compiler_spec is None else dict(self.compiler_spec),
            "policy_spec": None if self.policy_spec is None else dict(self.policy_spec),
            "retained": self.retained,
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class IntentCorpusRun:
    """Complete retained run, including all negative and failed results."""

    corpus_id: str
    corpus_revision: str
    implementations: tuple[str, ...]
    results: tuple[IntentCorpusCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": INTENT_CORPUS_SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "implementations": list(self.implementations),
            "passed": self.passed,
            "results": [item.to_record() for item in self.results],
        }


CompilerImplementation = Callable[[IntentRecord], IntentCompilation]


def load_intent_corpus(path: Path) -> IntentCorpus:
    """Load and strictly validate a bounded JSON corpus."""

    try:
        if path.stat().st_size > MAX_INTENT_CORPUS_BYTES:
            raise IntentCorpusError("intent corpus exceeds the byte budget")
        root = json.loads(path.read_text(encoding="utf-8"))
    except IntentCorpusError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntentCorpusError("intent corpus is not valid UTF-8 JSON") from error
    return _corpus(root)


def run_intent_corpus(
    corpus: IntentCorpus,
    implementations: Mapping[str, CompilerImplementation] | None = None,
) -> IntentCorpusRun:
    """Run every corpus case against every supplied compiler implementation.

    The default is the repository compiler.  Provider-backed or experimental
    compilers can be passed explicitly once they implement this same typed
    boundary; no provider is silently substituted when one is unavailable.
    """

    selected: Mapping[str, CompilerImplementation] = (
        {"modelsurgeon": compile_intent_record} if implementations is None else implementations
    )
    if not selected or len(selected) > MAX_CORPUS_IMPLEMENTATIONS:
        raise IntentCorpusError("compiler implementation count exceeds the corpus bound")
    names = tuple(sorted(selected))
    for name in names:
        _identifier(name, "compiler implementation name")
        if not callable(selected[name]):
            raise IntentCorpusError(f"compiler implementation {name!r} is not callable")

    results = tuple(
        _run_case(name, selected[name], case)
        for name in names
        for case in corpus.cases
    )
    return IntentCorpusRun(corpus.corpus_id, corpus.corpus_revision, names, results)


def _run_case(
    name: str, implementation: CompilerImplementation, case: IntentCorpusCase
) -> IntentCorpusCaseResult:
    try:
        compilation = implementation(case.intent)
        if not isinstance(compilation, IntentCompilation):
            raise IntentCorpusError("compiler returned an invalid compilation result")
        decision = evaluate_intent_policy(case.intent, compilation)
        mismatches = _mismatches(case, compilation, decision)
        return IntentCorpusCaseResult(
            name,
            case.case_id,
            case.equivalence_group,
            case.intent.intent_id,
            compilation.outcome,
            decision.outcome,
            tuple(item.code for item in compilation.diagnostics),
            tuple(item.code for item in decision.diagnostics),
            compilation.spec_record,
            None if decision.contract is None else decision.contract.to_record(),
            True,
            tuple(sorted(mismatches)),
        )
    except Exception as error:  # retained as visible corpus evidence
        return IntentCorpusCaseResult(
            name,
            case.case_id,
            case.equivalence_group,
            case.intent.intent_id,
            None,
            None,
            (),
            (),
            None,
            None,
            True,
            ("implementation-error",),
            f"{type(error).__name__}: {error}",
        )


def _mismatches(
    case: IntentCorpusCase, compilation: IntentCompilation, decision: IntentPolicyDecision
) -> list[str]:
    expected = case.expected
    mismatches: list[str] = []
    if compilation.outcome is not expected.compiler_outcome:
        mismatches.append("compiler-outcome")
    if decision.outcome is not expected.policy_outcome:
        mismatches.append("policy-outcome")
    if compilation.spec_record != expected.compiler_spec:
        mismatches.append("compiler-spec")
    actual_policy_spec = None if decision.contract is None else decision.contract.to_record()
    if actual_policy_spec != expected.policy_spec:
        mismatches.append("policy-spec")
    if tuple(item.code for item in compilation.diagnostics) != expected.compiler_diagnostic_codes:
        mismatches.append("compiler-diagnostics")
    if tuple(item.code for item in decision.diagnostics) != expected.policy_diagnostic_codes:
        mismatches.append("policy-diagnostics")
    categories = tuple(item.category.value for item in decision.ambiguities)
    if categories != expected.ambiguity_categories:
        mismatches.append("ambiguity-categories")
    if decision.provenance.to_record() != expected.provenance.to_record():
        mismatches.append("provenance")
    if not expected.retained:
        mismatches.append("evidence-not-retained")
    return mismatches


def _corpus(value: object) -> IntentCorpus:
    root = _object(value, "intent corpus")
    expected = {"schema_version", "corpus_id", "corpus_revision", "limits", "cases"}
    if set(root) != expected:
        raise IntentCorpusError("intent corpus has missing or unknown fields")
    if root["schema_version"] != INTENT_CORPUS_SCHEMA_VERSION:
        raise IntentCorpusError("unsupported intent corpus schema version")
    limits = _limits(root["limits"])
    raw_cases = root["cases"]
    if not isinstance(raw_cases, list):
        raise IntentCorpusError("intent corpus cases must be an array")
    cases = tuple(_case(item) for item in raw_cases)
    corpus_id = _text(root["corpus_id"], "corpus ID")
    revision = _text(root["corpus_revision"], "corpus revision")
    return IntentCorpus(corpus_id, revision, limits, cases)


def _case(value: object) -> IntentCorpusCase:
    record = _object(value, "corpus case")
    expected_keys = {"case_id", "equivalence_group", "intent", "expected"}
    if set(record) != expected_keys:
        raise IntentCorpusError("corpus case has missing or unknown fields")
    intent_record = _object(record["intent"], "corpus intent")
    try:
        intent = IntentRecord.from_json(_canonical(intent_record, "corpus intent"))
    except (TypeError, ValueError) as error:
        raise IntentCorpusError("corpus intent is not a valid canonical intent record") from error
    group = record["equivalence_group"]
    if group is not None and not isinstance(group, str):
        raise IntentCorpusError("equivalence group must be a string or null")
    return IntentCorpusCase(
        _text(record["case_id"], "case ID"),
        group,
        intent,
        _expected(record["expected"]),
    )


def _expected(value: object) -> IntentCorpusExpected:
    record = _object(value, "corpus expected result")
    expected_keys = {
        "compiler_outcome",
        "policy_outcome",
        "compiler_spec",
        "policy_spec",
        "compiler_diagnostic_codes",
        "policy_diagnostic_codes",
        "ambiguity_categories",
        "provenance",
        "classification",
        "retained",
    }
    if set(record) != expected_keys:
        raise IntentCorpusError("corpus expected result has missing or unknown fields")
    compiler_spec = _optional_object(record["compiler_spec"], "compiler spec")
    policy_spec = _optional_object(record["policy_spec"], "policy spec")
    return IntentCorpusExpected(
        _outcome(record["compiler_outcome"], "compiler outcome"),
        _outcome(record["policy_outcome"], "policy outcome"),
        compiler_spec,
        policy_spec,
        _sorted_strings(record["compiler_diagnostic_codes"], "compiler diagnostic codes"),
        _sorted_strings(record["policy_diagnostic_codes"], "policy diagnostic codes"),
        _sorted_strings(record["ambiguity_categories"], "ambiguity categories"),
        _provenance(record["provenance"]),
        _classification(record["classification"]),
        _bool(record["retained"], "retained"),
    )


def _limits(value: object) -> IntentCorpusLimits:
    record = _object(value, "corpus limits")
    expected = {"max_cases", "max_request_chars", "max_fields", "max_ambiguities"}
    if set(record) != expected:
        raise IntentCorpusError("corpus limits have missing or unknown fields")
    return IntentCorpusLimits(
        _positive_int(record["max_cases"], "max_cases"),
        _positive_int(record["max_request_chars"], "max_request_chars"),
        _positive_int(record["max_fields"], "max_fields"),
        _positive_int(record["max_ambiguities"], "max_ambiguities"),
    )


def _provenance(value: object) -> IntentProvenance:
    record = _object(value, "expected provenance")
    expected = {
        "source_revision",
        "provider_id",
        "provider_revision",
        "tool_revision",
        "evidence_refs",
    }
    if set(record) != expected:
        raise IntentCorpusError("expected provenance has missing or unknown fields")
    return IntentProvenance(
        _text(record["source_revision"], "source revision"),
        _text(record["provider_id"], "provider ID"),
        _text(record["provider_revision"], "provider revision"),
        _text(record["tool_revision"], "tool revision"),
        _sorted_strings(record["evidence_refs"], "evidence references"),
    )


def _classification(value: object) -> CorpusCaseClass:
    try:
        return CorpusCaseClass(_text(value, "classification"))
    except ValueError as error:
        raise IntentCorpusError("unknown corpus case classification") from error


def _outcome(value: object, label: str) -> IntentOutcome:
    try:
        return IntentOutcome(_text(value, label))
    except ValueError as error:
        raise IntentCorpusError(f"unknown {label}") from error


def _optional_object(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _object(value, label)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise IntentCorpusError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _sorted_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise IntentCorpusError(f"{label} must be an array of strings")
    result = tuple(cast(str, item) for item in value)
    if result != tuple(sorted(set(result))):
        raise IntentCorpusError(f"{label} must be sorted and unique")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntentCorpusError(f"{label} must be a non-empty string")
    return value


def _identifier(value: str, label: str) -> None:
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for char in value):
        raise IntentCorpusError(f"{label} is not a canonical identifier")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntentCorpusError(f"{label} must be a positive integer")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntentCorpusError(f"{label} must be boolean")
    return value


def _canonical(value: object, label: str) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise IntentCorpusError(f"{label} is not canonical JSON") from error


__all__ = [
    "INTENT_CORPUS_SCHEMA_VERSION",
    "MAX_INTENT_CORPUS_BYTES",
    "CompilerImplementation",
    "CorpusCaseClass",
    "IntentCorpus",
    "IntentCorpusCase",
    "IntentCorpusCaseResult",
    "IntentCorpusError",
    "IntentCorpusExpected",
    "IntentCorpusLimits",
    "IntentCorpusRun",
    "load_intent_corpus",
    "run_intent_corpus",
]
