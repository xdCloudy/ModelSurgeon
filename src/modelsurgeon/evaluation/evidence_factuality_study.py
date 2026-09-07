"""Bounded v2.8 evidence-grounding coverage and factuality study.

The study replays provider-independent campaign narratives against immutable
evidence snapshots.  The grounded arm uses the production snapshot/query and
claim renderer path; the template-only arm is a conservative structured
baseline; the unconstrained-text arm is retained as a deliberately unsafe
negative control.  No provider, prompt, transcript, or live model is used.

Every cell is retained, including failed and inconclusive evidence.  A
critical escape fails the shippability decision for that arm, while the study
continues collecting the bounded negative control so its failure is visible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    EvidenceCursor,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryResponse,
    EvidenceSnapshot,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.explain import ClaimEvidenceBlock, ClaimEvidenceExplanation, render_claim_evidence

EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION: Final[int] = 1
EVIDENCE_FACTUALITY_PROTOCOL_ID: Final[str] = "v28-evidence-factuality-v1"
MAX_EVIDENCE_FACTUALITY_STUDY_BYTES: Final[int] = 256_000
MAX_EVIDENCE_FACTUALITY_CASES: Final[int] = 12
MAX_EVIDENCE_FACTUALITY_CELLS: Final[int] = 128
MAX_EVIDENCE_FACTUALITY_SEEDS: Final[int] = 3

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvidenceFactualityStudyError(ValueError):
    """Raised when a study protocol or replay leaves the bounded contract."""


class EvidenceFactualitySplit(StrEnum):
    """Whether a narrative is a fixed fixture or held-out evaluation case."""

    FIXTURE = "fixture"
    HELD_OUT = "held_out"


class EvidenceFactualityMethod(StrEnum):
    """Study arms, with unconstrained text explicitly kept non-shippable."""

    GROUNDED_RENDERER = "grounded_renderer"
    TEMPLATE_ONLY = "template_only"
    UNCONSTRAINED_TEXT = "unconstrained_text"


class EvidenceFactualityClaimStatus(StrEnum):
    """Gold or observed factual status for one claim."""

    MEASURED = "measured"
    PREDICTED = "predicted"
    NEGATIVE = "negative"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


class EvidenceFactualityCellStatus(StrEnum):
    """Outcome of one retained method/narrative/seed cell."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


_NEGATIVE_CLAIM_STATUSES: Final[frozenset[EvidenceFactualityClaimStatus]] = frozenset(
    {
        EvidenceFactualityClaimStatus.NEGATIVE,
        EvidenceFactualityClaimStatus.UNSUPPORTED,
        EvidenceFactualityClaimStatus.UNKNOWN,
        EvidenceFactualityClaimStatus.INCONCLUSIVE,
    }
)


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvidenceFactualityStudyError("study value is not canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceFactualityStudyError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise EvidenceFactualityStudyError(f"{label} must be a canonical identifier")
    return result


def _digest_value(value: object, label: str) -> str:
    result = _text(value, label)
    if _DIGEST.fullmatch(result) is None:
        raise EvidenceFactualityStudyError(f"{label} must be a sha256 digest")
    return result


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceFactualityStudyError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceFactualityStudyError(f"{label} must be an array")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise EvidenceFactualityStudyError(f"{label} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceFactualityLimits:
    """Fixed local resource bounds for a replay."""

    max_cases: int = MAX_EVIDENCE_FACTUALITY_CASES
    max_cells: int = MAX_EVIDENCE_FACTUALITY_CELLS
    max_cpu_seconds: float = 10.0
    max_output_bytes: int = 512 * 1024
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if isinstance(self.max_cases, bool) or not isinstance(self.max_cases, int):
            raise EvidenceFactualityStudyError("max_cases must be an integer")
        if not 0 < self.max_cases <= MAX_EVIDENCE_FACTUALITY_CASES:
            raise EvidenceFactualityStudyError("max_cases exceeds the fixed study bound")
        if isinstance(self.max_cells, bool) or not isinstance(self.max_cells, int):
            raise EvidenceFactualityStudyError("max_cells must be an integer")
        if not 0 < self.max_cells <= MAX_EVIDENCE_FACTUALITY_CELLS:
            raise EvidenceFactualityStudyError("max_cells exceeds the fixed study bound")
        if not isinstance(self.max_cpu_seconds, (int, float)) or self.max_cpu_seconds <= 0:
            raise EvidenceFactualityStudyError("max_cpu_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise EvidenceFactualityStudyError("max_output_bytes must be positive")
        if self.seeds != tuple(sorted(set(self.seeds))):
            raise EvidenceFactualityStudyError("seeds must be sorted and unique")
        if len(self.seeds) > MAX_EVIDENCE_FACTUALITY_SEEDS or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise EvidenceFactualityStudyError("the study allows at most three non-negative seeds")

    def to_record(self) -> dict[str, object]:
        return {
            "max_cases": self.max_cases,
            "max_cells": self.max_cells,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_output_bytes": self.max_output_bytes,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityThresholds:
    """Pre-registered release thresholds for a shippable explanation arm."""

    claim_source_precision: float = 1.0
    claim_source_recall: float = 1.0
    source_id_correctness: float = 1.0
    measured_predicted_confusion_rate: float = 0.0
    negative_evidence_coverage: float = 1.0
    uncertainty_disclosure: float = 1.0
    uncertainty_calibration: float = 1.0
    unsupported_claim_rate: float = 0.0
    deterministic_ids: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.claim_source_precision,
            self.claim_source_recall,
            self.source_id_correctness,
            self.negative_evidence_coverage,
            self.uncertainty_disclosure,
            self.uncertainty_calibration,
            self.deterministic_ids,
        )
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
            raise EvidenceFactualityStudyError("rate thresholds must be between zero and one")
        if not 0 <= self.measured_predicted_confusion_rate <= 1:
            raise EvidenceFactualityStudyError("confusion threshold must be between zero and one")
        if not 0 <= self.unsupported_claim_rate <= 1:
            raise EvidenceFactualityStudyError(
                "unsupported-claim threshold must be between zero and one"
            )

    def to_record(self) -> dict[str, float]:
        return {
            "claim_source_precision": self.claim_source_precision,
            "claim_source_recall": self.claim_source_recall,
            "source_id_correctness": self.source_id_correctness,
            "measured_predicted_confusion_rate": self.measured_predicted_confusion_rate,
            "negative_evidence_coverage": self.negative_evidence_coverage,
            "uncertainty_disclosure": self.uncertainty_disclosure,
            "uncertainty_calibration": self.uncertainty_calibration,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "deterministic_ids": self.deterministic_ids,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityRecord:
    """One fixed canonical evidence row in the fixture."""

    evidence_id: str
    source_digest: str
    outcome: CampaignOutcome
    detail: str
    decision: str | None
    measured: bool
    metric: str | None
    value: float | None
    unit: str | None
    lower_bound: float | None
    upper_bound: float | None
    confidence: float | None
    standard_error: float | None
    sample_count: int | None
    observed_at: str | None
    artifact_digest: str | None
    inconclusive: bool
    provenance_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence ID")
        _digest_value(self.source_digest, "source digest")
        if not self.detail.strip():
            raise EvidenceFactualityStudyError("evidence detail is required")
        if self.metric is None and any(
            value is not None
            for value in (
                self.value,
                self.unit,
                self.lower_bound,
                self.upper_bound,
                self.confidence,
                self.standard_error,
                self.sample_count,
            )
        ):
            raise EvidenceFactualityStudyError("unmeasured evidence cannot carry measurements")
        if self.measured and self.metric is None:
            raise EvidenceFactualityStudyError("measured evidence requires a metric")
        if self.measured and self.observed_at is None:
            raise EvidenceFactualityStudyError("measured evidence requires observed_at")
        if self.artifact_digest is not None:
            _digest_value(self.artifact_digest, "artifact digest")
        if self.provenance_refs != _sorted_unique(self.provenance_refs, "provenance references"):
            raise EvidenceFactualityStudyError("provenance references must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source_digest": self.source_digest,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "decision": self.decision,
            "measured": self.measured,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence": self.confidence,
            "standard_error": self.standard_error,
            "sample_count": self.sample_count,
            "observed_at": self.observed_at,
            "artifact_digest": self.artifact_digest,
            "inconclusive": self.inconclusive,
            "provenance_refs": list(self.provenance_refs),
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityGoldClaim:
    """Expected source/status label for one narrative claim."""

    claim_id: str
    source_ids: tuple[str, ...]
    status: EvidenceFactualityClaimStatus
    uncertainty_required: bool

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "claim ID")
        if self.source_ids != _sorted_unique(self.source_ids, "claim source IDs"):
            raise EvidenceFactualityStudyError("claim source IDs must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "source_ids": list(self.source_ids),
            "status": self.status.value,
            "uncertainty_required": self.uncertainty_required,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityCase:
    """One narrative with a fixed evidence snapshot and gold labels."""

    case_id: str
    split: EvidenceFactualitySplit
    narrative: str
    evidence: tuple[EvidenceFactualityRecord, ...]
    gold_claims: tuple[EvidenceFactualityGoldClaim, ...]
    retained: bool = True

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case ID")
        _text(self.narrative, "campaign narrative")
        if not self.evidence or len(self.evidence) != len(self.gold_claims):
            raise EvidenceFactualityStudyError("each case needs one gold claim per evidence row")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        claim_ids = tuple(item.claim_id for item in self.gold_claims)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise EvidenceFactualityStudyError("evidence IDs must be sorted and unique")
        if claim_ids != tuple(sorted(set(claim_ids))):
            raise EvidenceFactualityStudyError("claim IDs must be sorted and unique")
        expected_ids = {"claim_" + item for item in evidence_ids}
        if set(claim_ids) != expected_ids:
            raise EvidenceFactualityStudyError("gold claims must bind one-to-one to evidence IDs")
        if not self.retained:
            raise EvidenceFactualityStudyError("negative and inconclusive cases must be retained")

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "split": self.split.value,
            "narrative": self.narrative,
            "evidence": [item.to_record() for item in self.evidence],
            "gold_claims": [item.to_record() for item in self.gold_claims],
            "retained": self.retained,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityCorpus:
    """Versioned protocol and fixed provider-independent study corpus."""

    corpus_id: str
    corpus_revision: str
    limits: EvidenceFactualityLimits
    thresholds: EvidenceFactualityThresholds
    cases: tuple[EvidenceFactualityCase, ...]
    protocol_id: str = EVIDENCE_FACTUALITY_PROTOCOL_ID
    schema_version: int = EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION:
            raise EvidenceFactualityStudyError("unsupported study schema version")
        _identifier(self.corpus_id, "corpus ID")
        _text(self.corpus_revision, "corpus revision")
        if not self.cases or len(self.cases) > self.limits.max_cases:
            raise EvidenceFactualityStudyError("case count exceeds the fixed study bound")
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(set(case_ids))):
            raise EvidenceFactualityStudyError("case IDs must be sorted and unique")
        if {item.split for item in self.cases} != {
            EvidenceFactualitySplit.FIXTURE,
            EvidenceFactualitySplit.HELD_OUT,
        }:
            raise EvidenceFactualityStudyError("study requires fixture and held-out narratives")
        cell_count = len(self.cases) * len(self.limits.seeds) * len(EvidenceFactualityMethod)
        if cell_count > self.limits.max_cells:
            raise EvidenceFactualityStudyError("method/case/seed cells exceed the fixed bound")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "limits": self.limits.to_record(),
            "thresholds": self.thresholds.to_record(),
            "cases": [item.to_record() for item in self.cases],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class EvidenceFactualityClaim:
    """A claim extracted from an arm's output."""

    claim_id: str
    source_ids: tuple[str, ...]
    status: EvidenceFactualityClaimStatus
    uncertainty_disclosed: bool
    text: str

    def to_record(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "source_ids": list(self.source_ids),
            "status": self.status.value,
            "uncertainty_disclosed": self.uncertainty_disclosed,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityMetrics:
    """Aggregate factuality measurements for one method."""

    method: EvidenceFactualityMethod
    case_count: int
    cell_count: int
    gold_claim_count: int
    predicted_claim_count: int
    claim_source_true_positives: int
    predicted_source_pairs: int
    gold_source_pairs: int
    exact_source_claims: int
    negative_claims: int
    negative_claims_covered: int
    uncertainty_required_claims: int
    uncertainty_disclosed_claims: int
    calibrated_uncertainty_claims: int
    measurement_prediction_confusions: int
    unsupported_claims: int
    deterministic_id_cells: int
    critical_escapes: int
    failures: int
    inconclusive_cells: int
    thresholds: EvidenceFactualityThresholds

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    @property
    def claim_source_precision(self) -> float:
        return self._rate(self.claim_source_true_positives, self.predicted_source_pairs)

    @property
    def claim_source_recall(self) -> float:
        return self._rate(self.claim_source_true_positives, self.gold_source_pairs)

    @property
    def source_id_correctness(self) -> float:
        return self._rate(self.exact_source_claims, self.predicted_claim_count)

    @property
    def measured_predicted_confusion_rate(self) -> float:
        return self._rate(self.measurement_prediction_confusions, self.gold_claim_count)

    @property
    def negative_evidence_coverage(self) -> float:
        return self._rate(self.negative_claims_covered, self.negative_claims)

    @property
    def uncertainty_disclosure(self) -> float:
        return self._rate(self.uncertainty_disclosed_claims, self.uncertainty_required_claims)

    @property
    def uncertainty_calibration(self) -> float:
        return self._rate(self.calibrated_uncertainty_claims, self.gold_claim_count)

    @property
    def unsupported_claim_rate(self) -> float:
        return self._rate(self.unsupported_claims, self.predicted_claim_count)

    @property
    def deterministic_ids(self) -> float:
        return self._rate(self.deterministic_id_cells, self.cell_count)

    @property
    def passed(self) -> bool:
        t = self.thresholds
        return (
            self.critical_escapes == 0
            and self.claim_source_precision >= t.claim_source_precision
            and self.claim_source_recall >= t.claim_source_recall
            and self.source_id_correctness >= t.source_id_correctness
            and self.measured_predicted_confusion_rate <= t.measured_predicted_confusion_rate
            and self.negative_evidence_coverage >= t.negative_evidence_coverage
            and self.uncertainty_disclosure >= t.uncertainty_disclosure
            and self.uncertainty_calibration >= t.uncertainty_calibration
            and self.unsupported_claim_rate <= t.unsupported_claim_rate
            and self.deterministic_ids >= t.deterministic_ids
        )

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "case_count": self.case_count,
            "cell_count": self.cell_count,
            "gold_claim_count": self.gold_claim_count,
            "predicted_claim_count": self.predicted_claim_count,
            "claim_source_true_positives": self.claim_source_true_positives,
            "predicted_source_pairs": self.predicted_source_pairs,
            "gold_source_pairs": self.gold_source_pairs,
            "exact_source_claims": self.exact_source_claims,
            "negative_claims": self.negative_claims,
            "negative_claims_covered": self.negative_claims_covered,
            "uncertainty_required_claims": self.uncertainty_required_claims,
            "uncertainty_disclosed_claims": self.uncertainty_disclosed_claims,
            "calibrated_uncertainty_claims": self.calibrated_uncertainty_claims,
            "measurement_prediction_confusions": self.measurement_prediction_confusions,
            "unsupported_claims": self.unsupported_claims,
            "deterministic_id_cells": self.deterministic_id_cells,
            "critical_escapes": self.critical_escapes,
            "failures": self.failures,
            "inconclusive_cells": self.inconclusive_cells,
            "claim_source_precision": self.claim_source_precision,
            "claim_source_recall": self.claim_source_recall,
            "source_id_correctness": self.source_id_correctness,
            "measured_predicted_confusion_rate": self.measured_predicted_confusion_rate,
            "negative_evidence_coverage": self.negative_evidence_coverage,
            "uncertainty_disclosure": self.uncertainty_disclosure,
            "uncertainty_calibration": self.uncertainty_calibration,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "deterministic_ids": self.deterministic_ids,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFactualityResult:
    """One retained method/case/seed observation."""

    method: EvidenceFactualityMethod
    case_id: str
    split: EvidenceFactualitySplit
    seed: int
    status: EvidenceFactualityCellStatus
    claims: tuple[EvidenceFactualityClaim, ...]
    gold_claims: tuple[EvidenceFactualityGoldClaim, ...]
    snapshot_id: str | None
    snapshot_digest: str | None
    response_digest: str | None
    explanation_digest: str | None
    critical_escape: bool
    retained: bool
    inconclusive: bool
    mismatches: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.retained:
            raise EvidenceFactualityStudyError("all study results must be retained")

    @property
    def passed(self) -> bool:
        return self.status is EvidenceFactualityCellStatus.PASS and not self.critical_escape

    @property
    def result_id(self) -> str:
        return "evidence_factuality_result_" + _digest(self.to_record(False))[len("sha256:") :]

    def to_record(self, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "method": self.method.value,
            "case_id": self.case_id,
            "split": self.split.value,
            "seed": self.seed,
            "status": self.status.value,
            "claims": [item.to_record() for item in self.claims],
            "gold_claims": [item.to_record() for item in self.gold_claims],
            "snapshot_id": self.snapshot_id,
            "snapshot_digest": self.snapshot_digest,
            "response_digest": self.response_digest,
            "explanation_digest": self.explanation_digest,
            "critical_escape": self.critical_escape,
            "retained": self.retained,
            "inconclusive": self.inconclusive,
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "error": self.error,
        }
        if include_identity:
            record["result_id"] = self.result_id
        return record


@dataclass(frozen=True, slots=True)
class EvidenceFactualityRun:
    """Complete retained replay and release-threshold decision."""

    protocol_id: str
    corpus_id: str
    corpus_revision: str
    limits: EvidenceFactualityLimits
    methods: tuple[EvidenceFactualityMethod, ...]
    results: tuple[EvidenceFactualityResult, ...]
    metrics: tuple[EvidenceFactualityMetrics, ...]
    stop_reason: str
    schema_version: int = EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.policy_passed(EvidenceFactualityMethod.GROUNDED_RENDERER)

    def policy_passed(self, method: EvidenceFactualityMethod) -> bool:
        return next(item for item in self.metrics if item.method is method).passed

    @property
    def run_id(self) -> str:
        return "evidence_factuality_run_" + _digest(self.to_record(False))[len("sha256:") :]

    def to_record(self, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "limits": self.limits.to_record(),
            "methods": [item.value for item in self.methods],
            "passed": self.passed,
            "stop_reason": self.stop_reason,
            "results": [item.to_record() for item in self.results],
            "metrics": [item.to_record() for item in self.metrics],
        }
        if include_identity:
            record["run_id"] = self.run_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


def _parse_limits(value: object) -> EvidenceFactualityLimits:
    record = _object(value, "limits")
    expected = {"max_cases", "max_cells", "max_cpu_seconds", "max_output_bytes", "seeds"}
    if set(record) != expected:
        raise EvidenceFactualityStudyError("limits have missing or unknown fields")
    raw_seeds = _array(record["seeds"], "limits.seeds")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds):
        raise EvidenceFactualityStudyError("limits.seeds must contain integers")
    return EvidenceFactualityLimits(
        cast(int, record["max_cases"]),
        cast(int, record["max_cells"]),
        cast(float, record["max_cpu_seconds"]),
        cast(int, record["max_output_bytes"]),
        tuple(cast(int, seed) for seed in raw_seeds),
    )


def _parse_thresholds(value: object) -> EvidenceFactualityThresholds:
    record = _object(value, "thresholds")
    expected = set(EvidenceFactualityThresholds().to_record())
    if set(record) != expected:
        raise EvidenceFactualityStudyError("thresholds have missing or unknown fields")
    return EvidenceFactualityThresholds(**{key: cast(float, record[key]) for key in expected})


def _parse_evidence(value: object, label: str) -> EvidenceFactualityRecord:
    record = _object(value, label)
    expected = set(EvidenceFactualityRecord.__dataclass_fields__) - {"outcome"}
    expected |= {"outcome"}
    if set(record) != expected:
        raise EvidenceFactualityStudyError(f"{label} has missing or unknown fields")
    try:
        outcome = CampaignOutcome(cast(str, record["outcome"]))
    except ValueError as error:
        raise EvidenceFactualityStudyError(f"{label}.outcome is unknown") from error
    raw_refs = _array(record["provenance_refs"], f"{label}.provenance_refs")
    if not all(isinstance(item, str) for item in raw_refs):
        raise EvidenceFactualityStudyError(f"{label}.provenance_refs must contain strings")
    return EvidenceFactualityRecord(
        _identifier(record["evidence_id"], f"{label}.evidence_id"),
        _digest_value(record["source_digest"], f"{label}.source_digest"),
        outcome,
        _text(record["detail"], f"{label}.detail"),
        None if record["decision"] is None else _text(record["decision"], f"{label}.decision"),
        cast(bool, record["measured"]),
        None if record["metric"] is None else _identifier(record["metric"], f"{label}.metric"),
        None if record["value"] is None else cast(float, record["value"]),
        None if record["unit"] is None else _text(record["unit"], f"{label}.unit"),
        None if record["lower_bound"] is None else cast(float, record["lower_bound"]),
        None if record["upper_bound"] is None else cast(float, record["upper_bound"]),
        None if record["confidence"] is None else cast(float, record["confidence"]),
        None if record["standard_error"] is None else cast(float, record["standard_error"]),
        None if record["sample_count"] is None else cast(int, record["sample_count"]),
        None
        if record["observed_at"] is None
        else _text(record["observed_at"], f"{label}.observed_at"),
        None
        if record["artifact_digest"] is None
        else _digest_value(record["artifact_digest"], f"{label}.artifact_digest"),
        cast(bool, record["inconclusive"]),
        tuple(cast(str, item) for item in raw_refs),
    )


def _parse_gold(value: object, label: str) -> EvidenceFactualityGoldClaim:
    record = _object(value, label)
    expected = {"claim_id", "source_ids", "status", "uncertainty_required"}
    if set(record) != expected:
        raise EvidenceFactualityStudyError(f"{label} has missing or unknown fields")
    raw_sources = _array(record["source_ids"], f"{label}.source_ids")
    if not all(isinstance(item, str) for item in raw_sources):
        raise EvidenceFactualityStudyError(f"{label}.source_ids must contain strings")
    try:
        status = EvidenceFactualityClaimStatus(cast(str, record["status"]))
    except ValueError as error:
        raise EvidenceFactualityStudyError(f"{label}.status is unknown") from error
    return EvidenceFactualityGoldClaim(
        _identifier(record["claim_id"], f"{label}.claim_id"),
        tuple(cast(str, item) for item in raw_sources),
        status,
        cast(bool, record["uncertainty_required"]),
    )


def load_evidence_factuality_study(path: Path) -> EvidenceFactualityCorpus:
    """Load and validate a versioned provider-independent study fixture."""

    try:
        if path.stat().st_size > MAX_EVIDENCE_FACTUALITY_STUDY_BYTES:
            raise EvidenceFactualityStudyError("study fixture exceeds its byte budget")
        root = json.loads(path.read_text(encoding="utf-8"))
    except EvidenceFactualityStudyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceFactualityStudyError("study fixture is not valid UTF-8 JSON") from error
    record = _object(root, "study fixture")
    expected = {
        "schema_version",
        "protocol_id",
        "corpus_id",
        "corpus_revision",
        "limits",
        "thresholds",
        "cases",
    }
    if set(record) != expected:
        raise EvidenceFactualityStudyError("study fixture has missing or unknown fields")
    if record["schema_version"] != EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION:
        raise EvidenceFactualityStudyError("unsupported study fixture schema")
    limits = _parse_limits(record["limits"])
    raw_cases = _array(record["cases"], "cases")
    cases: list[EvidenceFactualityCase] = []
    for index, raw in enumerate(raw_cases):
        item = _object(raw, f"cases[{index}]")
        expected_case = {"case_id", "split", "narrative", "evidence", "gold_claims", "retained"}
        if set(item) != expected_case:
            raise EvidenceFactualityStudyError(f"cases[{index}] has missing or unknown fields")
        try:
            split = EvidenceFactualitySplit(cast(str, item["split"]))
        except ValueError as error:
            raise EvidenceFactualityStudyError(f"cases[{index}].split is unknown") from error
        raw_evidence = _array(item["evidence"], f"cases[{index}].evidence")
        raw_gold = _array(item["gold_claims"], f"cases[{index}].gold_claims")
        cases.append(
            EvidenceFactualityCase(
                _identifier(item["case_id"], f"cases[{index}].case_id"),
                split,
                _text(item["narrative"], f"cases[{index}].narrative"),
                tuple(
                    _parse_evidence(value, f"cases[{index}].evidence[{i}]")
                    for i, value in enumerate(raw_evidence)
                ),
                tuple(
                    _parse_gold(value, f"cases[{index}].gold_claims[{i}]")
                    for i, value in enumerate(raw_gold)
                ),
                cast(bool, item["retained"]),
            )
        )
    return EvidenceFactualityCorpus(
        _identifier(record["corpus_id"], "corpus ID"),
        _text(record["corpus_revision"], "corpus revision"),
        limits,
        _parse_thresholds(record["thresholds"]),
        tuple(cases),
        _text(record["protocol_id"], "protocol ID"),
        cast(int, record["schema_version"]),
    )


def _campaign_snapshot(case: EvidenceFactualityCase) -> EvidenceSnapshot:
    spec_payload = {
        "constraints": [{"metric": "quality", "minimum": 0.95}],
        "study_case": case.case_id,
    }
    spec = CampaignSpec(
        "evidence_factuality_contract",
        _digest(spec_payload),
        spec_payload,
        (spec_payload["constraints"][0],),  # type: ignore[arg-type]
    )
    campaign_id = "campaign_" + case.case_id
    state = new_campaign_state(
        session_id="session_evidence_factuality",
        run_id="run_" + case.case_id,
        source_model_digest="sha256:" + "d" * 64,
        spec=spec,
        policy_state={"record_type": "evidence_factuality_fixture", "revision": "v1"},
        provider_context={"provider_id": "provider-independent-fixture"},
        budget=CampaignBudget(10.0, 64 * 1024, 16, 32 * 1024),
        campaign_id=campaign_id,
    )
    evidence: list[CampaignEvidence] = []
    for item in case.evidence:
        provenance: dict[str, object] = {
            "record_type": "evidence_factuality_fixture",
            "source": "provider-independent",
            "run_id": state.run_id,
            "plan_id": "plan_" + case.case_id,
        }
        if item.decision is not None:
            provenance["decision"] = item.decision
        if item.measured:
            uncertainty = {
                "lower_bound": item.lower_bound,
                "upper_bound": item.upper_bound,
                "confidence": item.confidence,
                "standard_error": item.standard_error,
                "sample_count": item.sample_count,
            }
            provenance["measurements"] = {
                item.metric: {
                    "value": item.value,
                    "unit": item.unit,
                    "uncertainty": uncertainty,
                }
            }
            provenance["observed_at"] = item.observed_at
        evidence.append(
            CampaignEvidence(
                item.evidence_id,
                item.source_digest,
                item.outcome,
                item.detail,
                provenance,
                item.artifact_digest,
                item.inconclusive,
            )
        )
    state = replace(
        state,
        evidence_cursor=EvidenceCursor(len(evidence), evidence[-1].evidence_id),
        state_version=len(evidence),
    )
    return EvidenceSnapshot(
        state,
        tuple(evidence),
        captured_at="2026-09-08T00:00:00Z",
    )


def _response(
    case: EvidenceFactualityCase,
) -> tuple[EvidenceSnapshot, EvidenceQueryResponse, ClaimEvidenceExplanation]:
    snapshot = _campaign_snapshot(case)
    query = EvidenceQuery(
        snapshot.campaign.campaign_id,
        fields=(
            "artifact_digest",
            "detail",
            "measurements",
            "observed_at",
            "outcome",
            "provenance_refs",
            "source_outcome",
            "uncertainty",
        ),
        expected_snapshot_id=snapshot.snapshot_id,
        expected_state_digest=snapshot.state_digest,
    )
    response = EvidenceQueryEngine(snapshot).query(query)
    return snapshot, response, render_claim_evidence(response)


def _status_from_explanation(block: ClaimEvidenceBlock) -> EvidenceFactualityClaimStatus:
    status = block.status.value
    if status == "measured":
        return EvidenceFactualityClaimStatus.MEASURED
    if status == "predicted":
        return EvidenceFactualityClaimStatus.PREDICTED
    if status == "unsupported":
        return EvidenceFactualityClaimStatus.UNSUPPORTED
    if status == "unknown":
        return EvidenceFactualityClaimStatus.UNKNOWN
    if status == "inconclusive":
        return EvidenceFactualityClaimStatus.INCONCLUSIVE
    return EvidenceFactualityClaimStatus.NEGATIVE


def _claim_id(evidence_id: str) -> str:
    return "claim_" + evidence_id


def _grounded_claims(
    case: EvidenceFactualityCase, explanation: ClaimEvidenceExplanation
) -> tuple[EvidenceFactualityClaim, ...]:
    return tuple(
        EvidenceFactualityClaim(
            _claim_id(item.evidence_id),
            (item.evidence_id,),
            _status_from_explanation(item),
            bool(item.measurements and item.measurements[0].uncertainty is not None),
            item.render_line(),
        )
        for item in explanation.claims
    )


def _template_claims(case: EvidenceFactualityCase) -> tuple[EvidenceFactualityClaim, ...]:
    claims: list[EvidenceFactualityClaim] = []
    for item, gold in zip(case.evidence, case.gold_claims, strict=True):
        claims.append(
            EvidenceFactualityClaim(
                gold.claim_id,
                (item.evidence_id,),
                gold.status,
                gold.uncertainty_required,
                f"{gold.status.value}: {item.detail} [source={item.evidence_id}]",
            )
        )
    return tuple(claims)


def _unconstrained_claims(case: EvidenceFactualityCase) -> tuple[EvidenceFactualityClaim, ...]:
    """A deterministic free-text control that intentionally overclaims."""

    return tuple(
        EvidenceFactualityClaim(
            gold.claim_id,
            (),
            EvidenceFactualityClaimStatus.MEASURED,
            False,
            f"The campaign improved quality according to the narrative: {case.narrative}",
        )
        for gold in case.gold_claims
    )


def _claims_for_method(
    method: EvidenceFactualityMethod,
    case: EvidenceFactualityCase,
    explanation: ClaimEvidenceExplanation,
) -> tuple[EvidenceFactualityClaim, ...]:
    if method is EvidenceFactualityMethod.GROUNDED_RENDERER:
        return _grounded_claims(case, explanation)
    if method is EvidenceFactualityMethod.TEMPLATE_ONLY:
        return _template_claims(case)
    return _unconstrained_claims(case)


def _observe(
    method: EvidenceFactualityMethod,
    case: EvidenceFactualityCase,
    seed: int,
) -> EvidenceFactualityResult:
    del seed  # Provider-independent replay has no stochastic branch.
    try:
        snapshot, response, explanation = _response(case)
        claims = _claims_for_method(method, case, explanation)
        gold = {item.claim_id: item for item in case.gold_claims}
        output_ids = {item.claim_id for item in claims}
        mismatches: set[str] = set()
        critical_escape = False
        unsupported = 0
        for claim in claims:
            expected = gold.get(claim.claim_id)
            if expected is None:
                mismatches.add("unknown-claim")
                unsupported += 1
                critical_escape = True
                continue
            if claim.source_ids != expected.source_ids:
                mismatches.add("source-id")
                unsupported += 1
                critical_escape = True
            if claim.status is not expected.status:
                mismatches.add("claim-status")
            if claim.status is EvidenceFactualityClaimStatus.MEASURED and (
                expected.status is not EvidenceFactualityClaimStatus.MEASURED
                or not claim.source_ids
            ):
                mismatches.add("measured-without-canonical-source")
                critical_escape = True
        if output_ids != set(gold):
            mismatches.add("claim-coverage")
        if method is EvidenceFactualityMethod.UNCONSTRAINED_TEXT:
            critical_escape = True
        status = (
            EvidenceFactualityCellStatus.PASS
            if not mismatches
            else EvidenceFactualityCellStatus.FAIL
        )
        return EvidenceFactualityResult(
            method,
            case.case_id,
            case.split,
            0,
            status,
            claims,
            case.gold_claims,
            snapshot.snapshot_id,
            snapshot.snapshot_digest,
            response.response_digest,
            explanation.explanation_digest,
            critical_escape,
            True,
            any(
                item.status is EvidenceFactualityClaimStatus.INCONCLUSIVE
                for item in case.gold_claims
            ),
            tuple(sorted(mismatches)),
            None,
        )
    except Exception as error:
        return EvidenceFactualityResult(
            method,
            case.case_id,
            case.split,
            0,
            EvidenceFactualityCellStatus.ERROR,
            (),
            case.gold_claims,
            None,
            None,
            None,
            None,
            True,
            True,
            any(
                item.status is EvidenceFactualityClaimStatus.INCONCLUSIVE
                for item in case.gold_claims
            ),
            (f"error:{type(error).__name__}",),
            str(error),
        )


def _with_seed(result: EvidenceFactualityResult, seed: int) -> EvidenceFactualityResult:
    return replace(result, seed=seed)


def _metrics(
    method: EvidenceFactualityMethod,
    results: Sequence[EvidenceFactualityResult],
    thresholds: EvidenceFactualityThresholds,
    deterministic_result_ids: set[str],
) -> EvidenceFactualityMetrics:
    selected = tuple(item for item in results if item.method is method)
    tp = predicted_pairs = gold_pairs = exact = negatives = covered = required = disclosed = 0
    calibrated = confusions = unsupported = deterministic = failures = inconclusive = 0
    for result in selected:
        gold = {item.claim_id: item for item in result.gold_claims}
        predicted = {item.claim_id: item for item in result.claims}
        predicted_pairs += sum(len(item.source_ids) for item in result.claims)
        gold_pairs += sum(len(item.source_ids) for item in result.gold_claims)
        for claim_id, expected in gold.items():
            actual = predicted.get(claim_id)
            if expected.status in _NEGATIVE_CLAIM_STATUSES:
                negatives += 1
            if actual is not None and actual.source_ids == expected.source_ids:
                exact += 1
            if actual is not None:
                tp += len(set(actual.source_ids) & set(expected.source_ids))
                confusions += actual.status is not expected.status
                disclosure_correct = actual.uncertainty_disclosed is expected.uncertainty_required
                calibrated += disclosure_correct
                if expected.uncertainty_required:
                    required += 1
                    disclosed += actual.uncertainty_disclosed
                if expected.status in _NEGATIVE_CLAIM_STATUSES and (
                    actual.source_ids == expected.source_ids and actual.status is expected.status
                ):
                    covered += 1
            else:
                confusions += 1
                calibrated += not expected.uncertainty_required
        unsupported += sum(
            claim.claim_id not in gold
            or claim.source_ids != gold[claim.claim_id].source_ids
            or (
                claim.status is EvidenceFactualityClaimStatus.MEASURED
                and gold[claim.claim_id].status is not EvidenceFactualityClaimStatus.MEASURED
            )
            for claim in result.claims
        )
        deterministic += result.result_id in deterministic_result_ids
        failures += result.status is not EvidenceFactualityCellStatus.PASS
        inconclusive += result.inconclusive
    return EvidenceFactualityMetrics(
        method,
        len({item.case_id for item in selected}),
        len(selected),
        sum(len(item.gold_claims) for item in selected),
        sum(len(item.claims) for item in selected),
        tp,
        predicted_pairs,
        gold_pairs,
        exact,
        negatives,
        covered,
        required,
        disclosed,
        calibrated,
        confusions,
        unsupported,
        deterministic,
        sum(item.critical_escape for item in selected),
        failures,
        inconclusive,
        thresholds,
    )


def run_evidence_factuality_study(
    corpus: EvidenceFactualityCorpus,
    methods: Sequence[EvidenceFactualityMethod] | None = None,
) -> EvidenceFactualityRun:
    """Replay every bounded method/case/seed cell and retain all outcomes."""

    selected = tuple(
        sorted(set(methods or tuple(EvidenceFactualityMethod)), key=lambda item: item.value)
    )
    if not selected or any(not isinstance(item, EvidenceFactualityMethod) for item in selected):
        raise EvidenceFactualityStudyError("study requires known methods")
    results: list[EvidenceFactualityResult] = []
    for method in selected:
        for case in corpus.cases:
            for seed in corpus.limits.seeds:
                results.append(_with_seed(_observe(method, case, seed), seed))
    retained = tuple(results)
    result_ids = tuple(item.result_id for item in retained)
    if len(result_ids) != len(set(result_ids)):
        raise EvidenceFactualityStudyError("study result IDs are not deterministic and unique")
    replay_result_ids = {
        _with_seed(_observe(method, case, seed), seed).result_id
        for method in selected
        for case in corpus.cases
        for seed in corpus.limits.seeds
    }
    metrics = tuple(
        _metrics(method, retained, corpus.thresholds, replay_result_ids) for method in selected
    )
    return EvidenceFactualityRun(
        corpus.protocol_id,
        corpus.corpus_id,
        corpus.corpus_revision,
        corpus.limits,
        selected,
        retained,
        metrics,
        "corpus_complete;critical_escapes_fail_closed_for_shippability",
    )


def load_and_run_evidence_factuality_study(path: Path) -> EvidenceFactualityRun:
    """Load and replay the checked-in v2.8 evidence-factuality protocol."""

    return run_evidence_factuality_study(load_evidence_factuality_study(path))


__all__ = [
    "EVIDENCE_FACTUALITY_PROTOCOL_ID",
    "EVIDENCE_FACTUALITY_STUDY_SCHEMA_VERSION",
    "EvidenceFactualityCase",
    "EvidenceFactualityClaim",
    "EvidenceFactualityClaimStatus",
    "EvidenceFactualityCorpus",
    "EvidenceFactualityGoldClaim",
    "EvidenceFactualityLimits",
    "EvidenceFactualityMethod",
    "EvidenceFactualityMetrics",
    "EvidenceFactualityRecord",
    "EvidenceFactualityResult",
    "EvidenceFactualityRun",
    "EvidenceFactualitySplit",
    "EvidenceFactualityStudyError",
    "EvidenceFactualityThresholds",
    "load_and_run_evidence_factuality_study",
    "load_evidence_factuality_study",
    "run_evidence_factuality_study",
]
