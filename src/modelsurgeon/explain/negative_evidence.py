"""Evidence-grounded explanations for negative and incomplete experiments.

This module is a read-only projection over the canonical evidence query
boundary.  It never evaluates a mutation, infers a missing measurement, or
turns a rollback into an acceptance.  Canonical mutation, evaluation, and
rollback records are copied only from their allowlisted evidence fields and
remain linked to the source evidence IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from modelsurgeon.conversation.evidence_query import (
    EvidenceMeasurement,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryOutcome,
    EvidenceQueryResponse,
    EvidenceQueryStatus,
    EvidenceSnapshot,
    EvidenceUncertainty,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.policy import (
    PolicyCandidate,
    PolicyDecision,
    PolicyOutcome,
    PolicySource,
    resolve_policy,
)

NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION = 1

type JSONValue = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None


class NegativeEvidenceExplanationError(ValueError):
    """Raised when canonical evidence cannot support a safe explanation."""


class ExplanationCompleteness(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NegativeEvidenceExplanationError("explanation contains non-canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegativeEvidenceExplanationError(f"{label} must be non-empty text")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NegativeEvidenceExplanationError(f"{label} must be finite")
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NegativeEvidenceExplanationError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise NegativeEvidenceExplanationError(f"{label} must be a boolean")
    return value


def _mapping(value: object, label: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise NegativeEvidenceExplanationError(f"{label} must be an object")
    try:
        decoded = json.loads(_canonical(value))
    except json.JSONDecodeError as error:
        raise NegativeEvidenceExplanationError(f"{label} must be canonical JSON") from error
    if not isinstance(decoded, dict):
        raise NegativeEvidenceExplanationError(f"{label} must be an object")
    return cast(dict[str, JSONValue], decoded)


def _sorted_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))) or any(not item.strip() for item in result):
        raise NegativeEvidenceExplanationError(f"{label} must be sorted, unique, and non-empty")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise NegativeEvidenceExplanationError(f"{label} must be a string array")
    return tuple(cast(list[str], value))


@dataclass(frozen=True, slots=True)
class ExplanationMetric:
    """One measured metric with its decision direction and uncertainty."""

    name: str
    value: float | None
    unit: str | None
    direction: str | None
    threshold: float | None
    comparison: str | None
    uncertainty: EvidenceUncertainty | None
    measured: bool
    reason: str | None
    evidence_id: str

    def __post_init__(self) -> None:
        _text(self.name, "explanation metric name")
        if self.value is not None:
            _finite(self.value, "explanation metric value")
        if self.unit is not None:
            _text(self.unit, "explanation metric unit")
        if self.direction is not None:
            _text(self.direction, "explanation metric direction")
        if self.threshold is not None:
            _finite(self.threshold, "explanation metric threshold")
        if self.comparison is not None:
            _text(self.comparison, "explanation metric comparison")
        if not isinstance(self.measured, bool):
            raise NegativeEvidenceExplanationError("explanation metric measured flag is invalid")
        if self.value is None and self.measured:
            raise NegativeEvidenceExplanationError("measured explanation metrics require a value")
        if self.value is not None and not self.measured:
            raise NegativeEvidenceExplanationError(
                "unmeasured explanation metrics cannot carry a value"
            )
        if self.reason is not None:
            _text(self.reason, "explanation metric reason")
        _text(self.evidence_id, "explanation metric evidence ID")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "direction": self.direction,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_record(),
            "measured": self.measured,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_record(cls, value: object) -> ExplanationMetric:
        record = _mapping(value, "explanation metric")
        expected = {
            "name",
            "value",
            "unit",
            "direction",
            "threshold",
            "comparison",
            "uncertainty",
            "measured",
            "reason",
            "evidence_id",
        }
        if set(record) != expected:
            raise NegativeEvidenceExplanationError(
                "explanation metric has missing or unknown fields"
            )
        raw_uncertainty = record["uncertainty"]
        uncertainty = None
        if raw_uncertainty is not None:
            uncertainty_record = _mapping(raw_uncertainty, "metric uncertainty")
            uncertainty = EvidenceUncertainty(
                None
                if uncertainty_record["lower_bound"] is None
                else _finite(uncertainty_record["lower_bound"], "uncertainty lower bound"),
                None
                if uncertainty_record["upper_bound"] is None
                else _finite(uncertainty_record["upper_bound"], "uncertainty upper bound"),
                None
                if uncertainty_record["confidence"] is None
                else _finite(uncertainty_record["confidence"], "uncertainty confidence"),
                None
                if uncertainty_record["standard_error"] is None
                else _finite(uncertainty_record["standard_error"], "uncertainty standard error"),
                None
                if uncertainty_record["sample_count"] is None
                else _integer(uncertainty_record["sample_count"], "uncertainty sample count"),
            )
        return cls(
            _text(record["name"], "metric name"),
            None if record["value"] is None else _finite(record["value"], "metric value"),
            None if record["unit"] is None else _text(record["unit"], "metric unit"),
            None if record["direction"] is None else _text(record["direction"], "metric direction"),
            None
            if record["threshold"] is None
            else _finite(record["threshold"], "metric threshold"),
            None
            if record["comparison"] is None
            else _text(record["comparison"], "metric comparison"),
            uncertainty,
            _boolean(record["measured"], "metric measured flag"),
            None if record["reason"] is None else _text(record["reason"], "metric reason"),
            _text(record["evidence_id"], "metric evidence ID"),
        )


@dataclass(frozen=True, slots=True)
class ExplanationLineage:
    """Canonical IDs tying an explanation back to mutation and lifecycle records."""

    evidence_id: str
    source_digest: str
    mutation_id: str | None
    evaluation_id: str | None
    rollback_id: str | None
    parent_ids: tuple[str, ...]
    artifact_digests: tuple[str, ...]
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.evidence_id, "lineage evidence ID")
        _text(self.source_digest, "lineage source digest")
        for value, label in (
            (self.mutation_id, "lineage mutation ID"),
            (self.evaluation_id, "lineage evaluation ID"),
            (self.rollback_id, "lineage rollback ID"),
        ):
            if value is not None:
                _text(value, label)
        object.__setattr__(
            self, "parent_ids", _sorted_unique(self.parent_ids, "lineage parent IDs")
        )
        object.__setattr__(
            self,
            "artifact_digests",
            _sorted_unique(self.artifact_digests, "lineage artifact digests"),
        )
        object.__setattr__(
            self, "references", _sorted_unique(self.references, "lineage references")
        )

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "evidence_id": self.evidence_id,
            "source_digest": self.source_digest,
            "mutation_id": self.mutation_id,
            "evaluation_id": self.evaluation_id,
            "rollback_id": self.rollback_id,
            "parent_ids": list(self.parent_ids),
            "artifact_digests": list(self.artifact_digests),
            "references": list(self.references),
        }

    @classmethod
    def from_record(cls, value: object) -> ExplanationLineage:
        record = _mapping(value, "explanation lineage")
        expected = {
            "evidence_id",
            "source_digest",
            "mutation_id",
            "evaluation_id",
            "rollback_id",
            "parent_ids",
            "artifact_digests",
            "references",
        }
        if set(record) != expected:
            raise NegativeEvidenceExplanationError(
                "explanation lineage has missing or unknown fields"
            )
        return cls(
            _text(record["evidence_id"], "lineage evidence ID"),
            _text(record["source_digest"], "lineage source digest"),
            None
            if record["mutation_id"] is None
            else _text(record["mutation_id"], "lineage mutation ID"),
            None
            if record["evaluation_id"] is None
            else _text(record["evaluation_id"], "lineage evaluation ID"),
            None
            if record["rollback_id"] is None
            else _text(record["rollback_id"], "lineage rollback ID"),
            _string_tuple(record["parent_ids"], "lineage parent IDs"),
            _string_tuple(record["artifact_digests"], "lineage artifact digests"),
            _string_tuple(record["references"], "lineage references"),
        )


@dataclass(frozen=True, slots=True)
class NegativeEvidenceExplanation:
    """One auditable explanation, including explicit incomplete evidence."""

    evidence_id: str
    campaign_id: str
    outcome: EvidenceQueryOutcome
    source_outcome: str
    summary: str
    why_not_qualified: str
    metrics: tuple[ExplanationMetric, ...]
    lineage: ExplanationLineage
    canonical_records: Mapping[str, JSONValue]
    unknown_fields: tuple[str, ...]
    completeness: ExplanationCompleteness
    schema_version: int = NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION
    explanation_id: str = ""

    def __post_init__(self) -> None:
        _text(self.evidence_id, "explanation evidence ID")
        _text(self.campaign_id, "explanation campaign ID")
        if not isinstance(self.outcome, EvidenceQueryOutcome):
            raise NegativeEvidenceExplanationError("explanation outcome is invalid")
        _text(self.source_outcome, "explanation source outcome")
        _text(self.summary, "explanation summary")
        _text(self.why_not_qualified, "explanation qualification reason")
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise NegativeEvidenceExplanationError("explanation metrics must be sorted and unique")
        object.__setattr__(
            self, "unknown_fields", _sorted_unique(self.unknown_fields, "unknown fields")
        )
        if self.completeness is not ExplanationCompleteness(
            "incomplete" if self.unknown_fields else "complete"
        ):
            raise NegativeEvidenceExplanationError(
                "explanation completeness disagrees with unknown fields"
            )
        records = _mapping(self.canonical_records, "explanation canonical records")
        object.__setattr__(self, "canonical_records", records)
        if self.schema_version != NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION:
            raise NegativeEvidenceExplanationError("unsupported explanation schema version")
        payload = self.to_record(include_id=False)
        expected_id = "negative_explanation_" + _digest(payload)[len("sha256:") :]
        if self.explanation_id and self.explanation_id != expected_id:
            raise NegativeEvidenceExplanationError(
                "explanation ID does not match canonical content"
            )
        object.__setattr__(self, "explanation_id", expected_id)

    def to_record(self, *, include_id: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "campaign_id": self.campaign_id,
            "outcome": self.outcome.value,
            "source_outcome": self.source_outcome,
            "summary": self.summary,
            "why_not_qualified": self.why_not_qualified,
            "metrics": [item.to_record() for item in self.metrics],
            "lineage": self.lineage.to_record(),
            "canonical_records": dict(self.canonical_records),
            "unknown_fields": list(self.unknown_fields),
            "completeness": self.completeness.value,
        }
        if include_id:
            record["explanation_id"] = self.explanation_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> NegativeEvidenceExplanation:
        record = _mapping(value, "negative evidence explanation")
        expected = {
            "schema_version",
            "explanation_id",
            "evidence_id",
            "campaign_id",
            "outcome",
            "source_outcome",
            "summary",
            "why_not_qualified",
            "metrics",
            "lineage",
            "canonical_records",
            "unknown_fields",
            "completeness",
        }
        if set(record) != expected:
            raise NegativeEvidenceExplanationError(
                "negative evidence explanation has missing or unknown fields"
            )
        try:
            outcome = EvidenceQueryOutcome(_text(record["outcome"], "explanation outcome"))
            completeness = ExplanationCompleteness(
                _text(record["completeness"], "explanation completeness")
            )
        except ValueError as error:
            raise NegativeEvidenceExplanationError(
                "negative evidence explanation has an unknown enum value"
            ) from error
        metrics = record["metrics"]
        if not isinstance(metrics, list):
            raise NegativeEvidenceExplanationError("explanation metrics must be an array")
        return cls(
            _text(record["evidence_id"], "explanation evidence ID"),
            _text(record["campaign_id"], "explanation campaign ID"),
            outcome,
            _text(record["source_outcome"], "explanation source outcome"),
            _text(record["summary"], "explanation summary"),
            _text(record["why_not_qualified"], "explanation qualification reason"),
            tuple(ExplanationMetric.from_record(item) for item in metrics),
            ExplanationLineage.from_record(record["lineage"]),
            _mapping(record["canonical_records"], "explanation canonical records"),
            _string_tuple(record["unknown_fields"], "explanation unknown fields"),
            completeness,
            _integer(record["schema_version"], "explanation schema version"),
            _text(record["explanation_id"], "explanation ID"),
        )


@dataclass(frozen=True, slots=True)
class NegativeEvidenceReport:
    """Replayable report containing one explanation for every selected row."""

    query_id: str
    snapshot_id: str
    response_digest: str
    status: EvidenceQueryStatus
    explanations: tuple[NegativeEvidenceExplanation, ...]
    missing_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    report_id: str = ""
    policy_decision: PolicyDecision | None = None

    def __post_init__(self) -> None:
        _text(self.query_id, "negative evidence query ID")
        _text(self.snapshot_id, "negative evidence snapshot ID")
        _text(self.response_digest, "negative evidence response digest")
        if not isinstance(self.status, EvidenceQueryStatus):
            raise NegativeEvidenceExplanationError("negative evidence report status is invalid")
        object.__setattr__(
            self, "missing_fields", _sorted_unique(self.missing_fields, "report missing fields")
        )
        object.__setattr__(
            self,
            "unavailable_fields",
            _sorted_unique(self.unavailable_fields, "report unavailable fields"),
        )
        payload = self.to_record(include_id=False)
        expected_id = "negative_report_" + _digest(payload)[len("sha256:") :]
        if self.report_id and self.report_id != expected_id:
            raise NegativeEvidenceExplanationError(
                "negative evidence report ID is not deterministic"
            )
        object.__setattr__(self, "report_id", expected_id)

    def to_record(self, *, include_id: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "record_type": "negative_evidence_explanation_report",
            "schema_version": NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION,
            "query_id": self.query_id,
            "snapshot_id": self.snapshot_id,
            "response_digest": self.response_digest,
            "status": self.status.value,
            "explanations": [item.to_record() for item in self.explanations],
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
        }
        if include_id:
            record["report_id"] = self.report_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> NegativeEvidenceReport:
        record = _mapping(value, "negative evidence report")
        expected = {
            "record_type",
            "schema_version",
            "report_id",
            "query_id",
            "snapshot_id",
            "response_digest",
            "status",
            "explanations",
            "missing_fields",
            "unavailable_fields",
        }
        if (
            set(record) != expected
            or record["record_type"] != "negative_evidence_explanation_report"
            or record["schema_version"] != NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION
        ):
            raise NegativeEvidenceExplanationError(
                "negative evidence report has missing or unknown fields"
            )
        try:
            status = EvidenceQueryStatus(_text(record["status"], "report status"))
        except ValueError as error:
            raise NegativeEvidenceExplanationError("report status is unknown") from error
        explanations = record["explanations"]
        if not isinstance(explanations, list):
            raise NegativeEvidenceExplanationError("report explanations must be an array")
        return cls(
            _text(record["query_id"], "report query ID"),
            _text(record["snapshot_id"], "report snapshot ID"),
            _text(record["response_digest"], "report response digest"),
            status,
            tuple(NegativeEvidenceExplanation.from_record(item) for item in explanations),
            _string_tuple(record["missing_fields"], "report missing fields"),
            _string_tuple(record["unavailable_fields"], "report unavailable fields"),
            _text(record["report_id"], "report ID"),
        )

    def to_text(self) -> str:
        lines: list[str] = []
        for item in self.explanations:
            lines.append(f"{item.evidence_id}: {item.outcome.value} — {item.summary}")
            lines.append(f"  Why it did not qualify: {item.why_not_qualified}")
            if item.metrics:
                for metric in item.metrics:
                    value = "unknown" if metric.value is None else format(metric.value, ".12g")
                    unit = "" if metric.unit is None else f" {metric.unit}"
                    lines.append(f"  {metric.name}: {value}{unit}")
            if item.unknown_fields:
                lines.append(f"  Unknown: {', '.join(item.unknown_fields)}")
        return "\n".join(lines) + ("\n" if lines else "")


_REQUIRED_QUERY_FIELDS = (
    "artifact_digest",
    "detail",
    "evaluation_record",
    "lineage",
    "measurements",
    "mutation_record",
    "outcome",
    "provenance_refs",
    "rollback_record",
    "source_digest",
    "source_outcome",
)


def _record_mapping(record: object) -> dict[str, JSONValue] | None:
    return None if not isinstance(record, Mapping) else _mapping(record, "canonical record")


def _id(record: Mapping[str, JSONValue] | None, *keys: str) -> str | None:
    if record is None:
        return None
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _record_list(
    record: Mapping[str, JSONValue] | None, key: str
) -> tuple[dict[str, JSONValue], ...]:
    if record is None or not isinstance(record.get(key), list):
        return ()
    return tuple(
        _mapping(item, f"canonical record {key} item")
        for item in cast(list[object], record[key])
        if isinstance(item, Mapping)
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _evaluation_results(record: Mapping[str, JSONValue] | None) -> tuple[dict[str, JSONValue], ...]:
    if record is None:
        return ()
    nested = record.get("constraint_evaluation")
    if isinstance(nested, Mapping):
        record = _mapping(nested, "constraint evaluation")
    return _record_list(record, "results")


def _metric_from_measurement(
    measurement: EvidenceMeasurement,
    evidence_id: str,
    evaluation: Mapping[str, JSONValue] | None,
) -> ExplanationMetric:
    match: Mapping[str, JSONValue] | None = None
    match_result: Mapping[str, JSONValue] | None = None
    for result in _evaluation_results(evaluation):
        constraint = result.get("constraint")
        if isinstance(constraint, Mapping) and constraint.get("metric") == measurement.metric:
            match = constraint
            match_result = result
            break
    return ExplanationMetric(
        measurement.metric,
        measurement.value,
        measurement.unit
        if measurement.unit is not None
        else cast(str | None, None if match is None else match.get("unit")),
        None if match is None else cast(str | None, match.get("comparison")),
        None
        if match is None
        else (
            None
            if match.get("threshold") is None
            else _finite(match.get("threshold"), "constraint threshold")
        ),
        None if match is None else cast(str | None, match.get("comparison")),
        measurement.uncertainty,
        True,
        None
        if match_result is None or match_result.get("reason") is None
        else cast(str, match_result.get("reason")),
        evidence_id,
    )


def _evaluation_metrics(
    evaluation: Mapping[str, JSONValue] | None,
    evidence_id: str,
    existing: set[str],
) -> tuple[ExplanationMetric, ...]:
    metrics: list[ExplanationMetric] = []
    for result in _evaluation_results(evaluation):
        constraint = result.get("constraint")
        if not isinstance(constraint, Mapping):
            continue
        metric = constraint.get("metric")
        if not isinstance(metric, str) or metric in existing:
            continue
        observed = result.get("observed")
        passed = result.get("passed")
        reason = result.get("reason")
        measured = isinstance(observed, (int, float)) and not isinstance(observed, bool)
        metrics.append(
            ExplanationMetric(
                metric,
                _finite(observed, f"observed {metric}") if measured else None,
                cast(str | None, constraint.get("unit")),
                cast(str | None, constraint.get("comparison")),
                None
                if constraint.get("threshold") is None
                else _finite(constraint.get("threshold"), f"threshold {metric}"),
                cast(str | None, constraint.get("comparison")),
                None,
                measured,
                None if reason is None else cast(str, reason),
                evidence_id,
            )
        )
        existing.add(metric)
        if passed is False and reason is None:
            metrics[-1] = replace(metrics[-1], reason="constraint_not_passed")
    return tuple(metrics)


def _lineage(record: object, evidence_id: str, source_digest: str) -> ExplanationLineage:
    fields = _mapping(record, "query evidence record")
    mutation = _record_mapping(fields.get("mutation_record"))
    evaluation = _record_mapping(fields.get("evaluation_record"))
    rollback = _record_mapping(fields.get("rollback_record"))
    references = tuple(
        sorted(
            set(
                str(item)
                for key in ("provenance_refs", "lineage")
                for item in _string_list(fields.get(key))
            )
        )
    )
    parent_ids = tuple(
        sorted(
            {
                value
                for source in (mutation, evaluation, rollback)
                if source is not None
                for key in (
                    "parent_id",
                    "parent_checkpoint_id",
                    "parent_evidence_id",
                    "parent_candidate_id",
                )
                for value in [source.get(key)]
                if isinstance(value, str)
            }
        )
    )
    artifacts = tuple(
        sorted(
            {
                value
                for source in (mutation, evaluation, rollback)
                if source is not None
                for key in ("artifact_digest", "source_artifact_digest")
                for value in [source.get(key)]
                if isinstance(value, str)
            }
        )
    )
    rollback_id = _id(rollback, "rollback_id", "decision_id", "candidate_id")
    return ExplanationLineage(
        evidence_id,
        source_digest,
        _id(mutation, "mutation_id"),
        _id(evaluation, "evaluation_id"),
        rollback_id,
        parent_ids,
        artifacts,
        references,
    )


def _why_not_qualified(
    outcome: EvidenceQueryOutcome,
    detail: str | None,
    metrics: tuple[ExplanationMetric, ...],
    unknown_fields: tuple[str, ...],
) -> str:
    failed = tuple(item for item in metrics if item.reason or item.measured is False)
    if outcome is EvidenceQueryOutcome.ROLLED_BACK:
        if failed:
            return "The candidate was rolled back after evaluation; rollback is not acceptance."
        return "The candidate was rolled back; rollback is not acceptance."
    if outcome is EvidenceQueryOutcome.UNSUPPORTED:
        return ((detail + " ") if detail else "") + (
            "The requested operation is outside the verified capability boundary; "
            "it was not treated as a failure; unsupported is distinct from failed."
        )
    if outcome is EvidenceQueryOutcome.FAILED:
        return detail or "The experiment failed before a qualifying result was established."
    if outcome is EvidenceQueryOutcome.UNKNOWN:
        return (
            "The outcome is unknown because the retained evidence is incomplete; "
            "no rejection or success is inferred."
        )
    if outcome is EvidenceQueryOutcome.INCONCLUSIVE:
        return (
            "The evidence is inconclusive; the retained measurements do not support "
            "a qualifying decision."
        )
    if failed:
        names = ", ".join(item.name for item in failed)
        return f"The measured constraint evidence did not qualify for: {names}."
    if unknown_fields:
        return (
            "The candidate cannot be classified from complete evidence; missing fields "
            "remain unknown."
        )
    return detail or "No negative qualification reason was retained."


def _explanation(record: object) -> NegativeEvidenceExplanation:
    raw_record = _mapping(record, "query evidence record")
    selected_fields = raw_record.get("fields")
    fields = {
        **raw_record,
        **(
            _mapping(selected_fields, "query evidence selected fields")
            if isinstance(selected_fields, Mapping)
            else {}
        ),
    }
    evidence_id = _text(fields.get("evidence_id"), "query evidence ID")
    campaign_id = _text(fields.get("campaign_id"), "query campaign ID")
    try:
        outcome = EvidenceQueryOutcome(_text(fields.get("outcome"), "query outcome"))
    except ValueError as error:
        raise NegativeEvidenceExplanationError("query outcome is unknown") from error
    source_outcome = _text(fields.get("source_outcome"), "query source outcome")
    detail = fields.get("detail")
    detail_text = None if detail is None else _text(detail, "query detail")
    measurements: list[EvidenceMeasurement] = []
    for value in _mapping_list(fields.get("measurements")):
        if not isinstance(value, Mapping):
            continue
        name = value.get("metric")
        raw = value.get("value")
        if isinstance(name, str) and isinstance(raw, (int, float)) and not isinstance(raw, bool):
            uncertainty = value.get("uncertainty")
            parsed_uncertainty = None
            if isinstance(uncertainty, Mapping):
                parsed_uncertainty = EvidenceUncertainty(
                    None
                    if uncertainty.get("lower_bound") is None
                    else _finite(uncertainty.get("lower_bound"), "uncertainty lower bound"),
                    None
                    if uncertainty.get("upper_bound") is None
                    else _finite(uncertainty.get("upper_bound"), "uncertainty upper bound"),
                    None
                    if uncertainty.get("confidence") is None
                    else _finite(uncertainty.get("confidence"), "uncertainty confidence"),
                    None
                    if uncertainty.get("standard_error") is None
                    else _finite(uncertainty.get("standard_error"), "uncertainty standard error"),
                    None
                    if not isinstance(uncertainty.get("sample_count"), int)
                    or isinstance(uncertainty.get("sample_count"), bool)
                    else uncertainty.get("sample_count"),
                )
            measurements.append(
                EvidenceMeasurement(
                    name,
                    float(raw),
                    cast(str | None, value.get("unit")),
                    parsed_uncertainty,
                )
            )
    evaluation = _record_mapping(fields.get("evaluation_record"))
    metric_values = tuple(
        _metric_from_measurement(item, evidence_id, evaluation) for item in measurements
    )
    existing = {item.name for item in metric_values}
    metrics = tuple(
        sorted(
            (*metric_values, *_evaluation_metrics(evaluation, evidence_id, existing)),
            key=lambda item: item.name,
        )
    )
    unknown: set[str] = set()
    for name in ("mutation_record",):
        if fields.get(name) in (None, {}, []):
            unknown.add(name)
    if outcome in {EvidenceQueryOutcome.REJECTED, EvidenceQueryOutcome.ROLLED_BACK} and fields.get(
        "evaluation_record"
    ) in (None, {}, []):
        unknown.add("evaluation_record")
    if outcome is EvidenceQueryOutcome.ROLLED_BACK and fields.get("rollback_record") in (
        None,
        {},
        [],
    ):
        unknown.add("rollback_record")
    if outcome in {EvidenceQueryOutcome.REJECTED, EvidenceQueryOutcome.ROLLED_BACK} and not metrics:
        unknown.add("measurements")
    if outcome in {EvidenceQueryOutcome.UNKNOWN, EvidenceQueryOutcome.INCONCLUSIVE}:
        unknown.add("qualification")
    required_fields = set(unknown)
    unknown.update(
        item for item in _string_list(fields.get("missing_fields")) if item in required_fields
    )
    unknown_fields = tuple(sorted(unknown))
    return NegativeEvidenceExplanation(
        evidence_id,
        campaign_id,
        outcome,
        source_outcome,
        detail_text or f"Experiment outcome: {outcome.value}.",
        _why_not_qualified(outcome, detail_text, metrics, unknown_fields),
        metrics,
        _lineage(fields, evidence_id, _text(fields.get("source_digest"), "query source digest")),
        {
            key: fields[key]
            for key in ("mutation_record", "evaluation_record", "rollback_record")
            if fields.get(key) is not None
        },
        unknown_fields,
        ExplanationCompleteness.INCOMPLETE if unknown_fields else ExplanationCompleteness.COMPLETE,
    )


def build_negative_evidence_report(response: EvidenceQueryResponse) -> NegativeEvidenceReport:
    """Build a deterministic report from one canonical #475 response."""

    evidence_outcome = (
        PolicyOutcome.ALLOW
        if response.status is EvidenceQueryStatus.COMPLETE and not response.missing_fields
        else PolicyOutcome.UNKNOWN
    )
    policy_decision = resolve_policy(
        "negative-evidence-explanation",
        (
            PolicyCandidate(
                PolicySource.HARD_CONSTRAINTS,
                PolicyOutcome.UNKNOWN,
                "the negative-evidence query does not authorize constraint changes",
            ),
            PolicyCandidate(
                PolicySource.VALIDATED_SPEC,
                PolicyOutcome.UNKNOWN,
                "no new executable specification is emitted by an explanation",
            ),
            PolicyCandidate(
                PolicySource.EVIDENCE_STATUS,
                evidence_outcome,
                "incomplete or unavailable evidence remains explicitly non-executable",
            ),
            PolicyCandidate(
                PolicySource.PROMPT,
                PolicyOutcome.ALLOW,
                "prompt text cannot turn negative evidence into acceptance",
            ),
            PolicyCandidate(
                PolicySource.PROVIDER,
                PolicyOutcome.ALLOW,
                "provider text cannot replace the canonical evidence query",
            ),
        ),
    )
    return NegativeEvidenceReport(
        response.query.query_id,
        response.snapshot.snapshot_id,
        response.response_digest,
        response.status,
        tuple(_explanation(item.to_record()) for item in response.records),
        response.missing_fields,
        response.unavailable_fields,
        policy_decision=policy_decision,
    )


def explain_negative_evidence(
    snapshot: EvidenceSnapshot, request: EvidenceQuery
) -> NegativeEvidenceReport:
    """Query canonical evidence and explain every selected outcome."""

    selected = tuple(sorted(set((*request.fields, *_REQUIRED_QUERY_FIELDS))))
    response = EvidenceQueryEngine(snapshot).query(replace(request, fields=selected))
    return build_negative_evidence_report(response)


def direct_negative_evidence_report(
    snapshot: EvidenceSnapshot, request: EvidenceQuery
) -> NegativeEvidenceReport:
    """Direct/report API parity entry point for negative evidence."""

    return explain_negative_evidence(snapshot, request)


__all__ = [
    "NEGATIVE_EVIDENCE_EXPLANATION_SCHEMA_VERSION",
    "ExplanationCompleteness",
    "ExplanationLineage",
    "ExplanationMetric",
    "NegativeEvidenceExplanation",
    "NegativeEvidenceExplanationError",
    "NegativeEvidenceReport",
    "build_negative_evidence_report",
    "direct_negative_evidence_report",
    "explain_negative_evidence",
]
