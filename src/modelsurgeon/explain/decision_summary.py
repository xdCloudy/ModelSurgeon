"""Deterministic machine and human summaries of mutation decisions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Final

from modelsurgeon.active_learning.candidate_scoring import CandidateScore
from modelsurgeon.experiments.candidates import MutationCandidate
from modelsurgeon.explain.attribution import AttributionReport, AttributionUnavailable
from modelsurgeon.surgery.contracts import MutationDelta

DECISION_SUMMARY_SCHEMA_VERSION: Final[int] = 1


class DecisionSummaryError(ValueError):
    """Raised when a decision summary would combine inconsistent evidence."""


@dataclass(frozen=True, slots=True)
class QuantizationContext:
    model_format: str
    codec: str | None
    evidence_source: str
    direct_quantized_features: bool | None

    def __post_init__(self) -> None:
        if not self.model_format or not self.evidence_source:
            raise DecisionSummaryError("quantization context identity is required")
        if self.codec is not None and not self.codec:
            raise DecisionSummaryError("quantization codec cannot be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "model_format": self.model_format,
            "codec": self.codec,
            "evidence_source": self.evidence_source,
            "direct_quantized_features": self.direct_quantized_features,
        }


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    feature_name: str
    contribution: float
    input_value: float
    source_kind: str
    source_name: str
    missing: bool

    def __post_init__(self) -> None:
        if not self.feature_name or not self.source_kind or not self.source_name:
            raise DecisionSummaryError("decision evidence provenance is required")
        if not math.isfinite(self.contribution) or not math.isfinite(self.input_value):
            raise DecisionSummaryError("decision evidence values must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "contribution": self.contribution,
            "input_value": self.input_value,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class ExpectedDeltaSummary:
    parameters: int | None
    flops: int | None
    memory_bytes: int | None
    storage_bytes: int | None

    @property
    def status(self) -> str:
        return "unknown" if self.parameters is None else "estimated"

    def __post_init__(self) -> None:
        values = (self.parameters, self.flops, self.memory_bytes, self.storage_bytes)
        if any(value is None for value in values) and any(value is not None for value in values):
            raise DecisionSummaryError("expected deltas must be entirely known or unknown")
        if any(
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in values
        ):
            raise DecisionSummaryError("known expected deltas must be integer counts")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "parameters": self.parameters,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class MutationDecisionSummary:
    candidate_id: str
    component_id: str
    mutation_id: str
    scope: str
    safe_probability: float
    raw_safe_probability: float
    utility: float
    uncertainty: float
    expected_outcomes: tuple[tuple[str, float], ...]
    expected_delta: ExpectedDeltaSummary
    top_evidence: tuple[DecisionEvidence, ...]
    attribution_status: str
    attribution_reason: str | None
    attribution_output_space: str | None
    quantization_context: QuantizationContext | None
    schema_version: int = DECISION_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != DECISION_SUMMARY_SCHEMA_VERSION
            or not self.candidate_id.startswith("cand_")
            or not self.component_id
            or re.fullmatch(r"[0-9a-f]{64}", self.mutation_id) is None
            or not self.scope
        ):
            raise DecisionSummaryError("decision summary identity is invalid")
        values = (
            self.safe_probability,
            self.raw_safe_probability,
            self.utility,
            self.uncertainty,
            *(value for _, value in self.expected_outcomes),
        )
        if any(not math.isfinite(value) for value in values):
            raise DecisionSummaryError("decision summary estimates must be finite")
        if not 0 <= self.safe_probability <= 1 or not 0 <= self.raw_safe_probability <= 1:
            raise DecisionSummaryError("decision safe probabilities must be within [0, 1]")
        if self.uncertainty < 0:
            raise DecisionSummaryError("decision uncertainty cannot be negative")
        names = tuple(name for name, _ in self.expected_outcomes)
        if names != tuple(sorted(set(names))) or any(not name for name in names):
            raise DecisionSummaryError("expected outcome names must be canonical")
        evidence_names = tuple(item.feature_name for item in self.top_evidence)
        if len(evidence_names) != len(set(evidence_names)):
            raise DecisionSummaryError("top decision evidence cannot repeat features")
        if self.attribution_status not in {"available", "unavailable"}:
            raise DecisionSummaryError("attribution status is invalid")
        if self.attribution_status == "available":
            if self.attribution_reason is not None or self.attribution_output_space is None:
                raise DecisionSummaryError("available attribution metadata is inconsistent")
        elif self.attribution_reason is None or self.attribution_output_space is not None:
            raise DecisionSummaryError("unavailable attribution metadata is inconsistent")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate": {
                "candidate_id": self.candidate_id,
                "component_id": self.component_id,
                "mutation_id": self.mutation_id,
                "scope": self.scope,
            },
            "prediction": {
                "safe_probability": self.safe_probability,
                "raw_safe_probability": self.raw_safe_probability,
                "utility": self.utility,
                "uncertainty": self.uncertainty,
                "expected_outcomes": dict(self.expected_outcomes),
            },
            "expected_delta": self.expected_delta.to_record(),
            "attribution": {
                "status": self.attribution_status,
                "reason": self.attribution_reason,
                "output_space": self.attribution_output_space,
                "top_evidence": [item.to_record() for item in self.top_evidence],
            },
            "quantization_context": (
                None if self.quantization_context is None else self.quantization_context.to_record()
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))

    def to_text(self) -> str:
        unknown = "unknown"

        def number(value: float) -> str:
            return format(value, ".12g")

        delta = self.expected_delta
        lines = [
            f"Candidate: {self.candidate_id}",
            f"Component: {self.component_id}",
            f"Mutation: {self.mutation_id} ({self.scope})",
            f"Safe probability: {number(self.safe_probability)}",
            f"Raw safe probability: {number(self.raw_safe_probability)}",
            f"Utility: {number(self.utility)}",
            f"Uncertainty: {number(self.uncertainty)}",
            "Expected outcomes:",
        ]
        lines.extend(f"  {name}: {number(value)}" for name, value in self.expected_outcomes)
        lines.append(f"Expected deltas ({delta.status}):")
        for name, value in (
            ("parameters", delta.parameters),
            ("flops", delta.flops),
            ("memory_bytes", delta.memory_bytes),
            ("storage_bytes", delta.storage_bytes),
        ):
            lines.append(f"  {name}: {unknown if value is None else value}")
        lines.append(f"Attribution: {self.attribution_status}")
        if self.attribution_status == "available":
            lines.append(f"Attribution output space: {self.attribution_output_space}")
            lines.append("Top evidence:")
            lines.extend(
                f"  {item.feature_name}: contribution={number(item.contribution)}, "
                f"value={number(item.input_value)}, missing={str(item.missing).lower()}"
                for item in self.top_evidence
            )
        else:
            lines.append(f"Attribution reason: {self.attribution_reason}")
        context = self.quantization_context
        lines.append("Quantization context:")
        lines.extend(
            (
                f"  model_format: {unknown if context is None else context.model_format}",
                "  codec: "
                + (unknown if context is None or context.codec is None else context.codec),
                f"  evidence_source: {unknown if context is None else context.evidence_source}",
                "  direct_quantized_features: "
                + (
                    unknown
                    if context is None or context.direct_quantized_features is None
                    else str(context.direct_quantized_features).lower()
                ),
            )
        )
        return "\n".join(lines) + "\n"


def generate_mutation_decision_summary(
    candidate: MutationCandidate,
    score: CandidateScore,
    *,
    expected_delta: MutationDelta | None,
    attribution: AttributionReport | AttributionUnavailable,
    quantization_context: QuantizationContext | None,
    top_evidence_count: int = 5,
) -> MutationDecisionSummary:
    """Combine one candidate's evidence without manufacturing absent estimates."""

    if score.candidate_id != candidate.candidate_id:
        raise DecisionSummaryError("candidate and score identities do not match")
    if not 1 <= top_evidence_count <= 100:
        raise DecisionSummaryError("top evidence count must be within 1..100")
    if expected_delta is None:
        delta = ExpectedDeltaSummary(None, None, None, None)
    else:
        delta = ExpectedDeltaSummary(
            expected_delta.parameters,
            expected_delta.flops,
            expected_delta.memory_bytes,
            expected_delta.storage_bytes,
        )
    if isinstance(attribution, AttributionReport):
        first = attribution.predictions[0]
        ranked = sorted(
            first.contributions,
            key=lambda item: (-abs(item.contribution), item.provenance.feature_name),
        )[:top_evidence_count]
        evidence = tuple(
            DecisionEvidence(
                item.provenance.feature_name,
                item.contribution,
                item.input_value,
                item.provenance.source_kind,
                item.provenance.source_name,
                item.missing,
            )
            for item in ranked
        )
        attribution_status = "available"
        attribution_reason = None
        output_space = attribution.output_space
    else:
        evidence = ()
        attribution_status = "unavailable"
        attribution_reason = attribution.reason
        output_space = None
    return MutationDecisionSummary(
        candidate.candidate_id,
        str(candidate.component_id),
        candidate.mutation_id,
        candidate.scope.value,
        score.safe_probability,
        score.raw_safe_probability,
        score.utility,
        score.uncertainty,
        score.outcomes,
        delta,
        evidence,
        attribution_status,
        attribution_reason,
        output_space,
        quantization_context,
    )
