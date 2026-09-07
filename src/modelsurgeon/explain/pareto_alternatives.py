"""Measured Pareto alternatives and conservative trade-off explanations.

This module is a read-only projection over the canonical evidence archive used
by the measured feasibility explanation.  It never turns a prediction or a
terminal negative result into a frontier point and never invents a metric
observation.  Objective intervals are compared conservatively: a point only
dominates another when its worst case is no worse than the other's best case.
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.explain.infeasibility import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityExplanation,
    FeasibilityProvenance,
    FeasibilityResourceBounds,
    RetainedEvidenceReference,
    build_feasibility_explanation,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricObservation,
    ObjectiveContract,
    ObjectiveDirection,
    SoftObjective,
)

PARETO_ALTERNATIVES_SCHEMA_VERSION = 1


class ParetoAlternativesError(ValueError):
    """Raised when canonical evidence cannot support a safe projection."""


class ParetoAlternativeStatus(StrEnum):
    """The evidence-backed disposition of a measured alternative."""

    FRONTIER = "frontier"
    TIED_FRONTIER = "tied_frontier"
    DOMINATED = "dominated"
    INCOMPARABLE = "incomparable"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INCOMPLETE = "incomplete"
    DISPOSITION_NOT_ACCEPTED = "disposition_not_accepted"


class ParetoRenderFormat(StrEnum):
    """Supported deterministic representations of one explanation."""

    DIRECT = "direct"
    CHAT = "chat"
    HTML = "html"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParetoAlternativesError(f"{label} must be non-empty text")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if result != tuple(values) or len(result) != len(values):
        raise ParetoAlternativesError(f"{label} must be sorted and unique")
    if any(not value.strip() for value in result):
        raise ParetoAlternativesError(f"{label} cannot contain blank values")
    return result


@dataclass(frozen=True, slots=True)
class ParetoResourceBounds:
    """Hard limits for frontier comparison and rendered explanation output."""

    max_candidates: int = 10_000
    max_alternatives: int = 10_000
    max_frontier: int = 10_000
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_candidates, "maximum candidates"),
            (self.max_alternatives, "maximum alternatives"),
            (self.max_frontier, "maximum frontier candidates"),
            (self.max_output_bytes, "maximum output bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ParetoAlternativesError(f"{label} must be positive")

    def to_record(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_alternatives": self.max_alternatives,
            "max_frontier": self.max_frontier,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class ParetoResourceUsage:
    """Measured accounting for one bounded Pareto projection."""

    evidence_records: int
    measured_alternatives: int
    frontier_candidates: int
    dominance_comparisons: int
    output_bytes: int

    def to_record(self) -> dict[str, int]:
        return {
            "evidence_records": self.evidence_records,
            "measured_alternatives": self.measured_alternatives,
            "frontier_candidates": self.frontier_candidates,
            "dominance_comparisons": self.dominance_comparisons,
            "output_bytes": self.output_bytes,
        }


_DEFAULT_PARETO_RESOURCE_BOUNDS = ParetoResourceBounds()


@dataclass(frozen=True, slots=True)
class ParetoMetricDelta:
    """One measured objective value and its explicit baseline delta."""

    metric: str
    unit: str
    direction: ObjectiveDirection
    value: float
    conservative_value: float
    lower: float | None
    upper: float | None
    baseline: float | None
    delta_from_baseline: float | None

    @property
    def uncertain(self) -> bool:
        return self.lower is not None or self.upper is not None

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "direction": self.direction.value,
            "value": self.value,
            "conservative_value": self.conservative_value,
            "lower": self.lower,
            "upper": self.upper,
            "baseline": self.baseline,
            "delta_from_baseline": self.delta_from_baseline,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True, slots=True)
class ParetoConstraintViolation:
    """A conservative measured violation of one declared hard constraint."""

    constraint: HardConstraint
    observed: float
    gap: float
    normalized_gap: float
    uncertain: bool

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "observed": self.observed,
            "gap": self.gap,
            "normalized_gap": self.normalized_gap,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True, slots=True)
class ParetoAlternative:
    """One measured candidate with all context needed to explain its trade-off."""

    candidate_id: str
    evidence_id: str
    evaluation_id: str | None
    disposition: CandidateDisposition
    status: ParetoAlternativeStatus
    metric_deltas: tuple[ParetoMetricDelta, ...]
    constraint_violations: tuple[ParetoConstraintViolation, ...]
    missing_metrics: tuple[str, ...]
    uncertainty_metrics: tuple[str, ...]
    dominated_by_candidate_ids: tuple[str, ...]
    tied_candidate_ids: tuple[str, ...]
    candidate_provenance: Mapping[str, object]
    resource_usage: Mapping[str, object] | None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate ID")
        _text(self.evidence_id, "evidence ID")
        _sorted_unique(self.missing_metrics, "alternative missing metrics")
        _sorted_unique(self.uncertainty_metrics, "alternative uncertainty metrics")
        _sorted_unique(self.dominated_by_candidate_ids, "alternative dominator IDs")
        _sorted_unique(self.tied_candidate_ids, "alternative tie IDs")
        metrics = tuple(item.metric for item in self.metric_deltas)
        if metrics != tuple(sorted(set(metrics))):
            raise ParetoAlternativesError("alternative metric deltas must be sorted and unique")
        constraints = tuple(item.constraint.metric for item in self.constraint_violations)
        if constraints != tuple(sorted(set(constraints))):
            raise ParetoAlternativesError("alternative constraint violations must be sorted")

    @property
    def on_frontier(self) -> bool:
        return self.status in {
            ParetoAlternativeStatus.FRONTIER,
            ParetoAlternativeStatus.TIED_FRONTIER,
        }

    @property
    def uncertain(self) -> bool:
        return bool(self.uncertainty_metrics)

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "evaluation_id": self.evaluation_id,
            "measured": True,
            "disposition": self.disposition.value,
            "status": self.status.value,
            "on_frontier": self.on_frontier,
            "metric_deltas": [item.to_record() for item in self.metric_deltas],
            "constraint_violations": [item.to_record() for item in self.constraint_violations],
            "missing_metrics": list(self.missing_metrics),
            "uncertainty_metrics": list(self.uncertainty_metrics),
            "dominated_by_candidate_ids": list(self.dominated_by_candidate_ids),
            "tied_candidate_ids": list(self.tied_candidate_ids),
            "candidate_provenance": dict(self.candidate_provenance),
            "resource_usage": (None if self.resource_usage is None else dict(self.resource_usage)),
        }


@dataclass(frozen=True, slots=True)
class ParetoAlternativesExplanation:
    """Canonical direct, chat, and HTML source for measured alternatives."""

    contract_id: str
    archive_id: str
    alternatives: tuple[ParetoAlternative, ...]
    frontier_candidate_ids: tuple[str, ...]
    retained_evidence: tuple[RetainedEvidenceReference, ...]
    feasibility: FeasibilityExplanation
    provenance: FeasibilityProvenance
    resource_bounds: ParetoResourceBounds
    resource_usage: ParetoResourceUsage
    reason: str
    schema_version: int = PARETO_ALTERNATIVES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.contract_id, "objective contract ID")
        _text(self.archive_id, "evidence archive ID")
        candidate_ids = tuple(item.candidate_id for item in self.alternatives)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ParetoAlternativesError("alternatives must be sorted and unique")
        _sorted_unique(self.frontier_candidate_ids, "frontier candidate IDs")
        if not set(self.frontier_candidate_ids) <= set(candidate_ids):
            raise ParetoAlternativesError("frontier candidates must be measured alternatives")
        retained_ids = tuple(item.candidate_id for item in self.retained_evidence)
        if retained_ids != tuple(sorted(set(retained_ids))):
            raise ParetoAlternativesError("retained evidence must be sorted and unique")
        if self.feasibility.contract_id != self.contract_id:
            raise ParetoAlternativesError("feasibility contract does not match alternatives")
        if self.feasibility.provenance.archive_id != self.archive_id:
            raise ParetoAlternativesError("feasibility archive does not match alternatives")
        _text(self.reason, "Pareto explanation reason")

    def _record_without_id(self) -> dict[str, object]:
        return {
            "record_type": "measured_pareto_alternatives",
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "archive_id": self.archive_id,
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "alternatives": [item.to_record() for item in self.alternatives],
            "retained_evidence": [item.to_record() for item in self.retained_evidence],
            "feasibility": self.feasibility.to_record(),
            "provenance": self.provenance.to_record(),
            "resource_bounds": self.resource_bounds.to_record(),
            "resource_usage": self.resource_usage.to_record(),
            "reason": self.reason,
        }

    @property
    def result_id(self) -> str:
        return (
            "pareto_result_"
            + hashlib.sha256(
                canonical_identity_json(self._record_without_id()).encode("utf-8")
            ).hexdigest()
        )

    def to_record(self) -> dict[str, object]:
        return {**self._record_without_id(), "result_id": self.result_id}

    def direct_report(self) -> dict[str, object]:
        """Return the canonical direct representation."""

        return self.to_record()

    def chat_text(self) -> str:
        """Render a deterministic, evidence-linked control-plane explanation."""

        lines = [
            "Measured Pareto alternatives",
            f"Outcome: {self.feasibility.outcome.value}",
            "Frontier: " + (", ".join(self.frontier_candidate_ids) or "none"),
            f"Archive: {self.archive_id}",
        ]
        for alternative in self.alternatives:
            lines.append(
                f"- {alternative.candidate_id} (evidence {alternative.evidence_id}): "
                f"{alternative.status.value}; disposition={alternative.disposition.value}"
            )
            if alternative.metric_deltas:
                metrics = ", ".join(
                    f"{item.metric}={item.conservative_value:g}"
                    for item in alternative.metric_deltas
                )
                lines.append(f"  metrics: {metrics}")
            if alternative.constraint_violations:
                violations = ", ".join(
                    f"{item.constraint.metric} gap={item.gap:g}"
                    for item in alternative.constraint_violations
                )
                lines.append(f"  hard-constraint violations: {violations}")
            if alternative.missing_metrics:
                lines.append("  missing evidence: " + ", ".join(alternative.missing_metrics))
            if alternative.uncertainty_metrics:
                lines.append("  uncertainty: " + ", ".join(alternative.uncertainty_metrics))
            if alternative.dominated_by_candidate_ids:
                lines.append("  dominated by: " + ", ".join(alternative.dominated_by_candidate_ids))
            if alternative.tied_candidate_ids:
                lines.append("  tied with: " + ", ".join(alternative.tied_candidate_ids))
        for item in self.retained_evidence:
            if item.status is not CandidateEvidenceStatus.MEASURED:
                lines.append(
                    f"- retained evidence {item.evidence_id} for {item.candidate_id}: "
                    f"{item.status.value}; disposition={item.disposition.value}"
                )
        lines.append(f"Provenance: source={self.provenance.source_model_digest}")
        lines.append(f"Result: {self.result_id}")
        return "\n".join(lines)


def _objective_delta(observation: MetricObservation, objective: SoftObjective) -> ParetoMetricDelta:
    conservative = observation.conservative(objective.direction)
    baseline = objective.baseline
    if baseline is None:
        delta = None
    elif objective.direction is ObjectiveDirection.MAXIMIZE:
        delta = conservative - baseline
    else:
        delta = baseline - conservative
    return ParetoMetricDelta(
        objective.metric,
        objective.unit.value,
        objective.direction,
        observation.value,
        conservative,
        observation.lower,
        observation.upper,
        baseline,
        delta,
    )


def _constraint_violations(
    candidate: FeasibilityCandidateEvidence,
    constraints: Sequence[HardConstraint],
) -> tuple[ParetoConstraintViolation, ...]:
    observations = {item.metric: item for item in candidate.observations}
    output: list[ParetoConstraintViolation] = []
    for constraint in constraints:
        observation = observations.get(constraint.metric)
        if observation is None:
            continue
        observed = observation.conservative(constraint.direction)
        passed = (
            observed >= constraint.threshold
            if constraint.direction is ConstraintDirection.MINIMUM
            else observed <= constraint.threshold
        )
        if not passed:
            gap = (
                constraint.threshold - observed
                if constraint.direction is ConstraintDirection.MINIMUM
                else observed - constraint.threshold
            )
            output.append(
                ParetoConstraintViolation(
                    constraint,
                    observed,
                    gap,
                    gap / max(abs(constraint.threshold), 1.0),
                    observation.lower is not None or observation.upper is not None,
                )
            )
    return tuple(output)


def _missing_metrics(
    candidate: FeasibilityCandidateEvidence,
    contract: ObjectiveContract,
) -> tuple[str, ...]:
    available = {item.metric for item in candidate.observations}
    required = {item.metric for item in contract.constraints}
    required.update(item.metric for item in contract.objectives)
    return tuple(sorted(required - available))


def _uncertainty_metrics(candidate: FeasibilityCandidateEvidence) -> tuple[str, ...]:
    return tuple(
        sorted(
            item.metric
            for item in candidate.observations
            if item.lower is not None or item.upper is not None
        )
    )


def _dominates(left: tuple[ParetoMetricDelta, ...], right: tuple[ParetoMetricDelta, ...]) -> bool:
    if tuple(item.metric for item in left) != tuple(item.metric for item in right):
        return False
    strictly_better = False
    for left_value, right_value in zip(left, right, strict=True):
        if left_value.direction is ObjectiveDirection.MAXIMIZE:
            left_low = left_value.lower if left_value.lower is not None else left_value.value
            right_high = right_value.upper if right_value.upper is not None else right_value.value
            if left_low < right_high:
                return False
            strictly_better = strictly_better or left_low > right_high
        else:
            left_high = left_value.upper if left_value.upper is not None else left_value.value
            right_low = right_value.lower if right_value.lower is not None else right_value.value
            if left_high > right_low:
                return False
            strictly_better = strictly_better or left_high < right_low
    return strictly_better


def _ties(left: tuple[ParetoMetricDelta, ...], right: tuple[ParetoMetricDelta, ...]) -> bool:
    if tuple(item.metric for item in left) != tuple(item.metric for item in right):
        return False
    return all(
        (item.lower if item.lower is not None else item.value)
        == (other.lower if other.lower is not None else other.value)
        and (item.upper if item.upper is not None else item.value)
        == (other.upper if other.upper is not None else other.value)
        for item, other in zip(left, right, strict=True)
    )


def build_pareto_alternatives(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    resource_bounds: ParetoResourceBounds = _DEFAULT_PARETO_RESOURCE_BOUNDS,
) -> ParetoAlternativesExplanation:
    """Build a conservative Pareto explanation from canonical measured evidence."""

    if not contract.objectives:
        raise ParetoAlternativesError("Pareto alternatives require at least one objective")
    if isinstance(evidence, CanonicalEvidenceArchive):
        archive = evidence
        if archive.contract_id != contract.contract_id:
            raise ParetoAlternativesError("archive objective contract does not match")
    else:
        if len(evidence) > resource_bounds.max_candidates:
            raise ParetoAlternativesError("evidence exceeds the declared candidate bound")
        archive = CanonicalEvidenceArchive.build(contract, evidence)
    if len(archive.candidates) > resource_bounds.max_candidates:
        raise ParetoAlternativesError("archive exceeds the declared candidate bound")
    if len(archive.candidates) > resource_bounds.max_alternatives:
        raise ParetoAlternativesError("alternatives exceed the declared alternative bound")

    feasibility = build_feasibility_explanation(
        contract,
        archive,
        provenance=provenance,
        resource_bounds=FeasibilityResourceBounds(
            max_candidates=resource_bounds.max_candidates,
            max_near_misses=resource_bounds.max_alternatives,
            max_next_actions=8,
            max_output_bytes=max(resource_bounds.max_output_bytes, 1_000_000),
        ),
    )
    measured = tuple(item for item in archive.candidates if item.measured)
    if len(measured) > resource_bounds.max_alternatives:
        raise ParetoAlternativesError("measured alternatives exceed the declared bound")

    partial: list[tuple[FeasibilityCandidateEvidence, ParetoAlternative, bool]] = []
    for candidate in measured:
        missing = _missing_metrics(candidate, contract)
        violations = _constraint_violations(candidate, contract.constraints)
        observations = {item.metric: item for item in candidate.observations}
        metric_deltas = tuple(
            _objective_delta(observations[item.metric], item)
            for item in sorted(contract.objectives, key=lambda value: value.metric)
            if item.metric in observations
        )
        eligible_for_frontier = not missing and not violations
        if candidate.disposition is not CandidateDisposition.ACCEPTED:
            eligible_for_frontier = False
        partial.append(
            (
                candidate,
                ParetoAlternative(
                    candidate.candidate_id,
                    candidate.evidence_id,
                    candidate.evaluation_id,
                    candidate.disposition,
                    ParetoAlternativeStatus.INCOMPLETE
                    if missing
                    else ParetoAlternativeStatus.CONSTRAINT_VIOLATION
                    if violations
                    else ParetoAlternativeStatus.DISPOSITION_NOT_ACCEPTED
                    if candidate.disposition is not CandidateDisposition.ACCEPTED
                    else ParetoAlternativeStatus.INCOMPARABLE,
                    metric_deltas,
                    violations,
                    missing,
                    _uncertainty_metrics(candidate),
                    (),
                    (),
                    candidate.provenance,
                    (
                        None
                        if candidate.resource_usage is None
                        else candidate.resource_usage.to_record()
                    ),
                ),
                eligible_for_frontier,
            )
        )

    eligible = [(candidate, alternative) for candidate, alternative, ok in partial if ok]
    dominators: dict[str, tuple[str, ...]] = {}
    ties: dict[str, tuple[str, ...]] = {}
    comparisons = 0
    for candidate, alternative in eligible:
        dominates: list[str] = []
        tied: list[str] = []
        for other_candidate, other in eligible:
            if candidate.candidate_id == other_candidate.candidate_id:
                continue
            comparisons += 1
            if _dominates(other.metric_deltas, alternative.metric_deltas):
                dominates.append(other_candidate.candidate_id)
            elif _ties(other.metric_deltas, alternative.metric_deltas):
                tied.append(other_candidate.candidate_id)
        dominators[candidate.candidate_id] = tuple(sorted(dominates))
        ties[candidate.candidate_id] = tuple(sorted(tied))

    frontier = tuple(
        sorted(
            candidate.candidate_id
            for candidate, _ in eligible
            if not dominators[candidate.candidate_id]
        )
    )
    if len(frontier) > resource_bounds.max_frontier:
        raise ParetoAlternativesError("frontier exceeds the declared frontier bound")
    alternatives: list[ParetoAlternative] = []
    for candidate, alternative, _ in partial:
        candidate_dominators = dominators.get(candidate.candidate_id, ())
        candidate_ties = ties.get(candidate.candidate_id, ())
        if alternative.status in {
            ParetoAlternativeStatus.INCOMPLETE,
            ParetoAlternativeStatus.CONSTRAINT_VIOLATION,
            ParetoAlternativeStatus.DISPOSITION_NOT_ACCEPTED,
        }:
            status = alternative.status
        elif candidate_dominators:
            status = ParetoAlternativeStatus.DOMINATED
        elif candidate.candidate_id in frontier and candidate_ties:
            status = ParetoAlternativeStatus.TIED_FRONTIER
        else:
            status = ParetoAlternativeStatus.FRONTIER
        alternatives.append(
            replace(
                alternative,
                status=status,
                dominated_by_candidate_ids=candidate_dominators,
                tied_candidate_ids=candidate_ties,
            )
        )

    if frontier:
        reason = (
            "measured hard-constraint-feasible candidates define a conservative Pareto frontier"
        )
    else:
        reason = feasibility.reason
    usage = ParetoResourceUsage(
        len(archive.candidates),
        len(alternatives),
        len(frontier),
        comparisons,
        0,
    )
    result = ParetoAlternativesExplanation(
        contract.contract_id,
        archive.archive_id,
        tuple(alternatives),
        frontier,
        feasibility.retained_evidence,
        feasibility,
        feasibility.provenance,
        resource_bounds,
        usage,
        reason,
    )
    for _ in range(3):
        output_bytes = len(canonical_identity_json(result.to_record()).encode("utf-8"))
        if output_bytes == result.resource_usage.output_bytes:
            break
        result = replace(
            result,
            resource_usage=replace(result.resource_usage, output_bytes=output_bytes),
        )
    if result.resource_usage.output_bytes > resource_bounds.max_output_bytes:
        raise ParetoAlternativesError("Pareto explanation exceeds the declared output bound")
    return result


def _safe_script(value: object) -> str:
    return (
        canonical_identity_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_pareto_html(
    explanation: ParetoAlternativesExplanation,
    *,
    title: str = "Measured Pareto alternatives",
) -> str:
    """Render a dependency-free HTML view from the canonical explanation."""

    _text(title, "Pareto HTML title")
    rows: list[str] = []
    for item in explanation.alternatives:
        metrics = (
            ", ".join(
                f"{html.escape(value.metric)}={value.conservative_value:g}"
                for value in item.metric_deltas
            )
            or "none"
        )
        violations = (
            ", ".join(
                f"{html.escape(value.constraint.metric)} gap={value.gap:g}"
                for value in item.constraint_violations
            )
            or "none"
        )
        uncertainty = ", ".join(html.escape(value) for value in item.uncertainty_metrics) or "none"
        dominators = (
            ", ".join(html.escape(value) for value in item.dominated_by_candidate_ids) or "none"
        )
        rows.append(
            f'<tr data-status="{html.escape(item.status.value)}">'
            f'<th scope="row">{html.escape(item.candidate_id)}</th>'
            f'<td><a href="#evidence-{html.escape(item.evidence_id)}">'
            f"{html.escape(item.evidence_id)}</a></td>"
            f"<td>{html.escape(item.status.value)}</td>"
            f"<td>{'yes' if item.on_frontier else 'no'}</td>"
            f"<td>{metrics}</td><td>{violations}</td><td>{uncertainty}</td><td>{dominators}</td></tr>"
        )
    retained = (
        "".join(
            f'<li id="evidence-{html.escape(item.evidence_id)}">{html.escape(item.candidate_id)} '
            f"/{html.escape(item.evidence_id)}: {html.escape(item.status.value)} "
            f"(disposition={html.escape(item.disposition.value)})</li>"
            for item in explanation.retained_evidence
            if item.status is not CandidateEvidenceStatus.MEASURED
        )
        or "<li>none</li>"
    )
    data = _safe_script(explanation.to_record())
    frontier = (
        ", ".join(html.escape(value) for value in explanation.frontier_candidate_ids) or "none"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>"
        ":root{font:16px system-ui,sans-serif;color:#172033;background:#f7f9fc}"
        "body{max-width:1600px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%;background:#fff}"
        "th,td{border:1px solid #ccd3df;padding:.5rem;text-align:left;"
        "vertical-align:top}th{background:#e9eef6}"
        ".frontier{color:#126b35;font-weight:700}.negative{color:#895b00}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p>Archive: "
        f"<code>{html.escape(explanation.archive_id)}</code>; "
        f"frontier: {frontier}</p>"
        "<table><caption>Measured alternatives and trade-offs</caption><thead><tr>"
        "<th>Candidate</th><th>Evidence</th><th>Frontier context</th><th>On frontier</th>"
        "<th>Metric deltas</th><th>Hard-constraint violations</th>"
        "<th>Uncertainty</th><th>Dominated by</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<h2>Retained non-measured evidence</h2><ul>"
        f"{retained}</ul>"
        f'<script type="application/json" id="modelsurgeon-pareto-data">{data}</script>'
        "</body></html>"
    )


def render_pareto_explanation(
    explanation: ParetoAlternativesExplanation,
    *,
    format: ParetoRenderFormat | str = ParetoRenderFormat.DIRECT,
) -> dict[str, object] | str:
    """Render direct, chat, or HTML from exactly one canonical explanation."""

    try:
        selected = ParetoRenderFormat(format)
    except ValueError as error:
        raise ParetoAlternativesError(f"unsupported Pareto render format: {format}") from error
    if selected is ParetoRenderFormat.DIRECT:
        return explanation.to_record()
    if selected is ParetoRenderFormat.CHAT:
        return explanation.chat_text()
    return render_pareto_html(explanation)


def render_pareto_alternatives(
    explanation: ParetoAlternativesExplanation,
    *,
    format: ParetoRenderFormat | str = ParetoRenderFormat.DIRECT,
) -> dict[str, object] | str:
    """Compatibility name for callers using the issue vocabulary."""

    return render_pareto_explanation(explanation, format=format)


def explain_pareto_alternatives(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    resource_bounds: ParetoResourceBounds = _DEFAULT_PARETO_RESOURCE_BOUNDS,
) -> ParetoAlternativesExplanation:
    """Verb-led compatibility alias for :func:`build_pareto_alternatives`."""

    return build_pareto_alternatives(
        contract,
        evidence,
        provenance=provenance,
        resource_bounds=resource_bounds,
    )


build_pareto_explanation = build_pareto_alternatives
ParetoExplanation = ParetoAlternativesExplanation


__all__ = [
    "PARETO_ALTERNATIVES_SCHEMA_VERSION",
    "ParetoAlternative",
    "ParetoAlternativeStatus",
    "ParetoAlternativesError",
    "ParetoAlternativesExplanation",
    "ParetoConstraintViolation",
    "ParetoExplanation",
    "ParetoMetricDelta",
    "ParetoRenderFormat",
    "ParetoResourceBounds",
    "ParetoResourceUsage",
    "build_pareto_alternatives",
    "build_pareto_explanation",
    "explain_pareto_alternatives",
    "render_pareto_alternatives",
    "render_pareto_explanation",
    "render_pareto_html",
]
