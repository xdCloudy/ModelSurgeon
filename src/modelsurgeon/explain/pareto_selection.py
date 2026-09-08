"""Deterministic explanations for measured Pareto candidate selection.

This module is a read-only projection over the measured Pareto archive.  It
does not choose from predictions, infeasible candidates, or terminal negative
evidence.  Selection scores are recomputed from the declared objective and
the conservative measured observations before the v2.0 decision-replay
record is emitted.
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from modelsurgeon.conversation.evidence_query import EvidenceQueryResponse
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import (
    DecisionReplay,
    replay_decision_evidence,
)
from modelsurgeon.explain.infeasibility import (
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
)
from modelsurgeon.explain.pareto_alternatives import (
    ParetoAlternative,
    ParetoAlternativesExplanation,
    ParetoAlternativeStatus,
    ParetoResourceBounds,
    build_pareto_alternatives,
)
from modelsurgeon.policy import (
    PolicyCandidate,
    PolicyDecision,
    PolicyOutcome,
    PolicySource,
    resolve_policy,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ObjectiveContract,
    ObjectiveMode,
    evaluate_contract,
)

PARETO_SELECTION_EXPLANATION_SCHEMA_VERSION = 1
_DEFAULT_PARETO_SELECTION_RESOURCE_BOUNDS = ParetoResourceBounds()


class ParetoSelectionError(ValueError):
    """Raised when canonical evidence cannot support a final selection."""


class ParetoSelectionOutcome(StrEnum):
    """The evidence-backed outcome of the final-selection projection."""

    SELECTED = "selected"
    NO_FEASIBLE_FRONTIER = "no_feasible_frontier"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParetoSelectionError(f"{label} must be non-empty text")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise ParetoSelectionError(f"{label} must be sorted and unique")
    if any(not value.strip() for value in result):
        raise ParetoSelectionError(f"{label} cannot contain blank values")
    return result


@dataclass(frozen=True, slots=True)
class SelectionObjectiveTerm:
    """One ordered hard constraint or soft objective shown to the user."""

    kind: str
    metric: str
    direction: str
    unit: str
    threshold: float | None = None
    weight: float | None = None
    normalization: str | None = None
    baseline: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"hard_constraint", "soft_objective"}:
            raise ParetoSelectionError("selection objective term kind is invalid")
        _text(self.metric, "selection objective metric")
        _text(self.direction, "selection objective direction")
        _text(self.unit, "selection objective unit")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "direction": self.direction,
            "unit": self.unit,
            "threshold": self.threshold,
            "weight": self.weight,
            "normalization": self.normalization,
            "baseline": self.baseline,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class SelectionConstraintStatus:
    """Conservative status for one hard constraint on the selected candidate."""

    metric: str
    direction: str
    threshold: float
    unit: str
    observed: float | None
    passed: bool
    uncertain: bool

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "threshold": self.threshold,
            "unit": self.unit,
            "observed": self.observed,
            "passed": self.passed,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True, slots=True)
class SelectionCandidateScore:
    """Recomputed objective values and deterministic rank for one alternative."""

    candidate_id: str
    evidence_id: str
    objective_values: tuple[float, ...] | None
    score: float | None
    rank: int | None
    on_frontier: bool
    uncertain: bool
    status: ParetoAlternativeStatus

    def __post_init__(self) -> None:
        _text(self.candidate_id, "selection candidate ID")
        _text(self.evidence_id, "selection evidence ID")
        if self.objective_values is not None and any(
            not isinstance(value, (int, float)) for value in self.objective_values
        ):
            raise ParetoSelectionError("selection objective values must be numeric")
        if self.rank is not None and self.rank < 0:
            raise ParetoSelectionError("selection rank cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "objective_values": (
                None if self.objective_values is None else list(self.objective_values)
            ),
            "score": self.score,
            "rank": self.rank,
            "on_frontier": self.on_frontier,
            "uncertain": self.uncertain,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class SelectionObjectiveIdentity:
    """Original/effective objective identities and amendment lineage."""

    original_objective_id: str
    effective_objective_id: str
    amendment_id: str | None = None
    amendment_diff_id: str | None = None
    amendment_status: str | None = None
    downstream_campaign_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.original_objective_id, "original objective identity")
        _text(self.effective_objective_id, "effective objective identity")
        for value, label in (
            (self.amendment_id, "amendment ID"),
            (self.amendment_diff_id, "amendment diff ID"),
            (self.amendment_status, "amendment status"),
            (self.downstream_campaign_id, "downstream campaign ID"),
        ):
            if value is not None:
                _text(value, label)

    def to_record(self) -> dict[str, object]:
        return {
            "original_objective_id": self.original_objective_id,
            "effective_objective_id": self.effective_objective_id,
            "amendment_id": self.amendment_id,
            "amendment_diff_id": self.amendment_diff_id,
            "amendment_status": self.amendment_status,
            "downstream_campaign_id": self.downstream_campaign_id,
        }


@dataclass(frozen=True, slots=True)
class ParetoSelectionExplanation:
    """Canonical direct, chat, and HTML explanation of one selection."""

    contract_id: str
    objective_identity: SelectionObjectiveIdentity
    archive_id: str
    frontier_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    outcome: ParetoSelectionOutcome
    objective_terms: tuple[SelectionObjectiveTerm, ...]
    hard_constraint_status: tuple[SelectionConstraintStatus, ...]
    candidate_scores: tuple[SelectionCandidateScore, ...]
    alternatives: tuple[ParetoAlternative, ...]
    pareto: ParetoAlternativesExplanation
    decision_replay: DecisionReplay
    approval_id: str | None
    approval_provenance: Mapping[str, object]
    evidence_query: Mapping[str, object] | None
    selection_rule: str
    rationale: str
    provenance: FeasibilityProvenance
    schema_version: int = PARETO_SELECTION_EXPLANATION_SCHEMA_VERSION
    resource_usage_output_bytes: int = 0
    policy_decision: PolicyDecision | None = None

    def __post_init__(self) -> None:
        _text(self.contract_id, "selection contract ID")
        if self.schema_version != PARETO_SELECTION_EXPLANATION_SCHEMA_VERSION:
            raise ParetoSelectionError("unsupported selection explanation schema")
        _text(self.archive_id, "selection archive ID")
        _sorted_unique(self.frontier_candidate_ids, "selection frontier IDs")
        alternative_ids = tuple(item.candidate_id for item in self.alternatives)
        if alternative_ids != tuple(sorted(set(alternative_ids))):
            raise ParetoSelectionError("selection alternatives must be sorted and unique")
        if self.frontier_candidate_ids != self.pareto.frontier_candidate_ids:
            raise ParetoSelectionError("selection and Pareto frontiers differ")
        if self.pareto.archive_id != self.archive_id:
            raise ParetoSelectionError("selection and Pareto archives differ")
        if self.pareto.contract_id != self.contract_id:
            raise ParetoSelectionError("selection and Pareto contracts differ")
        if self.selected_candidate_id is not None and self.selected_candidate_id not in set(
            self.frontier_candidate_ids
        ):
            raise ParetoSelectionError(
                "selected candidate is outside the measured feasible frontier"
            )
        if (
            self.outcome is ParetoSelectionOutcome.SELECTED
            and self.selected_candidate_id is None
        ):
            raise ParetoSelectionError("selected outcome requires a selected candidate")
        if (
            self.outcome is ParetoSelectionOutcome.NO_FEASIBLE_FRONTIER
            and self.selected_candidate_id
        ):
            raise ParetoSelectionError("no-frontier outcome cannot have a selection")
        if self.approval_id is not None:
            _text(self.approval_id, "selection approval ID")
        _text(self.selection_rule, "selection rule")
        _text(self.rationale, "selection rationale")
        if self.resource_usage_output_bytes < 0:
            raise ParetoSelectionError("selection output usage cannot be negative")

    def _record_without_id(self) -> dict[str, object]:
        return {
            "record_type": "measured_pareto_selection_explanation",
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "objective_identity": self.objective_identity.to_record(),
            "archive_id": self.archive_id,
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "outcome": self.outcome.value,
            # Hard constraints intentionally precede soft objectives.
            "objective_terms": [item.to_record() for item in self.objective_terms],
            "hard_constraint_status": [item.to_record() for item in self.hard_constraint_status],
            "candidate_scores": [item.to_record() for item in self.candidate_scores],
            "alternatives": [item.to_record() for item in self.alternatives],
            "pareto": self.pareto.to_record(),
            "decision_replay": self.decision_replay.to_record(),
            "approval_id": self.approval_id,
            "approval_provenance": dict(self.approval_provenance),
            "evidence_query": None if self.evidence_query is None else dict(self.evidence_query),
            "selection_rule": self.selection_rule,
            "rationale": self.rationale,
            "provenance": self.provenance.to_record(),
            "resource_usage_output_bytes": self.resource_usage_output_bytes,
            "policy_decision": (
                None if self.policy_decision is None else self.policy_decision.to_record()
            ),
        }

    @property
    def result_id(self) -> str:
        return "pareto_selection_" + hashlib.sha256(
            canonical_identity_json(self._record_without_id()).encode("utf-8")
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        record = self._record_without_id()
        record["result_id"] = self.result_id
        return record

    def direct_report(self) -> dict[str, object]:
        """Return the canonical direct representation."""

        return self.to_record()

    def chat_text(self) -> str:
        """Render the same evidence-backed facts as a deterministic summary."""

        lines = [
            "Measured Pareto selection explanation",
            f"Outcome: {self.outcome.value}",
            f"Objective: {self.objective_identity.effective_objective_id}",
            "Hard constraints:",
        ]
        if self.hard_constraint_status:
            lines.extend(
                f"- {item.metric} {item.direction} {item.threshold:g}: "
                f"{'passed' if item.passed else 'failed'} "
                f"(observed={item.observed if item.observed is not None else 'missing'})"
                + ("; uncertain" if item.uncertain else "")
                for item in self.hard_constraint_status
            )
        else:
            lines.append("- none")
        lines.append("Soft trade-offs:")
        lines.extend(
            f"- {item.metric} {item.direction} ({item.unit})"
            + (f", weight={item.weight:g}" if item.weight is not None else "")
            for item in self.objective_terms
            if item.kind == "soft_objective"
        )
        lines.append("Frontier: " + (", ".join(self.frontier_candidate_ids) or "none"))
        lines.append("Selected: " + (self.selected_candidate_id or "none"))
        for item in self.candidate_scores:
            lines.append(
                f"- {item.candidate_id}: status={item.status.value}; "
                f"frontier={'yes' if item.on_frontier else 'no'}; "
                f"rank={item.rank if item.rank is not None else 'n/a'}"
                + ("; uncertain" if item.uncertain else "")
            )
        lines.append("Approval: " + (self.approval_id or "none"))
        if self.objective_identity.amendment_id:
            lines.append(
                "Amendment: "
                f"{self.objective_identity.amendment_id} -> "
                f"{self.objective_identity.effective_objective_id}"
            )
        lines.append(f"Rationale: {self.rationale}")
        lines.append(f"Result: {self.result_id}")
        return "\n".join(lines)


def _objective_terms(contract: ObjectiveContract) -> tuple[SelectionObjectiveTerm, ...]:
    hard = tuple(
        SelectionObjectiveTerm(
            "hard_constraint",
            item.metric,
            item.direction.value,
            item.unit.value,
            threshold=item.threshold,
            baseline=None,
        )
        for item in contract.constraints
    )
    soft = tuple(
        SelectionObjectiveTerm(
            "soft_objective",
            item.metric,
            item.direction.value,
            item.unit.value,
            weight=item.weight,
            normalization=item.normalization.value,
            baseline=item.baseline,
            minimum=item.minimum,
            maximum=item.maximum,
        )
        for item in contract.objectives
    )
    return hard + soft


def _constraint_status(
    candidate: FeasibilityCandidateEvidence | None,
    contract: ObjectiveContract,
) -> tuple[SelectionConstraintStatus, ...]:
    if candidate is None:
        return tuple(
            SelectionConstraintStatus(
                item.metric,
                item.direction.value,
                item.threshold,
                item.unit.value,
                None,
                False,
                False,
            )
            for item in contract.constraints
        )
    observations = {item.metric: item for item in candidate.observations}
    result: list[SelectionConstraintStatus] = []
    for item in contract.constraints:
        observation = observations.get(item.metric)
        if observation is None:
            result.append(
                SelectionConstraintStatus(
                    item.metric,
                    item.direction.value,
                    item.threshold,
                    item.unit.value,
                    None,
                    False,
                    False,
                )
            )
            continue
        observed = observation.conservative(item.direction)
        passed = (
            observed >= item.threshold
            if item.direction is ConstraintDirection.MINIMUM
            else observed <= item.threshold
        )
        result.append(
            SelectionConstraintStatus(
                item.metric,
                item.direction.value,
                item.threshold,
                item.unit.value,
                observed,
                passed,
                observation.lower is not None or observation.upper is not None,
            )
        )
    return tuple(result)


def _selection_records(
    scores: Sequence[SelectionCandidateScore],
    alternatives: Sequence[ParetoAlternative],
) -> tuple[dict[str, object], ...]:
    by_id = {item.candidate_id: item for item in scores}
    records: list[dict[str, object]] = []
    for alternative in alternatives:
        score = by_id[alternative.candidate_id]
        records.append(
            {
                "candidate_id": alternative.candidate_id,
                "measured": True,
                "complete": not alternative.missing_metrics,
                "constraints_passed": not alternative.constraint_violations,
                # v2.0 replay minimizes score; rank is the canonical tie-safe
                # scalar for lexicographic and Pareto modes.
                "score": (
                    float(-score.score)
                    if score.score is not None
                    else float(score.rank if score.rank is not None else 10**9)
                ),
                "objective_values": (
                    None
                    if score.objective_values is None
                    else list(score.objective_values)
                ),
            }
        )
    return tuple(records)


def _rank_frontier(
    contract: ObjectiveContract,
    archive: CanonicalEvidenceArchive,
    pareto: ParetoAlternativesExplanation,
) -> tuple[tuple[SelectionCandidateScore, ...], str, str | None]:
    candidates = {item.candidate_id: item for item in archive.candidates}
    frontier = set(pareto.frontier_candidate_ids)
    computed: list[tuple[str, tuple[float, ...], float | None, ParetoAlternative]] = []
    by_alternative = {item.candidate_id: item for item in pareto.alternatives}
    for candidate_id in sorted(frontier):
        candidate = candidates[candidate_id]
        evaluation = evaluate_contract(contract, candidate.observations)
        if not evaluation.feasible or not evaluation.comparable:
            raise ParetoSelectionError(
                f"frontier candidate {candidate_id} has no canonical comparable objective evidence"
            )
        assert evaluation.objective_values is not None
        computed.append(
            (
                candidate_id,
                evaluation.objective_values,
                evaluation.score,
                by_alternative[candidate_id],
            )
        )

    if contract.mode is ObjectiveMode.WEIGHTED:
        ordered = sorted(
            computed,
            key=lambda item: (
                -(item[2] if item[2] is not None else float("-inf")),
                item[0],
            ),
        )
        rule = "weighted objective score descending; candidate ID ascending on ties"
    else:
        ordered = sorted(
            computed,
            key=lambda item: (*tuple(-value for value in item[1]), item[0]),
        )
        rule = (
            "lexicographic normalized objective values descending; candidate ID ascending on ties"
            if contract.mode is ObjectiveMode.LEXICOGRAPHIC
            else "canonical normalized objective-vector order; candidate ID ascending on ties"
        )

    selected = ordered[0][0] if ordered else None
    scores: list[SelectionCandidateScore] = []
    for rank, (candidate_id, values, score, alternative) in enumerate(ordered):
        scores.append(
            SelectionCandidateScore(
                candidate_id,
                alternative.evidence_id,
                values,
                score,
                rank,
                True,
                alternative.uncertain,
                alternative.status,
            )
        )
    score_by_id = {item.candidate_id: item for item in scores}
    all_scores = [
        score_by_id.get(
            alternative.candidate_id,
            SelectionCandidateScore(
                alternative.candidate_id,
                alternative.evidence_id,
                None,
                None,
                None,
                False,
                alternative.uncertain,
                alternative.status,
            ),
        )
        for alternative in pareto.alternatives
    ]
    return tuple(all_scores), rule, selected


def _amendment_identity(
    contract: ObjectiveContract,
    amendment: object | None,
) -> tuple[SelectionObjectiveIdentity, str | None, dict[str, object]]:
    if amendment is None:
        return (
            SelectionObjectiveIdentity(contract.contract_id, contract.contract_id),
            None,
            {},
        )
    record = amendment.to_record() if hasattr(amendment, "to_record") else amendment
    if not isinstance(record, Mapping):
        raise ParetoSelectionError("objective amendment must be a canonical record")
    effective = getattr(amendment, "effective_spec_identity", None)
    if not isinstance(effective, str):
        effective = cast(str | None, record.get("effective_spec_identity"))
    if effective != contract.contract_id:
        raise ParetoSelectionError(
            "selection contract does not match the amended objective identity"
        )
    status = getattr(amendment, "status", None)
    status_value = getattr(status, "value", status)
    if status_value in {"pending", "rejected", "expired", "cancelled"}:
        raise ParetoSelectionError(
            "selection cannot use an unapplied or unapproved objective amendment"
        )
    amendment_id = getattr(amendment, "amendment_id", record.get("amendment_id"))
    diff_id = getattr(amendment, "diff_id", record.get("diff_id"))
    approval_id = getattr(amendment, "approval_id", record.get("approval_id"))
    application = record.get("application")
    downstream = None
    if isinstance(application, Mapping):
        downstream = application.get("downstream_campaign_id")
    if downstream is None:
        downstream = record.get("downstream_campaign_id")
    if not isinstance(amendment_id, str) or not isinstance(diff_id, str):
        raise ParetoSelectionError("amended objective identity is incomplete")
    original_id = getattr(amendment, "original_spec_identity", record.get("original_spec_identity"))
    if not isinstance(original_id, str):
        raise ParetoSelectionError("amended objective original identity is incomplete")
    approval = (
        dict(amendment.approval_provenance)
        if hasattr(amendment, "approval_provenance")
        else {}
    )
    if not approval and isinstance(record.get("approval_provenance"), Mapping):
        approval = dict(record["approval_provenance"])
    if isinstance(approval_id, str):
        approval["approval_id"] = approval_id
    return (
        SelectionObjectiveIdentity(
            original_id,
            contract.contract_id,
            amendment_id,
            diff_id,
            None if status_value is None else str(status_value),
            None if downstream is None else str(downstream),
        ),
        None if approval_id is None else str(approval_id),
        approval,
    )


def _query_record(response: EvidenceQueryResponse | None) -> Mapping[str, object] | None:
    if response is None:
        return None
    return {
        "query_id": response.query.query_id,
        "response_digest": response.response_digest,
        "snapshot_id": response.snapshot.snapshot_id,
        "snapshot_digest": response.snapshot.snapshot_digest,
        "status": response.status.value,
        "record_evidence_ids": [item.evidence_id for item in response.records],
        "missing_fields": list(response.missing_fields),
        "unavailable_fields": list(response.unavailable_fields),
    }


def _rationale(
    *,
    outcome: ParetoSelectionOutcome,
    selected: str | None,
    contract: ObjectiveContract,
    frontier: tuple[str, ...],
    scores: Sequence[SelectionCandidateScore],
    alternatives: Sequence[ParetoAlternative],
    identity: SelectionObjectiveIdentity,
    approval_id: str | None,
    selected_via_final_evidence: bool,
) -> str:
    if selected is None:
        return (
            f"No candidate was selected because the measured evidence produced no feasible "
            f"frontier under objective {identity.effective_objective_id}; hard constraints "
            f"remain binding and no soft trade-off was applied. "
            f"Frontier={','.join(frontier) or 'none'}."
        )
    selected_score = next(item for item in scores if item.candidate_id == selected)
    selected_alternative = next(item for item in alternatives if item.candidate_id == selected)
    uncertainty = (
        "uncertainty is retained"
        if selected_score.uncertain
        else "no interval uncertainty is recorded"
    )
    mode = contract.mode.value
    approval = approval_id or "none"
    frontier_ties = tuple(
        sorted(
            {
                tie
                for item in alternatives
                if item.on_frontier
                for tie in item.tied_candidate_ids
            }
        )
    )
    dominated = tuple(
        sorted(
            item.candidate_id
            for item in alternatives
            if item.status is ParetoAlternativeStatus.DOMINATED
        )
    )
    frontier_context = (
        f" tied frontier candidates retained={','.join(frontier_ties) or 'none'}; "
        f"dominated alternatives retained={','.join(dominated) or 'none'}."
    )
    selection_basis = (
        "the canonical v2.0 final-selection evidence identifies it and the objective, "
        "hard-constraint, and frontier checks reproduce that decision"
        if selected_via_final_evidence
        else "the canonical objective values ranked it first under the declared selection rule"
    )
    relation = ""
    if selected_alternative.tied_candidate_ids:
        relation += (
            f" It is tied with {','.join(selected_alternative.tied_candidate_ids)}; "
            "the candidate ID tie-break is deterministic."
        )
    if selected_alternative.dominated_by_candidate_ids:
        relation += (
            f" It is dominated by {','.join(selected_alternative.dominated_by_candidate_ids)}, "
            "so it cannot be selected."
        )
    return (
        f"Selected measured candidate {selected} from the feasible Pareto frontier "
        f"under objective {identity.effective_objective_id} ({mode}); hard constraints "
        f"were satisfied before soft trade-offs, {selection_basis}, and {uncertainty}.{relation} "
        f"Frontier={','.join(frontier)}; approval={approval}.{frontier_context}"
    )


def build_pareto_selection_explanation(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    selected_candidate_id: str | None = None,
    final_selection_evidence: Sequence[Mapping[str, object]] | None = None,
    amendment: object | None = None,
    evidence_query: EvidenceQueryResponse | None = None,
    resource_bounds: ParetoResourceBounds = _DEFAULT_PARETO_SELECTION_RESOURCE_BOUNDS,
) -> ParetoSelectionExplanation:
    """Build a deterministic final-selection explanation from measured evidence."""

    if isinstance(evidence, CanonicalEvidenceArchive):
        archive = evidence
    else:
        archive = CanonicalEvidenceArchive.build(contract, evidence)
    if archive.contract_id != contract.contract_id:
        raise ParetoSelectionError("selection evidence does not match the objective contract")
    pareto = build_pareto_alternatives(
        contract,
        archive,
        provenance=provenance,
        resource_bounds=resource_bounds,
    )
    identity, amendment_approval_id, amendment_approval = _amendment_identity(contract, amendment)
    approval_id = amendment_approval_id or provenance.approval_id
    approval_provenance = dict(provenance.approval_provenance)
    approval_provenance.update(amendment_approval)
    scores, rule, canonical_selected = _rank_frontier(contract, archive, pareto)
    selection_records = _selection_records(scores, pareto.alternatives)
    replay = replay_decision_evidence(selection_records)
    selected = canonical_selected
    selected_via_final_evidence = False
    if final_selection_evidence is not None:
        external_replay = replay_decision_evidence(final_selection_evidence)
        selected_from_evidence = external_replay.selected_candidate_id
        if selected_from_evidence is None:
            raise ParetoSelectionError("final selection evidence contains no selected candidate")
        if selected_from_evidence not in set(pareto.frontier_candidate_ids):
            raise ParetoSelectionError(
                "final selection evidence selects a candidate outside the measured "
                "feasible frontier"
            )
        if selected_candidate_id is not None and selected_candidate_id != selected_from_evidence:
            raise ParetoSelectionError(
                "requested selection disagrees with the canonical final-selection evidence"
            )
        selected = selected_from_evidence
        selected_via_final_evidence = True
        replay = external_replay
    elif selected_candidate_id is not None:
        if selected_candidate_id != canonical_selected:
            raise ParetoSelectionError(
                "requested selection does not reproduce from the canonical objective and frontier"
            )
        selected = selected_candidate_id
    if not selected_via_final_evidence and replay.selected_candidate_id != selected:
        raise ParetoSelectionError("v2.0 decision replay did not reproduce the canonical selection")
    if selected_via_final_evidence:
        rule = f"{rule}; validated by v2.0 final-selection evidence"
    outcome = (
        ParetoSelectionOutcome.SELECTED
        if selected is not None
        else ParetoSelectionOutcome.NO_FEASIBLE_FRONTIER
    )
    candidate_map = {item.candidate_id: item for item in archive.candidates}
    selected_evidence = candidate_map.get(selected) if selected else None
    constraint_status = _constraint_status(selected_evidence, contract)
    rationale = _rationale(
        outcome=outcome,
        selected=selected,
        contract=contract,
        frontier=pareto.frontier_candidate_ids,
        scores=scores,
        alternatives=pareto.alternatives,
        identity=identity,
        approval_id=approval_id,
        selected_via_final_evidence=selected_via_final_evidence,
    )
    policy_decision = resolve_policy(
        "pareto-explanation",
        (
            PolicyCandidate(
                PolicySource.HARD_CONSTRAINTS,
                PolicyOutcome.ALLOW if selected is not None else PolicyOutcome.DENY,
                "measured hard-constraint qualification is binding before soft trade-offs",
            ),
            PolicyCandidate(
                PolicySource.VALIDATED_SPEC,
                PolicyOutcome.ALLOW,
                "the explanation is bound to the validated objective contract",
            ),
            PolicyCandidate(
                PolicySource.APPROVAL_POLICY,
                PolicyOutcome.ALLOW,
                "approval metadata is retained but cannot change measured qualification",
            ),
            PolicyCandidate(
                PolicySource.TOOL_CAPABILITY,
                PolicyOutcome.ALLOW,
                "only canonical evidence reached the explanation boundary",
            ),
            PolicyCandidate(
                PolicySource.EVIDENCE_STATUS,
                PolicyOutcome.ALLOW if selected is not None else PolicyOutcome.UNKNOWN,
                "evidence status is explicit and incomplete evidence is not promoted",
            ),
            PolicyCandidate(
                PolicySource.PROMPT,
                PolicyOutcome.ALLOW,
                "prompt wording cannot select an unqualified candidate",
            ),
            PolicyCandidate(
                PolicySource.PROVIDER,
                PolicyOutcome.ALLOW,
                "provider text cannot override canonical evidence",
            ),
        ),
    )
    result = ParetoSelectionExplanation(
        contract.contract_id,
        identity,
        archive.archive_id,
        pareto.frontier_candidate_ids,
        selected,
        outcome,
        _objective_terms(contract),
        constraint_status,
        scores,
        pareto.alternatives,
        pareto,
        replay,
        approval_id,
        approval_provenance,
        _query_record(evidence_query),
        rule,
        rationale,
        pareto.provenance,
        policy_decision=policy_decision,
    )
    for _ in range(3):
        output_bytes = len(canonical_identity_json(result.to_record()).encode("utf-8"))
        if output_bytes == result.resource_usage_output_bytes:
            break
        result = replace(result, resource_usage_output_bytes=output_bytes)
    if result.resource_usage_output_bytes > resource_bounds.max_output_bytes:
        raise ParetoSelectionError("selection explanation exceeds the declared output bound")
    return result


def replay_pareto_selection_explanation(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    expected_report: Mapping[str, object] | None = None,
    selected_candidate_id: str | None = None,
    final_selection_evidence: Sequence[Mapping[str, object]] | None = None,
    amendment: object | None = None,
    evidence_query: EvidenceQueryResponse | None = None,
    resource_bounds: ParetoResourceBounds = _DEFAULT_PARETO_SELECTION_RESOURCE_BOUNDS,
) -> ParetoSelectionExplanation:
    """Rebuild a selection and optionally assert exact direct-report parity."""

    result = build_pareto_selection_explanation(
        contract,
        evidence,
        provenance=provenance,
        selected_candidate_id=selected_candidate_id,
        final_selection_evidence=final_selection_evidence,
        amendment=amendment,
        evidence_query=evidence_query,
        resource_bounds=resource_bounds,
    )
    if expected_report is not None and canonical_identity_json(
        result.to_record()
    ) != canonical_identity_json(expected_report):
        raise ParetoSelectionError("selection replay differs from the canonical direct report")
    return result


def render_pareto_selection_html(
    explanation: ParetoSelectionExplanation,
    *,
    title: str = "Measured Pareto selection",
) -> str:
    """Render a dependency-free HTML view from the canonical selection."""

    _text(title, "selection HTML title")
    terms = "".join(
        f"<li>{html.escape(item.kind)}: {html.escape(item.metric)} "
        f"{html.escape(item.direction)} {item.threshold if item.threshold is not None else ''}</li>"
        for item in explanation.objective_terms
    )
    rows = "".join(
        f"<tr><th>{html.escape(item.candidate_id)}</th>"
        f"<td>{html.escape(item.status.value)}</td>"
        f"<td>{'yes' if item.on_frontier else 'no'}</td>"
        f"<td>{item.rank if item.rank is not None else 'n/a'}</td>"
        f"<td>{'yes' if item.uncertain else 'no'}</td></tr>"
        for item in explanation.candidate_scores
    )
    data = canonical_identity_json(explanation.to_record()).replace("<", "\\u003c")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p>Objective: <code>"
        f"{html.escape(explanation.objective_identity.effective_objective_id)}</code>"
        f"; selected: <code>{html.escape(explanation.selected_candidate_id or 'none')}</code></p>"
        f"<h2>Hard constraints before soft objectives</h2><ol>{terms}</ol>"
        "<h2>Candidate selection</h2><table><thead><tr>"
        "<th>Candidate</th><th>Status</th><th>Frontier</th><th>Rank</th><th>Uncertain</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        f"<p>{html.escape(explanation.rationale)}</p>"
        f"<script type='application/json' id='modelsurgeon-selection-data'>{data}</script>"
        "</body></html>"
    )


def render_pareto_selection_explanation(
    explanation: ParetoSelectionExplanation,
    *,
    format: str = "direct",
) -> dict[str, object] | str:
    """Render direct, chat, or HTML from exactly one canonical explanation."""

    if format == "direct":
        return explanation.to_record()
    if format == "chat":
        return explanation.chat_text()
    if format == "html":
        return render_pareto_selection_html(explanation)
    raise ParetoSelectionError(f"unsupported selection render format: {format}")


# Verb-led and issue-vocabulary aliases keep the public surface discoverable.
build_selection_explanation = build_pareto_selection_explanation
explain_pareto_selection = build_pareto_selection_explanation
replay_selection_explanation = replay_pareto_selection_explanation
render_selection_explanation = render_pareto_selection_explanation


__all__ = [
    "PARETO_SELECTION_EXPLANATION_SCHEMA_VERSION",
    "ParetoSelectionError",
    "ParetoSelectionExplanation",
    "ParetoSelectionOutcome",
    "SelectionCandidateScore",
    "SelectionConstraintStatus",
    "SelectionObjectiveIdentity",
    "SelectionObjectiveTerm",
    "build_pareto_selection_explanation",
    "build_selection_explanation",
    "explain_pareto_selection",
    "render_pareto_selection_explanation",
    "render_pareto_selection_html",
    "render_selection_explanation",
    "replay_pareto_selection_explanation",
    "replay_selection_explanation",
]
