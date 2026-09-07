"""Approval, promotion, and deterministic replanning around campaign execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from modelsurgeon.surgery.contracts import TransactionState

COORDINATOR_SCHEMA_VERSION = 1


class CoordinatorError(ValueError):
    """Raised when a campaign plan or promotion evidence is unsafe."""


class PromotionOutcome(StrEnum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoordinatorError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise CoordinatorError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise CoordinatorError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ApprovedCampaignPlan:
    """Immutable approval boundary for a candidate campaign."""

    plan_id: str
    source_artifact_digest: str
    objective_contract_id: str
    candidate_space_id: str
    candidate_ids: tuple[str, ...]
    seed: int
    evaluation_budget: int
    approval_id: str
    schema_version: int = COORDINATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.plan_id, "campaign plan ID")
        _digest(self.source_artifact_digest, "source artifact digest")
        _text(self.objective_contract_id, "objective contract ID")
        _text(self.candidate_space_id, "candidate-space ID")
        _text(self.approval_id, "approval ID")
        if not self.candidate_ids or any(
            not item.startswith("candidate_") for item in self.candidate_ids
        ):
            raise CoordinatorError("approved campaigns require canonical candidate IDs")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise CoordinatorError("approved candidate IDs must be unique")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise CoordinatorError("campaign seed must be an unsigned 64-bit integer")
        _positive(self.evaluation_budget, "evaluation budget")
        if self.schema_version != COORDINATOR_SCHEMA_VERSION:
            raise CoordinatorError("unsupported coordinator schema")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_artifact_digest": self.source_artifact_digest,
            "objective_contract_id": self.objective_contract_id,
            "candidate_space_id": self.candidate_space_id,
            "candidate_ids": list(self.candidate_ids),
            "seed": self.seed,
            "evaluation_budget": self.evaluation_budget,
            "approval_id": self.approval_id,
        }


@dataclass(frozen=True, slots=True)
class MeasuredCandidateEvidence:
    """Evidence required before an immutable candidate can be promoted."""

    candidate_id: str
    candidate_state_id: str
    evaluation_id: str
    artifact_digest: str | None
    source_artifact_digest: str
    measured: bool
    complete: bool
    constraints_passed: bool
    transaction_state: TransactionState
    artifact_immutable: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("candidate_"):
            raise CoordinatorError("candidate evidence requires a canonical candidate ID")
        if not self.candidate_state_id.startswith("state_"):
            raise CoordinatorError("candidate evidence requires a canonical state ID")
        if not self.evaluation_id.startswith("evaluation_"):
            raise CoordinatorError("candidate evidence requires a canonical evaluation ID")
        _digest(self.source_artifact_digest, "evidence source digest")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "evidence artifact digest")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    candidate_id: str
    outcome: PromotionOutcome
    promoted: bool
    reason: str
    decision_id: str
    evaluation_id: str
    artifact_digest: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": COORDINATOR_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "outcome": self.outcome.value,
            "promoted": self.promoted,
            "reason": self.reason,
            "decision_id": self.decision_id,
            "evaluation_id": self.evaluation_id,
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True, slots=True)
class ReplanResult:
    parent_plan_id: str
    next_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    next_plan_id: str
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": COORDINATOR_SCHEMA_VERSION,
            "parent_plan_id": self.parent_plan_id,
            "next_candidate_ids": list(self.next_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "next_plan_id": self.next_plan_id,
            "reasons": list(self.reasons),
        }


T_co = TypeVar("T_co", covariant=True)


class CampaignRunnerProtocol(Protocol[T_co]):
    def run(self) -> T_co: ...


class AutonomousCampaignCoordinator:
    """Gate runner output and promotion without duplicating CampaignRunner work."""

    def __init__(self, plan: ApprovedCampaignPlan) -> None:
        self.plan = plan
        self._decisions: dict[str, PromotionDecision] = {}
        self._promoted_artifacts: set[str] = set()

    def run_approved(self, runner: CampaignRunnerProtocol[T_co]) -> T_co:
        """Run only an approved plan through an existing lease-aware CampaignRunner."""

        return runner.run()

    def promote(self, evidence: MeasuredCandidateEvidence) -> PromotionDecision:
        if evidence.candidate_id not in self.plan.candidate_ids:
            return self._decision(
                evidence,
                PromotionOutcome.FAILED,
                "candidate is not in the approved plan",
            )
        existing = self._decisions.get(evidence.candidate_id)
        if existing is not None:
            return existing
        if not evidence.measured or not evidence.complete:
            return self._decision(
                evidence,
                PromotionOutcome.UNKNOWN,
                "predicted or partially evaluated states cannot be promoted",
            )
        if not evidence.constraints_passed:
            return self._decision(
                evidence,
                PromotionOutcome.REJECTED,
                "measured hard constraints did not pass",
            )
        if evidence.transaction_state is not TransactionState.COMMITTED:
            return self._decision(
                evidence,
                PromotionOutcome.REJECTED,
                "candidate transaction is not independently committed",
            )
        if not evidence.artifact_immutable or evidence.artifact_digest is None:
            return self._decision(
                evidence,
                PromotionOutcome.FAILED,
                "immutable child artifact evidence is missing",
            )
        if evidence.source_artifact_digest != self.plan.source_artifact_digest:
            return self._decision(
                evidence,
                PromotionOutcome.FAILED,
                "source artifact changed during campaign",
            )
        if evidence.artifact_digest == self.plan.source_artifact_digest:
            return self._decision(
                evidence,
                PromotionOutcome.FAILED,
                "candidate artifact must be a distinct immutable child",
            )
        if evidence.artifact_digest in self._promoted_artifacts:
            return self._decision(
                evidence,
                PromotionOutcome.FAILED,
                "artifact was already promoted by another candidate",
            )
        decision = self._decision(
            evidence,
            PromotionOutcome.PROMOTED,
            "measured candidate passed promotion gate",
        )
        self._promoted_artifacts.add(evidence.artifact_digest)
        return decision

    def _decision(
        self,
        evidence: MeasuredCandidateEvidence,
        outcome: PromotionOutcome,
        reason: str,
    ) -> PromotionDecision:
        payload = {
            "plan_id": self.plan.plan_id,
            "candidate_id": evidence.candidate_id,
            "evaluation_id": evidence.evaluation_id,
            "artifact_digest": evidence.artifact_digest,
            "outcome": outcome.value,
            "reason": reason,
        }
        decision_id = f"promotion_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"
        decision = PromotionDecision(
            evidence.candidate_id,
            outcome,
            outcome is PromotionOutcome.PROMOTED,
            reason,
            decision_id,
            evidence.evaluation_id,
            evidence.artifact_digest,
        )
        self._decisions[evidence.candidate_id] = decision
        return decision

    def replan(self, *, candidate_ids: Iterable[str] | None = None) -> ReplanResult:
        pending = tuple(candidate_ids or self.plan.candidate_ids)
        if any(candidate not in self.plan.candidate_ids for candidate in pending):
            raise CoordinatorError("replan candidates must come from the approved plan")
        rejected = tuple(
            sorted(
                candidate
                for candidate, decision in self._decisions.items()
                if decision.outcome in {PromotionOutcome.REJECTED, PromotionOutcome.FAILED}
            )
        )
        next_candidates = tuple(candidate for candidate in pending if candidate not in rejected)
        payload = {
            "parent_plan_id": self.plan.plan_id,
            "next_candidate_ids": list(next_candidates),
            "decisions": [self._decisions[key].to_record() for key in sorted(self._decisions)],
        }
        next_plan_id = f"campaign_plan_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"
        return ReplanResult(
            self.plan.plan_id,
            next_candidates,
            rejected,
            next_plan_id,
            ("replan excludes measured rejected/failed candidates and retains unknowns",),
        )


__all__ = [
    "COORDINATOR_SCHEMA_VERSION",
    "ApprovedCampaignPlan",
    "AutonomousCampaignCoordinator",
    "CampaignRunnerProtocol",
    "CoordinatorError",
    "MeasuredCandidateEvidence",
    "PromotionDecision",
    "PromotionOutcome",
    "ReplanResult",
]
