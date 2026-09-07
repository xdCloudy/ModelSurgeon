"""Interaction-aware acquisition, stale-state safety, and bounded replanning."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from .budgets import ExperimentResources

INTERACTION_ACQUISITION_SCHEMA_VERSION: Final[int] = 1
INTERACTION_ACQUISITION_PROTOCOL_REVISION: Final[str] = "interaction-acquisition-v1"


class InteractionPlanningError(ValueError):
    """Raised when a candidate plan or outcome violates state lineage."""


class InteractionOutcomeStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SURPRISING = "surprising"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class InteractionAction(StrEnum):
    CONTINUE = "continue"
    REPLAN = "replan"
    ROLLBACK = "rollback"


class InteractionPlanState(StrEnum):
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"


class InteractionSelectionReason(StrEnum):
    UTILITY = "utility"
    UNCERTAINTY = "uncertainty"
    INTERACTION = "interaction"
    DIVERSITY = "diversity"
    BUDGET_FILL = "budget-fill"


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InteractionPlanningError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InteractionPlanningError(f"{label} must be finite")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractionPlanningError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _add_resources(left: ExperimentResources, right: ExperimentResources) -> ExperimentResources:
    return ExperimentResources(
        left.wall_seconds + right.wall_seconds,
        left.tier_cost + right.tier_cost,
        left.gpu_seconds + right.gpu_seconds,
        left.disk_bytes + right.disk_bytes,
    )


@dataclass(frozen=True, slots=True)
class InteractionAcquisitionConfig:
    """Weights and hard bounds shared by every acquisition/replan decision."""

    max_candidates: int = 64
    max_replans: int = 8
    utility_weight: float = 0.35
    safety_weight: float = 0.25
    uncertainty_weight: float = 0.2
    interaction_weight: float = 0.15
    diversity_weight: float = 0.05
    surprise_threshold: float = 0.25
    seed: int = 0
    max_resources: ExperimentResources = field(
        default_factory=lambda: ExperimentResources(
            wall_seconds=3600.0,
            tier_cost=100.0,
            gpu_seconds=1800.0,
            disk_bytes=10_000_000_000,
        )
    )

    def __post_init__(self) -> None:
        if self.max_candidates <= 0 or self.max_replans < 0:
            raise InteractionPlanningError("interaction candidate/replan bounds are invalid")
        weights = (
            self.utility_weight,
            self.safety_weight,
            self.uncertainty_weight,
            self.interaction_weight,
            self.diversity_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise InteractionPlanningError("interaction weights must be finite and non-negative")
        if math.isclose(math.fsum(weights), 0.0, abs_tol=1e-12):
            raise InteractionPlanningError("interaction weights cannot all be zero")
        if not math.isfinite(self.surprise_threshold) or self.surprise_threshold < 0.0:
            raise InteractionPlanningError("interaction surprise threshold is invalid")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise InteractionPlanningError("interaction seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "max_candidates": self.max_candidates,
            "max_replans": self.max_replans,
            "weights": {
                "utility": self.utility_weight,
                "safety": self.safety_weight,
                "uncertainty": self.uncertainty_weight,
                "interaction": self.interaction_weight,
                "diversity": self.diversity_weight,
            },
            "surprise_threshold": self.surprise_threshold,
            "seed": self.seed,
            "max_resources": self.max_resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InteractionCandidate:
    """A candidate compiled for one exact parent state revision."""

    candidate_id: str
    parent_state_id: str
    candidate_revision: str
    utility: float
    safe_probability: float
    uncertainty: float
    interaction_uncertainty: float
    diversity: float
    resources: ExperimentResources = field(default_factory=ExperimentResources)

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise InteractionPlanningError("interaction candidates require canonical IDs")
        for label, value in (
            ("parent state ID", self.parent_state_id),
            ("candidate revision", self.candidate_revision),
        ):
            _text(value, label)
        values = (
            self.utility,
            self.safe_probability,
            self.uncertainty,
            self.interaction_uncertainty,
            self.diversity,
        )
        if any(not math.isfinite(value) for value in values):
            raise InteractionPlanningError("interaction candidate values must be finite")
        if not 0.0 <= self.safe_probability <= 1.0:
            raise InteractionPlanningError("safe probability must be within [0, 1]")
        if any(
            value < 0.0
            for value in (self.uncertainty, self.interaction_uncertainty, self.diversity)
        ):
            raise InteractionPlanningError("uncertainty and diversity cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_state_id": self.parent_state_id,
            "candidate_revision": self.candidate_revision,
            "utility": self.utility,
            "safe_probability": self.safe_probability,
            "uncertainty": self.uncertainty,
            "interaction_uncertainty": self.interaction_uncertainty,
            "diversity": self.diversity,
            "resources": self.resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InteractionSelection:
    candidate_id: str
    parent_state_id: str
    rank: int
    reason: InteractionSelectionReason
    propensity: float
    policy_score: float
    resources: ExperimentResources

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_") or self.rank <= 0:
            raise InteractionPlanningError("interaction selection identity/rank is invalid")
        if not 0.0 < self.propensity <= 1.0 or not math.isfinite(self.policy_score):
            raise InteractionPlanningError("interaction selection score is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_state_id": self.parent_state_id,
            "rank": self.rank,
            "reason": self.reason.value,
            "propensity": self.propensity,
            "policy_score": self.policy_score,
            "resources": self.resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InteractionOutcome:
    """Immutable result retained even when it triggers rollback or replanning."""

    candidate_id: str
    parent_state_id: str
    resulting_state_id: str | None
    status: InteractionOutcomeStatus
    expected_utility: float
    actual_utility: float | None
    interaction_residual: float
    lineage_token: str
    resources: ExperimentResources

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise InteractionPlanningError("interaction outcome candidate ID is invalid")
        _text(self.parent_state_id, "outcome parent state ID")
        if self.resulting_state_id is not None:
            _text(self.resulting_state_id, "outcome resulting state ID")
        for label, value in (
            ("expected utility", self.expected_utility),
            ("interaction residual", self.interaction_residual),
        ):
            _finite(value, label)
        if self.actual_utility is not None:
            _finite(self.actual_utility, "actual utility")
        _text(self.lineage_token, "outcome lineage token")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_state_id": self.parent_state_id,
            "resulting_state_id": self.resulting_state_id,
            "status": self.status.value,
            "expected_utility": self.expected_utility,
            "actual_utility": self.actual_utility,
            "interaction_residual": self.interaction_residual,
            "lineage_token": self.lineage_token,
            "resources": self.resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InteractionPlan:
    plan_id: str
    current_state_id: str
    generation: int
    selections: tuple[InteractionSelection, ...]
    rejected_stale_candidate_ids: tuple[str, ...]
    history: tuple[InteractionOutcome, ...]
    consumed_resources: ExperimentResources
    state: InteractionPlanState = InteractionPlanState.ACTIVE
    schema_version: int = INTERACTION_ACQUISITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.plan_id, "interaction plan ID")
        _text(self.current_state_id, "interaction plan state ID")
        if self.generation < 0:
            raise InteractionPlanningError("interaction plan generation cannot be negative")
        ids = tuple(item.candidate_id for item in self.selections)
        if len(ids) != len(set(ids)):
            raise InteractionPlanningError("interaction plan selections must be unique")
        if any(item.parent_state_id != self.current_state_id for item in self.selections):
            raise InteractionPlanningError(
                "interaction plan contains a stale parent-state candidate"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": INTERACTION_ACQUISITION_PROTOCOL_REVISION,
            "plan_id": self.plan_id,
            "current_state_id": self.current_state_id,
            "generation": self.generation,
            "state": self.state.value,
            "selections": [item.to_record() for item in self.selections],
            "rejected_stale_candidate_ids": list(self.rejected_stale_candidate_ids),
            "history": [item.to_record() for item in self.history],
            "consumed_resources": self.consumed_resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InteractionReplanResult:
    action: InteractionAction
    plan: InteractionPlan
    reason: str
    invalidated_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.reason, "interaction decision reason")

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "invalidated_candidate_ids": list(self.invalidated_candidate_ids),
            "plan": self.plan.to_record(),
        }


def _score(candidate: InteractionCandidate, config: InteractionAcquisitionConfig) -> float:
    return (
        config.utility_weight * candidate.utility * candidate.safe_probability
        + config.safety_weight * candidate.safe_probability
        + config.uncertainty_weight * candidate.uncertainty
        + config.interaction_weight * candidate.interaction_uncertainty
        + config.diversity_weight * candidate.diversity
    )


def _reason(
    candidate: InteractionCandidate, config: InteractionAcquisitionConfig
) -> InteractionSelectionReason:
    signals = {
        InteractionSelectionReason.UTILITY: config.utility_weight
        * candidate.utility
        * candidate.safe_probability,
        InteractionSelectionReason.UNCERTAINTY: config.uncertainty_weight * candidate.uncertainty,
        InteractionSelectionReason.INTERACTION: config.interaction_weight
        * candidate.interaction_uncertainty,
        InteractionSelectionReason.DIVERSITY: config.diversity_weight * candidate.diversity,
    }
    return max(signals, key=lambda item: (signals[item], item.value))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _make_plan(
    current_state_id: str,
    candidates: Sequence[InteractionCandidate],
    *,
    generation: int,
    history: tuple[InteractionOutcome, ...],
    config: InteractionAcquisitionConfig,
    rejected_stale: tuple[str, ...] = (),
) -> InteractionPlan:
    _text(current_state_id, "current state ID")
    ids = tuple(item.candidate_id for item in candidates)
    if len(ids) != len(set(ids)):
        raise InteractionPlanningError("interaction candidate IDs must be unique")
    stale = tuple(
        sorted(
            set(rejected_stale)
            | {item.candidate_id for item in candidates if item.parent_state_id != current_state_id}
        )
    )
    eligible = tuple(item for item in candidates if item.parent_state_id == current_state_id)
    ranked = sorted(
        eligible,
        key=lambda item: (-_score(item, config), item.candidate_id),
    )
    chosen: list[InteractionSelection] = []
    consumed = ExperimentResources()
    for candidate in ranked:
        if len(chosen) >= config.max_candidates:
            break
        projected = _add_resources(consumed, candidate.resources)
        if any(
            (
                projected.wall_seconds > config.max_resources.wall_seconds,
                projected.tier_cost > config.max_resources.tier_cost,
                projected.gpu_seconds > config.max_resources.gpu_seconds,
                projected.disk_bytes > config.max_resources.disk_bytes,
            )
        ):
            continue
        chosen.append(
            InteractionSelection(
                candidate.candidate_id,
                current_state_id,
                len(chosen) + 1,
                _reason(candidate, config),
                1.0,
                _score(candidate, config),
                candidate.resources,
            )
        )
        consumed = projected
    identity = {
        "state": current_state_id,
        "generation": generation,
        "selections": [item.to_record() for item in chosen],
        "history": [item.to_record() for item in history],
    }
    return InteractionPlan(
        _digest(identity),
        current_state_id,
        generation,
        tuple(chosen),
        stale,
        history,
        consumed,
    )


def plan_interaction_acquisition(
    current_state_id: str,
    candidates: Sequence[InteractionCandidate],
    *,
    config: InteractionAcquisitionConfig | None = None,
) -> InteractionPlan:
    """Select only candidates compiled for the current state and within budget."""

    return _make_plan(
        current_state_id,
        candidates,
        generation=0,
        history=(),
        config=config or InteractionAcquisitionConfig(),
    )


def replan_after_outcome(
    plan: InteractionPlan,
    outcome: InteractionOutcome,
    next_candidates: Sequence[InteractionCandidate],
    *,
    config: InteractionAcquisitionConfig | None = None,
) -> InteractionReplanResult:
    """Apply one outcome, retaining lineage and returning continue/replan/rollback."""

    resolved = config or InteractionAcquisitionConfig()
    if plan.state is not InteractionPlanState.ACTIVE:
        raise InteractionPlanningError("cannot apply an outcome to an inactive interaction plan")
    selected = tuple(item for item in plan.selections if item.candidate_id == outcome.candidate_id)
    if len(selected) != 1:
        raise InteractionPlanningError("outcome candidate was not selected by this plan")
    if (
        outcome.parent_state_id != plan.current_state_id
        or selected[0].parent_state_id != outcome.parent_state_id
    ):
        raise InteractionPlanningError("outcome parent state is stale or mismatched")
    history = (*plan.history, outcome)
    if len(history) > resolved.max_replans + 1:
        raise InteractionPlanningError("interaction replan bound exceeded")
    surprising = (
        outcome.status is InteractionOutcomeStatus.SURPRISING
        or abs(outcome.interaction_residual) >= resolved.surprise_threshold
    )
    failed = outcome.status in {
        InteractionOutcomeStatus.REJECTED,
        InteractionOutcomeStatus.FAILED,
        InteractionOutcomeStatus.UNSUPPORTED,
    }
    invalidated = tuple(
        item.candidate_id for item in plan.selections if item.candidate_id != outcome.candidate_id
    )
    if failed and resolved.max_replans == 0:
        terminal = InteractionPlan(
            plan.plan_id,
            plan.current_state_id,
            plan.generation,
            (),
            plan.rejected_stale_candidate_ids,
            history,
            plan.consumed_resources,
            InteractionPlanState.ROLLED_BACK,
        )
        return InteractionReplanResult(
            InteractionAction.ROLLBACK, terminal, "unsafe outcome rolled back", invalidated
        )
    if surprising or failed:
        state_id = outcome.resulting_state_id or plan.current_state_id
        next_plan = _make_plan(
            state_id,
            next_candidates,
            generation=plan.generation + 1,
            history=history,
            config=resolved,
        )
        reason = (
            "surprising interaction triggered bounded replan"
            if surprising
            else "negative outcome triggered bounded replan"
        )
        return InteractionReplanResult(InteractionAction.REPLAN, next_plan, reason, invalidated)
    next_state = outcome.resulting_state_id or plan.current_state_id
    continued = _make_plan(
        next_state,
        next_candidates,
        generation=plan.generation,
        history=history,
        config=resolved,
    )
    return InteractionReplanResult(
        InteractionAction.CONTINUE, continued, "accepted outcome retained", ()
    )
