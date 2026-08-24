"""Seeded, resumable greedy, beam, and uncertainty-aware search selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.constraints import (
    ConstraintObservation,
    ConstraintSet,
)
from modelsurgeon.search.objectives import (
    ObjectiveObservation,
    ObjectiveScore,
    ObjectiveSet,
)

SEARCH_POLICY_SCHEMA_VERSION = 1


class SearchPolicyError(ValueError):
    """Raised when predicted candidates or policy state are inconsistent."""


class SearchPolicyKind(StrEnum):
    GREEDY = "greedy"
    BEAM = "beam"
    UNCERTAINTY_AWARE = "uncertainty_aware"


@dataclass(frozen=True, slots=True)
class SearchPolicyConfig:
    kind: SearchPolicyKind
    evaluation_budget: int
    beam_width: int = 1
    exploration_weight: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.evaluation_budget <= 0 or self.beam_width <= 0 or self.seed < 0:
            raise SearchPolicyError("policy budget/width must be positive and seed non-negative")
        if not math.isfinite(self.exploration_weight) or self.exploration_weight < 0:
            raise SearchPolicyError("exploration weight must be finite and non-negative")
        if self.kind is SearchPolicyKind.GREEDY and self.beam_width != 1:
            raise SearchPolicyError("greedy search requires beam width one")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SEARCH_POLICY_SCHEMA_VERSION,
            "kind": self.kind.value,
            "evaluation_budget": self.evaluation_budget,
            "beam_width": self.beam_width,
            "exploration_weight": self.exploration_weight,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PredictedSearchCandidate:
    candidate_id: str
    parent_state_id: str
    objective_observations: tuple[ObjectiveObservation, ...]
    constraint_observations: tuple[ConstraintObservation, ...]
    reward_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.parent_state_id.startswith("state_"):
            raise SearchPolicyError("predicted candidates require candidate and parent state IDs")
        if not math.isfinite(self.reward_uncertainty) or self.reward_uncertainty < 0:
            raise SearchPolicyError("predicted reward uncertainty must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SearchPolicyState:
    policy_id: str
    selected_candidate_ids: tuple[str, ...] = ()
    decision_index: int = 0

    def __post_init__(self) -> None:
        if not self.policy_id.startswith("policy_"):
            raise SearchPolicyError("search state requires its canonical policy ID")
        if self.decision_index < 0:
            raise SearchPolicyError("search decision index cannot be negative")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise SearchPolicyError("selected candidate IDs must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SEARCH_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "decision_index": self.decision_index,
        }


@dataclass(frozen=True, slots=True)
class SearchDecision:
    candidate_id: str
    parent_state_id: str
    selected: bool
    reason: str
    predicted_reward: float | None
    reward_uncertainty: float
    acquisition_score: float | None
    objective_score: ObjectiveScore | None
    constraint_record: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_state_id": self.parent_state_id,
            "selected": self.selected,
            "reason": self.reason,
            "predicted_reward": self.predicted_reward,
            "reward_uncertainty": self.reward_uncertainty,
            "acquisition_score": self.acquisition_score,
            "predicted_objectives": (
                None if self.objective_score is None else self.objective_score.to_record()
            ),
            "predicted_constraints": self.constraint_record,
        }


@dataclass(frozen=True, slots=True)
class SearchSelection:
    decisions: tuple[SearchDecision, ...]
    next_state: SearchPolicyState
    budget_exhausted: bool

    @property
    def selected(self) -> tuple[SearchDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.selected)

    def to_record(self) -> dict[str, object]:
        return {
            "decisions": [decision.to_record() for decision in self.decisions],
            "next_state": self.next_state.to_record(),
            "budget_exhausted": self.budget_exhausted,
        }


def _tie_rank(seed: int, decision_index: int, candidate_id: str) -> int:
    payload = f"{seed}:{decision_index}:{candidate_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


class SearchPolicy:
    def __init__(
        self,
        config: SearchPolicyConfig,
        objectives: ObjectiveSet,
        constraints: ConstraintSet,
    ) -> None:
        self.config = config
        self.objectives = objectives
        self.constraints = constraints

    @property
    def policy_id(self) -> str:
        payload = canonical_identity_json(
            {
                "config": self.config.to_record(),
                "objective_set_id": self.objectives.objective_set_id,
                "constraint_set_id": self.constraints.constraint_set_id,
            }
        ).encode()
        return f"policy_{hashlib.sha256(payload).hexdigest()}"

    def select(
        self,
        candidates: tuple[PredictedSearchCandidate, ...],
        state: SearchPolicyState | None = None,
    ) -> SearchSelection:
        current = state or SearchPolicyState(self.policy_id)
        if current.policy_id != self.policy_id:
            raise SearchPolicyError("search state belongs to a different policy definition")
        if len(current.selected_candidate_ids) > self.config.evaluation_budget:
            raise SearchPolicyError("search state exceeds the configured evaluation budget")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise SearchPolicyError("candidate pool IDs must be unique")
        previously_selected = set(current.selected_candidate_ids)
        remaining_budget = self.config.evaluation_budget - len(previously_selected)
        scored: list[
            tuple[float, int, PredictedSearchCandidate, ObjectiveScore, dict[str, object]]
        ] = []
        decisions: list[SearchDecision] = []
        for candidate in candidates:
            if candidate.candidate_id in previously_selected:
                decisions.append(
                    SearchDecision(
                        candidate.candidate_id,
                        candidate.parent_state_id,
                        False,
                        "already_selected",
                        None,
                        candidate.reward_uncertainty,
                        None,
                        None,
                        {},
                    )
                )
                continue
            objective_score = self.objectives.score(candidate.objective_observations)
            acquisition = objective_score.reward
            if self.config.kind is SearchPolicyKind.UNCERTAINTY_AWARE:
                acquisition += self.config.exploration_weight * candidate.reward_uncertainty
            constraint_evaluation = self.constraints.evaluate(candidate.constraint_observations)
            constraint_record = constraint_evaluation.to_record()
            if not constraint_evaluation.passed:
                decisions.append(
                    SearchDecision(
                        candidate.candidate_id,
                        candidate.parent_state_id,
                        False,
                        "predicted_constraint_violation",
                        objective_score.reward,
                        candidate.reward_uncertainty,
                        acquisition,
                        objective_score,
                        constraint_record,
                    )
                )
                continue
            scored.append(
                (
                    acquisition,
                    _tie_rank(self.config.seed, current.decision_index, candidate.candidate_id),
                    candidate,
                    objective_score,
                    constraint_record,
                )
            )
        selection_limit = (
            1 if self.config.kind is SearchPolicyKind.GREEDY else self.config.beam_width
        )
        selection_limit = max(0, min(selection_limit, remaining_budget))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2].candidate_id))
        chosen = {item[2].candidate_id for item in ranked[:selection_limit]}
        reason = {
            SearchPolicyKind.GREEDY: "highest_predicted_reward",
            SearchPolicyKind.BEAM: "within_predicted_reward_beam",
            SearchPolicyKind.UNCERTAINTY_AWARE: "within_uncertainty_adjusted_beam",
        }[self.config.kind]
        for acquisition, _, candidate, objective_score, constraint_record in ranked:
            selected = candidate.candidate_id in chosen
            decisions.append(
                SearchDecision(
                    candidate.candidate_id,
                    candidate.parent_state_id,
                    selected,
                    reason if selected else "outside_policy_cutoff",
                    objective_score.reward,
                    candidate.reward_uncertainty,
                    acquisition,
                    objective_score,
                    constraint_record,
                )
            )
        decisions.sort(key=lambda decision: decision.candidate_id)
        selected_in_rank_order = tuple(item[2].candidate_id for item in ranked[:selection_limit])
        next_state = SearchPolicyState(
            self.policy_id,
            (*current.selected_candidate_ids, *selected_in_rank_order),
            current.decision_index + 1,
        )
        return SearchSelection(
            tuple(decisions),
            next_state,
            len(next_state.selected_candidate_ids) >= self.config.evaluation_budget,
        )
