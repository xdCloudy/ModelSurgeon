"""Deterministic Pareto-beam and constrained evolutionary architecture policies."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import StrEnum

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.candidate_space import ArchitectureCandidate
from modelsurgeon.search.constraints import ConstraintObservation, ConstraintSet
from modelsurgeon.search.objectives import ObjectiveSet
from modelsurgeon.search.pareto import (
    ParetoCandidate,
    ParetoObjectiveValue,
    conservatively_dominates,
)

ARCHITECTURE_POLICY_SCHEMA_VERSION = 1


class ArchitecturePolicyError(ValueError):
    """Raised when an architecture policy input or resume state is invalid."""


class ArchitecturePolicyKind(StrEnum):
    PARETO_BEAM = "pareto_beam"
    EVOLUTIONARY = "evolutionary"


class ArchitectureEvidenceStatus(StrEnum):
    PREDICTED = "predicted"
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ArchitecturePolicyError(f"{label} must be positive")
    return value


def _tie_rank(seed: int, decision_index: int, candidate_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{decision_index}:{candidate_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def _assignment_key(candidate: ArchitectureCandidate) -> tuple[str, str, str]:
    return (
        candidate.base_state_id,
        candidate.hardware_profile_id,
        canonical_identity_json([[axis.value, value] for axis, value in candidate.assignments]),
    )


def _assignment_map(candidate: ArchitectureCandidate) -> dict[str, object]:
    return {axis.value: value for axis, value in candidate.assignments}


def _distance(left: ArchitectureCandidate, right: ArchitectureCandidate) -> int:
    left_values = _assignment_map(left)
    right_values = _assignment_map(right)
    axes = set(left_values) | set(right_values)
    return sum(left_values.get(axis) != right_values.get(axis) for axis in axes) + int(
        left.hardware_profile_id != right.hardware_profile_id
    )


@dataclass(frozen=True, slots=True)
class CompleteArchitectureState:
    """A legal, complete architecture with immutable predicted/evaluated evidence."""

    candidate: ArchitectureCandidate
    objectives: tuple[ParetoObjectiveValue, ...]
    constraints: tuple[ConstraintObservation, ...]
    status: ArchitectureEvidenceStatus = ArchitectureEvidenceStatus.PREDICTED
    parent_candidate_ids: tuple[str, ...] = ()
    operation: str = "seed"
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ArchitecturePolicyError("complete architecture states require objectives")
        metrics = [item.metric for item in self.objectives]
        if len(metrics) != len(set(metrics)):
            raise ArchitecturePolicyError("architecture objective metrics must be unique")
        if not self.operation.strip():
            raise ArchitecturePolicyError("architecture state operation is required")
        if any(not item.strip() for item in self.parent_candidate_ids):
            raise ArchitecturePolicyError("architecture parent IDs cannot be empty")
        keys = [key for key, _ in self.provenance]
        if len(keys) != len(set(keys)) or any(not key.strip() for key in keys):
            raise ArchitecturePolicyError("architecture provenance keys must be unique")

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def as_pareto_candidate(self) -> ParetoCandidate:
        return ParetoCandidate(
            self.candidate_id,
            self.objectives,
            {
                "base_state_id": self.candidate.base_state_id,
                "hardware_profile_id": self.candidate.hardware_profile_id,
                "assignments": [
                    {"axis": axis.value, "value": value}
                    for axis, value in self.candidate.assignments
                ],
            },
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ARCHITECTURE_POLICY_SCHEMA_VERSION,
            "candidate": self.candidate.to_record(),
            "objectives": [item.to_record() for item in self.objectives],
            "constraints": [
                {
                    "metric": item.metric.value,
                    "value": item.value,
                    "baseline": item.baseline.value,
                }
                for item in self.constraints
            ],
            "status": self.status.value,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "operation": self.operation,
            "provenance": [{"key": key, "value": value} for key, value in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class ArchitecturePolicyConfig:
    kind: ArchitecturePolicyKind
    evaluation_budget: int
    beam_width: int = 4
    population_size: int = 8
    generations: int = 4
    elitism: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        _positive(self.evaluation_budget, "evaluation budget")
        _positive(self.beam_width, "beam width")
        _positive(self.population_size, "population size")
        _positive(self.generations, "generation budget")
        if isinstance(self.elitism, bool) or not 0 <= self.elitism <= self.population_size:
            raise ArchitecturePolicyError("elitism must be between zero and population size")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise ArchitecturePolicyError("architecture policy seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ARCHITECTURE_POLICY_SCHEMA_VERSION,
            "kind": self.kind.value,
            "evaluation_budget": self.evaluation_budget,
            "beam_width": self.beam_width,
            "population_size": self.population_size,
            "generations": self.generations,
            "elitism": self.elitism,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ArchitecturePolicyState:
    """Serializable cursor, population, frontier, and charged evaluation IDs."""

    policy_id: str
    selected_candidate_ids: tuple[str, ...] = ()
    population_candidate_ids: tuple[str, ...] = ()
    frontier_candidate_ids: tuple[str, ...] = ()
    generation: int = 0
    decision_index: int = 0

    def __post_init__(self) -> None:
        if not self.policy_id.startswith("architecture_policy_"):
            raise ArchitecturePolicyError("architecture policy state requires canonical policy ID")
        for label, values in (
            ("selected", self.selected_candidate_ids),
            ("population", self.population_candidate_ids),
            ("frontier", self.frontier_candidate_ids),
        ):
            if len(values) != len(set(values)):
                raise ArchitecturePolicyError(f"{label} candidate IDs must be unique")
        if isinstance(self.generation, bool) or self.generation < 0:
            raise ArchitecturePolicyError("architecture policy generation cannot be negative")
        if isinstance(self.decision_index, bool) or self.decision_index < 0:
            raise ArchitecturePolicyError("architecture policy decision index cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ARCHITECTURE_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "population_candidate_ids": list(self.population_candidate_ids),
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "generation": self.generation,
            "decision_index": self.decision_index,
        }

    @classmethod
    def from_record(cls, record: object) -> ArchitecturePolicyState:
        if not isinstance(record, dict):
            raise ArchitecturePolicyError("architecture policy state must be an object")
        required = {
            "schema_version",
            "policy_id",
            "selected_candidate_ids",
            "population_candidate_ids",
            "frontier_candidate_ids",
            "generation",
            "decision_index",
        }
        if set(record) != required:
            raise ArchitecturePolicyError("architecture policy state has unknown or missing fields")
        if record["schema_version"] != ARCHITECTURE_POLICY_SCHEMA_VERSION:
            raise ArchitecturePolicyError("unsupported architecture policy state schema")
        values = (
            record["selected_candidate_ids"],
            record["population_candidate_ids"],
            record["frontier_candidate_ids"],
        )
        if any(
            not isinstance(value, list) or not all(isinstance(item, str) for item in value)
            for value in values
        ):
            raise ArchitecturePolicyError("architecture policy candidate IDs must be string lists")
        if not isinstance(record["policy_id"], str):
            raise ArchitecturePolicyError("architecture policy ID must be text")
        if not isinstance(record["generation"], int) or isinstance(record["generation"], bool):
            raise ArchitecturePolicyError("architecture policy generation must be an integer")
        if not isinstance(record["decision_index"], int) or isinstance(
            record["decision_index"], bool
        ):
            raise ArchitecturePolicyError("architecture policy decision index must be an integer")
        return cls(
            record["policy_id"],
            tuple(values[0]),
            tuple(values[1]),
            tuple(values[2]),
            record["generation"],
            record["decision_index"],
        )


@dataclass(frozen=True, slots=True)
class ArchitecturePolicyDecision:
    candidate_id: str
    selected: bool
    status: ArchitectureEvidenceStatus
    reason: str
    operation: str
    generation: int
    parent_candidate_ids: tuple[str, ...]
    objective_records: tuple[dict[str, object], ...]
    constraint_record: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "selected": self.selected,
            "status": self.status.value,
            "reason": self.reason,
            "operation": self.operation,
            "generation": self.generation,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "objectives": list(self.objective_records),
            "constraints": self.constraint_record,
        }


@dataclass(frozen=True, slots=True)
class ArchitecturePolicySelection:
    decisions: tuple[ArchitecturePolicyDecision, ...]
    next_state: ArchitecturePolicyState
    budget_exhausted: bool

    @property
    def selected(self) -> tuple[ArchitecturePolicyDecision, ...]:
        return tuple(item for item in self.decisions if item.selected)

    def to_record(self) -> dict[str, object]:
        return {
            "decisions": [item.to_record() for item in self.decisions],
            "next_state": self.next_state.to_record(),
            "budget_exhausted": self.budget_exhausted,
        }


class ArchitectureSearchPolicy:
    """Select complete legal states with conservative Pareto and evolutionary policies."""

    def __init__(
        self,
        config: ArchitecturePolicyConfig,
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
        )
        return f"architecture_policy_{hashlib.sha256(payload.encode()).hexdigest()}"

    def _validate_pool(
        self, candidates: tuple[CompleteArchitectureState, ...]
    ) -> dict[str, CompleteArchitectureState]:
        pool = {candidate.candidate_id: candidate for candidate in candidates}
        if len(pool) != len(candidates):
            raise ArchitecturePolicyError("architecture candidate pool IDs must be unique")
        if not pool:
            raise ArchitecturePolicyError("architecture candidate pool cannot be empty")
        return pool

    def _constraint_record(
        self, state: CompleteArchitectureState
    ) -> tuple[bool, dict[str, object]]:
        evaluation = self.constraints.evaluate(state.constraints)
        return evaluation.passed, evaluation.to_record()

    def _eligible(
        self,
        state: CompleteArchitectureState,
        selected_ids: set[str],
    ) -> tuple[bool, str, dict[str, object]]:
        if state.candidate_id in selected_ids:
            return False, "already_selected", {}
        passed, record = self._constraint_record(state)
        if state.status is not ArchitectureEvidenceStatus.PREDICTED:
            return False, f"retained_{state.status.value}_evidence", record
        if not passed:
            return False, "predicted_constraint_violation", record
        return True, "", record

    def _frontier(
        self, states: tuple[CompleteArchitectureState, ...]
    ) -> tuple[CompleteArchitectureState, ...]:
        frontier = tuple(
            state
            for state in states
            if not any(
                other.candidate_id != state.candidate_id
                and conservatively_dominates(
                    other.as_pareto_candidate(), state.as_pareto_candidate(), self.objectives
                )
                for other in states
            )
        )
        return tuple(sorted(frontier, key=lambda state: state.candidate_id))

    def _utility(self, state: CompleteArchitectureState) -> float:
        values = {item.metric: item.estimate for item in state.objectives}
        total = 0.0
        weight = 0.0
        for term in self.objectives.terms:
            value = values.get(term.metric)
            if value is None:
                return -math.inf
            direction = 1.0 if term.direction.value == "maximize" else -1.0
            total += direction * term.weight * value
            weight += term.weight
        return total / weight

    def _diverse_order(
        self,
        states: tuple[CompleteArchitectureState, ...],
        limit: int,
        decision_index: int,
    ) -> tuple[CompleteArchitectureState, ...]:
        remaining = {state.candidate_id: state for state in states}
        chosen: list[CompleteArchitectureState] = []
        while remaining and len(chosen) < limit:
            ranked = sorted(
                remaining.values(),
                key=lambda state: (
                    -self._utility(state),
                    -(
                        min(
                            (_distance(state.candidate, previous.candidate) for previous in chosen),
                            default=0,
                        )
                    ),
                    _tie_rank(self.config.seed, decision_index, state.candidate_id),
                    state.candidate_id,
                ),
            )
            next_state = ranked[0]
            chosen.append(next_state)
            del remaining[next_state.candidate_id]
        return tuple(chosen)

    def _decision(
        self,
        state: CompleteArchitectureState,
        selected: bool,
        reason: str,
        generation: int,
        constraint_record: dict[str, object] | None = None,
    ) -> ArchitecturePolicyDecision:
        if constraint_record is None:
            _, constraint_record = self._constraint_record(state)
        return ArchitecturePolicyDecision(
            state.candidate_id,
            selected,
            state.status,
            reason,
            state.operation,
            generation,
            state.parent_candidate_ids,
            tuple(item.to_record() for item in state.objectives),
            constraint_record,
        )

    def _select_pareto_beam(
        self,
        pool: dict[str, CompleteArchitectureState],
        current: ArchitecturePolicyState,
    ) -> ArchitecturePolicySelection:
        selected_ids = set(current.selected_candidate_ids)
        decisions: list[ArchitecturePolicyDecision] = []
        eligible: list[CompleteArchitectureState] = []
        for state in pool.values():
            valid, reason, record = self._eligible(state, selected_ids)
            if valid:
                eligible.append(state)
            else:
                decisions.append(self._decision(state, False, reason, current.generation, record))
        frontier = self._frontier(tuple(eligible))
        remaining_budget = self.config.evaluation_budget - len(selected_ids)
        chosen = self._diverse_order(
            frontier, min(self.config.beam_width, remaining_budget), current.decision_index
        )
        chosen_ids = {state.candidate_id for state in chosen}
        for state in eligible:
            decisions.append(
                self._decision(
                    state,
                    state.candidate_id in chosen_ids,
                    "pareto_frontier"
                    if state.candidate_id in chosen_ids
                    else "dominated_or_beam_cutoff",
                    current.generation,
                )
            )
        decisions.sort(key=lambda item: item.candidate_id)
        next_state = ArchitecturePolicyState(
            self.policy_id,
            (*current.selected_candidate_ids, *[state.candidate_id for state in chosen]),
            tuple(state.candidate_id for state in chosen),
            tuple(state.candidate_id for state in frontier),
            current.generation,
            current.decision_index + 1,
        )
        return ArchitecturePolicySelection(
            tuple(decisions),
            next_state,
            len(next_state.selected_candidate_ids) >= self.config.evaluation_budget,
        )

    def _variants(
        self,
        parent: CompleteArchitectureState,
        pool: dict[str, CompleteArchitectureState],
    ) -> tuple[CompleteArchitectureState, ...]:
        variants = [
            state
            for state in pool.values()
            if state.candidate.base_state_id == parent.candidate.base_state_id
            and _distance(parent.candidate, state.candidate) == 1
        ]
        variants.sort(
            key=lambda state: (
                _tie_rank(
                    self.config.seed,
                    0,
                    f"mutation:{parent.candidate_id}:{state.candidate_id}",
                ),
                state.candidate_id,
            )
        )
        return tuple(
            replace(
                state,
                operation="mutation",
                parent_candidate_ids=(parent.candidate_id,),
            )
            for state in variants
        )

    def _crossover(
        self,
        left: CompleteArchitectureState,
        right: CompleteArchitectureState,
        pool: dict[str, CompleteArchitectureState],
    ) -> CompleteArchitectureState | None:
        if left.candidate.base_state_id != right.candidate.base_state_id:
            return None
        left_values = _assignment_map(left.candidate)
        right_values = _assignment_map(right.candidate)
        axes = sorted(set(left_values) | set(right_values))
        combined = {
            axis: (
                left_values.get(axis)
                if (
                    _tie_rank(
                        self.config.seed,
                        0,
                        f"crossover:{left.candidate_id}:{right.candidate_id}:{axis}",
                    )
                    % 2
                )
                else right_values.get(axis)
            )
            for axis in axes
        }
        profiles = sorted({left.candidate.hardware_profile_id, right.candidate.hardware_profile_id})
        profile = profiles[
            _tie_rank(self.config.seed, 0, f"profile:{left.candidate_id}:{right.candidate_id}")
            % len(profiles)
        ]
        for state in sorted(pool.values(), key=lambda item: item.candidate_id):
            candidate = state.candidate
            if (
                candidate.base_state_id == left.candidate.base_state_id
                and candidate.hardware_profile_id == profile
                and _assignment_map(candidate) == combined
            ):
                return replace(
                    state,
                    operation="crossover",
                    parent_candidate_ids=(left.candidate_id, right.candidate_id),
                )
        return None

    def _select_evolutionary(
        self,
        pool: dict[str, CompleteArchitectureState],
        current: ArchitecturePolicyState,
    ) -> ArchitecturePolicySelection:
        selected_ids = set(current.selected_candidate_ids)
        if current.generation >= self.config.generations:
            return ArchitecturePolicySelection(
                (), current, True
            )
        all_decisions: dict[str, ArchitecturePolicyDecision] = {}
        if not current.population_candidate_ids:
            eligible = tuple(
                state
                for state in pool.values()
                if self._eligible(state, selected_ids)[0]
            )
            frontier = self._frontier(eligible)
            chosen = self._diverse_order(
                frontier,
                min(self.config.population_size, self.config.evaluation_budget - len(selected_ids)),
                current.decision_index,
            )
            candidate_states = eligible
            next_population = tuple(state.candidate_id for state in chosen)
            frontier_ids = tuple(state.candidate_id for state in frontier)
            reason = "seeded_pareto_population"
        else:
            parents = tuple(
                pool[item]
                for item in current.population_candidate_ids
                if item in pool
            )
            generated: dict[str, CompleteArchitectureState] = {}
            for parent in parents:
                for state in self._variants(parent, pool):
                    generated.setdefault(state.candidate_id, state)
            for index, left in enumerate(parents):
                for right in parents[index + 1 :]:
                    child = self._crossover(left, right, pool)
                    if child is not None:
                        generated.setdefault(child.candidate_id, child)
            eligible_generated = tuple(
                state
                for state in generated.values()
                if state.candidate_id not in selected_ids and self._eligible(state, selected_ids)[0]
            )
            frontier = self._frontier(eligible_generated)
            remaining_budget = self.config.evaluation_budget - len(selected_ids)
            elite_ids = tuple(current.population_candidate_ids[: self.config.elitism])
            chosen = self._diverse_order(
                frontier,
                min(self.config.population_size - len(elite_ids), remaining_budget),
                current.decision_index,
            )
            next_population = tuple(
                dict.fromkeys((*elite_ids, *[state.candidate_id for state in chosen]))
            )
            candidate_states = tuple(generated.values())
            frontier_ids = tuple(state.candidate_id for state in frontier)
            reason = "evolutionary_pareto_survivor"
        chosen_ids = {state.candidate_id for state in chosen}
        for state in candidate_states:
            valid, invalid_reason, record = self._eligible(state, selected_ids)
            all_decisions[state.candidate_id] = self._decision(
                state,
                valid and state.candidate_id in chosen_ids,
                reason if valid and state.candidate_id in chosen_ids else (
                    invalid_reason if not valid else "outside_pareto_population"
                ),
                current.generation,
                record,
            )
        decisions = tuple(sorted(all_decisions.values(), key=lambda item: item.candidate_id))
        next_state = ArchitecturePolicyState(
            self.policy_id,
            (*current.selected_candidate_ids, *[state.candidate_id for state in chosen]),
            next_population,
            frontier_ids,
            current.generation + 1,
            current.decision_index + 1,
        )
        return ArchitecturePolicySelection(
            decisions,
            next_state,
            len(next_state.selected_candidate_ids) >= self.config.evaluation_budget
            or next_state.generation >= self.config.generations,
        )

    def select(
        self,
        candidates: tuple[CompleteArchitectureState, ...],
        state: ArchitecturePolicyState | None = None,
    ) -> ArchitecturePolicySelection:
        current = state or ArchitecturePolicyState(self.policy_id)
        if current.policy_id != self.policy_id:
            raise ArchitecturePolicyError("architecture policy state belongs to another policy")
        if len(current.selected_candidate_ids) > self.config.evaluation_budget:
            raise ArchitecturePolicyError("architecture policy state exceeds evaluation budget")
        pool = self._validate_pool(candidates)
        if self.config.kind is ArchitecturePolicyKind.PARETO_BEAM:
            return self._select_pareto_beam(pool, current)
        return self._select_evolutionary(pool, current)
