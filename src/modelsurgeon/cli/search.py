"""Constrained, resumable search planning command."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.config import ConstraintConfig, ObjectiveConfig, OptimizeMetric
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintMetric,
    ConstraintObservation,
    constraints_from_config,
)
from modelsurgeon.search.objectives import ObjectiveObservation, objectives_from_config
from modelsurgeon.search.policies import (
    PredictedSearchCandidate,
    SearchPolicy,
    SearchPolicyConfig,
    SearchPolicyKind,
    SearchPolicyState,
)
from modelsurgeon.search.resume import (
    PendingSearchEvaluation,
    SearchBudgetSnapshot,
    SearchResumeSnapshot,
    SearchResumeStore,
    SearchRngState,
)

SEARCH_COMMAND_SCHEMA_VERSION = 1


class SearchCommandError(RuntimeError):
    """Raised when a CLI search plan is ambiguous or unsafe to persist."""


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate: PredictedSearchCandidate
    candidate_state_id: str
    parent_checkpoint_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_state_id": self.candidate_state_id,
            "parent_state_id": self.candidate.parent_state_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "objective_observations": [
                {
                    "metric": item.metric.value,
                    "value": item.value,
                    "baseline_value": item.baseline_value,
                }
                for item in self.candidate.objective_observations
            ],
            "constraint_observations": [
                {
                    "metric": item.metric.value,
                    "value": item.value,
                    "baseline": item.baseline.value,
                }
                for item in self.candidate.constraint_observations
            ],
            "reward_uncertainty": self.candidate.reward_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class _SearchSpec:
    search_id: str
    source_checkpoint_id: str
    source_state_id: str
    accepted_checkpoint_ids: tuple[str, ...]
    frontier_checkpoint_ids: tuple[str, ...]
    policy: SearchPolicy
    candidates: tuple[_CandidateSpec, ...]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SearchCommandError(f"{label} must be an object")
    return value


def _exact(value: object, keys: set[str], label: str) -> dict[str, object]:
    record = _mapping(value, label)
    unknown = set(record) - keys
    if unknown:
        raise SearchCommandError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
    return record


def _text(value: object, label: str, prefix: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise SearchCommandError(f"{label} must be non-empty text")
    if prefix is not None and not value.startswith(prefix):
        raise SearchCommandError(f"{label} must start with {prefix}")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SearchCommandError(f"{label} must be numeric")
    return float(value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SearchCommandError(f"{label} must be an integer")
    return value


def _string_tuple(value: object, label: str, prefix: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SearchCommandError(f"{label} must be a list")
    result = tuple(_text(item, label, prefix) for item in value)
    if result != tuple(sorted(set(result))):
        raise SearchCommandError(f"{label} must be sorted and unique")
    return result


def _objective_observations(value: object) -> tuple[ObjectiveObservation, ...]:
    if not isinstance(value, list) or not value:
        raise SearchCommandError("candidate objective observations must be non-empty")
    result: list[ObjectiveObservation] = []
    for raw in value:
        record = _exact(raw, {"metric", "value", "baseline_value"}, "objective observation")
        baseline = record.get("baseline_value")
        result.append(
            ObjectiveObservation(
                OptimizeMetric(_text(record.get("metric"), "objective metric")),
                _number(record.get("value"), "objective value"),
                None if baseline is None else _number(baseline, "objective baseline"),
            )
        )
    return tuple(result)


def _constraint_observations(value: object) -> tuple[ConstraintObservation, ...]:
    if not isinstance(value, list) or not value:
        raise SearchCommandError("candidate constraint observations must be non-empty")
    result: list[ConstraintObservation] = []
    for raw in value:
        record = _exact(raw, {"metric", "value", "baseline"}, "constraint observation")
        result.append(
            ConstraintObservation(
                ConstraintMetric(_text(record.get("metric"), "constraint metric")),
                _number(record.get("value"), "constraint value"),
                BaselineReference(_text(record.get("baseline"), "constraint baseline")),
            )
        )
    return tuple(result)


def _candidate_spec(value: object) -> _CandidateSpec:
    record = _exact(
        value,
        {
            "candidate_id",
            "candidate_state_id",
            "parent_state_id",
            "parent_checkpoint_id",
            "objective_observations",
            "constraint_observations",
            "reward_uncertainty",
        },
        "candidate",
    )
    candidate = PredictedSearchCandidate(
        _text(record.get("candidate_id"), "candidate ID", "candidate_"),
        _text(record.get("parent_state_id"), "parent state ID", "state_"),
        _objective_observations(record.get("objective_observations")),
        _constraint_observations(record.get("constraint_observations")),
        _number(record.get("reward_uncertainty", 0.0), "reward uncertainty"),
    )
    return _CandidateSpec(
        candidate,
        _text(record.get("candidate_state_id"), "candidate state ID", "state_"),
        _text(record.get("parent_checkpoint_id"), "parent checkpoint ID", "checkpoint_"),
    )


def load_search_spec(path: Path) -> _SearchSpec:
    """Load a strict search plan and materialize its policy identity."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchCommandError(f"cannot read search config: {error}") from error
    record = _exact(
        raw,
        {
            "schema_version",
            "source_checkpoint_id",
            "source_state_id",
            "accepted_checkpoint_ids",
            "frontier_checkpoint_ids",
            "constraints",
            "objectives",
            "policy",
            "candidates",
        },
        "search config",
    )
    if record.get("schema_version") != SEARCH_COMMAND_SCHEMA_VERSION:
        raise SearchCommandError("unsupported search-command schema version")
    source_checkpoint_id = _text(
        record.get("source_checkpoint_id"), "source checkpoint ID", "checkpoint_"
    )
    source_state_id = _text(record.get("source_state_id"), "source state ID", "state_")
    accepted = _string_tuple(
        record.get("accepted_checkpoint_ids"), "accepted checkpoint IDs", "checkpoint_"
    )
    frontier = _string_tuple(
        record.get("frontier_checkpoint_ids"), "frontier checkpoint IDs", "checkpoint_"
    )
    if source_checkpoint_id not in accepted or not set(frontier) <= set(accepted) or not frontier:
        raise SearchCommandError("source/frontier checkpoints must belong to accepted lineage")
    constraint_config = ConstraintConfig.model_validate(
        _mapping(record.get("constraints"), "constraints")
    )
    objective_config = ObjectiveConfig.model_validate(
        _mapping(record.get("objectives"), "objectives")
    )
    policy_raw = _exact(
        record.get("policy"),
        {"kind", "evaluation_budget", "beam_width", "exploration_weight", "seed"},
        "policy",
    )
    policy_config = SearchPolicyConfig(
        SearchPolicyKind(_text(policy_raw.get("kind"), "policy kind")),
        _integer(policy_raw.get("evaluation_budget"), "evaluation budget"),
        _integer(policy_raw.get("beam_width", 1), "beam width"),
        _number(policy_raw.get("exploration_weight", 1.0), "exploration weight"),
        _integer(policy_raw.get("seed", 0), "policy seed"),
    )
    policy = SearchPolicy(
        policy_config,
        objectives_from_config(objective_config),
        constraints_from_config(constraint_config),
    )
    candidates_raw = record.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise SearchCommandError("search config requires a non-empty candidate pool")
    candidates = tuple(_candidate_spec(item) for item in candidates_raw)
    candidate_ids = tuple(item.candidate.candidate_id for item in candidates)
    state_ids = tuple(item.candidate_state_id for item in candidates)
    if len(candidate_ids) != len(set(candidate_ids)) or len(state_ids) != len(set(state_ids)):
        raise SearchCommandError("candidate and candidate-state IDs must be unique")
    if any(item.parent_checkpoint_id not in accepted for item in candidates):
        raise SearchCommandError("candidate parents must belong to accepted checkpoint lineage")
    identity = canonical_identity_json(
        {
            "schema_version": SEARCH_COMMAND_SCHEMA_VERSION,
            "source_checkpoint_id": source_checkpoint_id,
            "source_state_id": source_state_id,
            "accepted_checkpoint_ids": accepted,
            "frontier_checkpoint_ids": frontier,
            "constraint_set_id": policy.constraints.constraint_set_id,
            "objective_set_id": policy.objectives.objective_set_id,
            "policy_id": policy.policy_id,
        }
    ).encode()
    search_id = f"search_{hashlib.sha256(identity).hexdigest()}"
    return _SearchSpec(
        search_id,
        source_checkpoint_id,
        source_state_id,
        accepted,
        frontier,
        policy,
        candidates,
    )


def _dry_run_record(spec: _SearchSpec) -> dict[str, object]:
    selection = spec.policy.select(tuple(item.candidate for item in spec.candidates))
    return {
        "schema_version": SEARCH_COMMAND_SCHEMA_VERSION,
        "record_type": "search_dry_run",
        "search_id": spec.search_id,
        "constraints": spec.policy.constraints.to_record(),
        "objectives": spec.policy.objectives.to_record(),
        "budget": spec.policy.config.to_record(),
        "initial_pool": [item.to_record() for item in spec.candidates],
        "planned_selection": selection.to_record(),
        "accepted_checkpoint_lineage": list(spec.accepted_checkpoint_ids),
        "frontier_checkpoint_ids": list(spec.frontier_checkpoint_ids),
        "state_written": False,
    }


def run_search_plan(
    spec: _SearchSpec,
    state_path: Path,
    *,
    resume: bool,
) -> dict[str, object]:
    """Reserve one deterministic policy selection and atomically persist resume state."""

    by_candidate = {item.candidate.candidate_id: item for item in spec.candidates}
    with SearchResumeStore(state_path) as store:
        if resume:
            previous = store.load_latest(spec.search_id)
            if (
                previous.policy_state.policy_id != spec.policy.policy_id
                or previous.lineage_checkpoint_ids != spec.accepted_checkpoint_ids
                or previous.frontier_checkpoint_ids != spec.frontier_checkpoint_ids
            ):
                raise SearchCommandError("resume state does not match the supplied search plan")
            current_state = previous.policy_state
            generation = previous.generation + 1
            expected_generation: int | None = previous.generation
            prior_pending = previous.pending_evaluations
            evidence_cursor = previous.evidence_arrival_cursor
        else:
            current_state = SearchPolicyState(spec.policy.policy_id)
            generation = 0
            expected_generation = None
            prior_pending = ()
            evidence_cursor = 0
        selection = spec.policy.select(
            tuple(item.candidate for item in spec.candidates), current_state
        )
        new_pending = tuple(
            PendingSearchEvaluation(
                decision.candidate_id,
                by_candidate[decision.candidate_id].candidate_state_id,
                by_candidate[decision.candidate_id].parent_checkpoint_id,
            )
            for decision in selection.selected
        )
        snapshot = SearchResumeSnapshot(
            spec.search_id,
            generation,
            selection.next_state,
            SearchRngState(spec.policy.config.seed, selection.next_state.decision_index),
            spec.frontier_checkpoint_ids,
            spec.accepted_checkpoint_ids,
            SearchBudgetSnapshot(
                spec.policy.config.evaluation_budget,
                len(selection.next_state.selected_candidate_ids),
            ),
            (*prior_pending, *new_pending),
            evidence_cursor,
        )
        store.save(snapshot, expected_generation=expected_generation)
    return {
        "schema_version": SEARCH_COMMAND_SCHEMA_VERSION,
        "record_type": "search_selection",
        "search_id": spec.search_id,
        "generation": generation,
        "resumed": resume,
        "selection": selection.to_record(),
        "accepted_checkpoint_lineage": list(snapshot.lineage_checkpoint_ids),
        "frontier_checkpoint_ids": list(snapshot.frontier_checkpoint_ids),
        "pending_evaluations": [item.to_record() for item in snapshot.pending_evaluations],
        "state_persisted": True,
    }


def _write_output(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")


def search_command(
    config: Annotated[Path, typer.Argument(help="Canonical constrained-search JSON config")],
    state: Annotated[
        Path | None,
        typer.Option("--state", help="SQLite search-resume state path"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue from the latest atomic search snapshot"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print constraints, budget, pool, and selection only"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write canonical output without overwriting"),
    ] = None,
) -> None:
    """Start or resume one constrained search policy decision."""

    try:
        if resume and dry_run:
            raise SearchCommandError("--resume and --dry-run cannot be combined")
        spec = load_search_spec(config)
        if dry_run:
            record = _dry_run_record(spec)
        else:
            if state is None:
                raise SearchCommandError("--state is required unless --dry-run is used")
            record = run_search_plan(spec, state, resume=resume)
        payload = canonical_identity_json(record)
        if output is not None:
            _write_output(output, payload)
        typer.echo(payload)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"search error: {error}", err=True)
        raise typer.Exit(2) from error
