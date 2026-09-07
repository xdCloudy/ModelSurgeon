"""Bounded, reproducible v2.5 constraint-negotiation decision study.

The study is a read-only replay harness around the v2.5 control-plane
contracts.  It compares refusal-only, measured-Pareto, and prediction-only
policies on typed fixture campaigns.  Predictions are retained as a negative
control and are never passed to the measured Pareto or amendment APIs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from modelsurgeon.explain import FeasibilityCandidateEvidence, FeasibilityOutcome
    from modelsurgeon.search import (
        MetricObservation,
        ObjectiveContract,
        ObjectivePluginBinding,
    )

from modelsurgeon.experiments.identity import canonical_identity_json

NEGOTIATION_STUDY_SCHEMA_VERSION = 1
NEGOTIATION_STUDY_PROTOCOL_ID = "v25-negotiation-decision-quality-v1"
MAX_NEGOTIATION_STUDY_BYTES = 256_000
MAX_NEGOTIATION_STUDY_SCENARIOS = 16
MAX_NEGOTIATION_STUDY_SEEDS = 3
MAX_NEGOTIATION_STUDY_EVALUATIONS = 16


class NegotiationStudyError(ValueError):
    """Raised when a study is malformed or cannot remain fail-closed."""


class NegotiationPolicy(StrEnum):
    REFUSAL_ONLY = "refusal_only"
    MEASURED_PARETO = "measured_pareto"
    PREDICTION_ONLY = "prediction_only"


class NegotiationSplit(StrEnum):
    FIXTURE = "fixture"
    HELD_OUT = "held_out"
    ADVERSARIAL = "adversarial"


class NegotiationScenarioKind(StrEnum):
    INFEASIBLE = "infeasible"
    PREDICTED_ONLY = "predicted_only"
    UNSUPPORTED = "unsupported"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NegotiationStudyError("value is not canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegotiationStudyError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    result = _text(value, label)
    if prefix is not None and not result.startswith(prefix):
        raise NegotiationStudyError(f"{label} must start with {prefix}")
    return result


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NegotiationStudyError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise NegotiationStudyError(f"{label} must be an array")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise NegotiationStudyError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NegotiationStudyError(f"{label} must be an integer >= {minimum}")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NegotiationStudyError(f"{label} must be numeric")
    return float(value)


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))) or any(not value.strip() for value in result):
        raise NegotiationStudyError(f"{label} must be sorted, unique, and non-empty")
    return result


@dataclass(frozen=True, slots=True)
class NegotiationStudyLimits:
    """Hard CPU/evidence bounds for the fixture campaign replay."""

    max_scenarios: int = MAX_NEGOTIATION_STUDY_SCENARIOS
    fixed_evaluation_count: int = 4
    max_seeds_per_scenario: int = MAX_NEGOTIATION_STUDY_SEEDS
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_scenarios <= MAX_NEGOTIATION_STUDY_SCENARIOS:
            raise NegotiationStudyError("study scenario bound is invalid")
        if not 1 <= self.fixed_evaluation_count <= MAX_NEGOTIATION_STUDY_EVALUATIONS:
            raise NegotiationStudyError("fixed evaluation count is invalid")
        if not 1 <= self.max_seeds_per_scenario <= MAX_NEGOTIATION_STUDY_SEEDS:
            raise NegotiationStudyError("seed bound is invalid")
        if self.max_output_bytes <= 0:
            raise NegotiationStudyError("study output bound must be positive")

    def to_record(self) -> dict[str, int]:
        return {
            "max_scenarios": self.max_scenarios,
            "fixed_evaluation_count": self.fixed_evaluation_count,
            "max_seeds_per_scenario": self.max_seeds_per_scenario,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class NegotiationStudyScenario:
    """One campaign trace with an immutable source and typed evidence."""

    scenario_id: str
    split: NegotiationSplit
    model_fixture: str
    source_model_digest: str
    campaign_id: str
    session_id: str
    run_id: str
    fixed_evaluation_count: int
    seeds: tuple[int, ...]
    original_contract: ObjectiveContract
    candidates: tuple[FeasibilityCandidateEvidence, ...]
    kind: NegotiationScenarioKind
    target_candidate_id: str | None = None
    proposed_contract: ObjectiveContract | None = None
    target_recovery_expected: bool = False
    negative: bool = True
    inconclusive: bool = False

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario ID")
        _text(self.model_fixture, "model fixture")
        _text(self.source_model_digest, "source model digest")
        for field, value in (
            ("campaign ID", self.campaign_id),
            ("session ID", self.session_id),
            ("run ID", self.run_id),
        ):
            _identifier(value, field)
        if self.fixed_evaluation_count <= 0 or len(self.candidates) != self.fixed_evaluation_count:
            raise NegotiationStudyError(
                f"scenario {self.scenario_id} must retain exactly its fixed evaluations"
            )
        if self.seeds != tuple(sorted(set(self.seeds))) or not self.seeds:
            raise NegotiationStudyError("scenario seeds must be sorted and unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise NegotiationStudyError("scenario seeds must be non-negative integers")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise NegotiationStudyError("scenario candidates must be sorted and unique")
        if any(item.source_model_digest != self.source_model_digest for item in self.candidates):
            raise NegotiationStudyError("scenario evidence changes source-model identity")
        if self.target_candidate_id is not None and self.target_candidate_id not in candidate_ids:
            raise NegotiationStudyError("scenario target candidate is unknown")
        if self.target_recovery_expected and (
            self.target_candidate_id is None or self.proposed_contract is None
        ):
            raise NegotiationStudyError("expected recovery requires a target and proposed contract")
        if (
            self.proposed_contract is not None
            and self.proposed_contract.contract_id == self.original_contract.contract_id
        ):
            raise NegotiationStudyError("proposed contract must represent a visible amendment")

    def to_record(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split.value,
            "model_fixture": self.model_fixture,
            "source_model_digest": self.source_model_digest,
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "fixed_evaluation_count": self.fixed_evaluation_count,
            "seeds": list(self.seeds),
            "original_contract": self.original_contract.to_record(),
            "proposed_contract": (
                None if self.proposed_contract is None else self.proposed_contract.to_record()
            ),
            "candidates": [item.to_record() for item in self.candidates],
            "kind": self.kind.value,
            "target_candidate_id": self.target_candidate_id,
            "target_recovery_expected": self.target_recovery_expected,
            "negative": self.negative,
            "inconclusive": self.inconclusive,
        }


@dataclass(frozen=True, slots=True)
class NegotiationStudyCorpus:
    """Versioned fixture campaign protocol."""

    corpus_id: str
    corpus_revision: str
    limits: NegotiationStudyLimits
    scenarios: tuple[NegotiationStudyScenario, ...]
    policies: tuple[NegotiationPolicy, ...] = (
        NegotiationPolicy.MEASURED_PARETO,
        NegotiationPolicy.PREDICTION_ONLY,
        NegotiationPolicy.REFUSAL_ONLY,
    )
    protocol_id: str = NEGOTIATION_STUDY_PROTOCOL_ID
    schema_version: int = NEGOTIATION_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NEGOTIATION_STUDY_SCHEMA_VERSION:
            raise NegotiationStudyError("unsupported negotiation study schema")
        _identifier(self.corpus_id, "corpus ID")
        _text(self.corpus_revision, "corpus revision")
        if not self.scenarios or len(self.scenarios) > self.limits.max_scenarios:
            raise NegotiationStudyError("study scenario count exceeds its bound")
        ids = tuple(item.scenario_id for item in self.scenarios)
        if ids != tuple(sorted(set(ids))):
            raise NegotiationStudyError("study scenario IDs must be sorted and unique")
        values = tuple(item.value for item in self.policies)
        if values != tuple(sorted(set(values))) or not self.policies:
            raise NegotiationStudyError("study policies must be sorted and unique")
        if any(len(item.seeds) > self.limits.max_seeds_per_scenario for item in self.scenarios):
            raise NegotiationStudyError("study exceeds its per-scenario seed bound")
        if any(
            item.fixed_evaluation_count != self.limits.fixed_evaluation_count
            for item in self.scenarios
        ):
            raise NegotiationStudyError("study evaluation count is not fixed")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "limits": self.limits.to_record(),
            "policies": [item.value for item in self.policies],
            "scenarios": [item.to_record() for item in self.scenarios],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class NegotiationStudyResult:
    """One retained policy/seed/campaign trace, including negative cells."""

    policy: NegotiationPolicy
    scenario_id: str
    split: NegotiationSplit
    seed: int
    source_model_digest: str
    original_contract_id: str
    original_hard_constraint_digest: str
    effective_contract_id: str
    explanation_outcome: FeasibilityOutcome
    measured_candidate_ids: tuple[str, ...]
    retained_evidence_ids: tuple[str, ...]
    presented_alternative_ids: tuple[str, ...]
    presented_alternative_statuses: tuple[str, ...]
    claimed_alternative_count: int
    amendment_attempted: bool
    recovery_target_expected: bool
    amendment_traceable: bool
    amendment_accurate: bool
    amendment_id: str | None
    amendment_diff_id: str | None
    amendment_application: Mapping[str, object] | None
    constraint_preserved: bool
    feasible_target_recovered: bool
    misleading_claim: bool
    interaction_cost: int
    negative: bool
    inconclusive: bool
    shippable: bool
    trace_events: tuple[str, ...]
    error: str | None = None
    silent_constraint_change: bool = False

    @property
    def passed(self) -> bool:
        return self.error is None and not self.silent_constraint_change and self.shippable

    @property
    def measured_alternative_grounding(self) -> bool:
        return bool(self.presented_alternative_ids) and all(
            status != "predicted" for status in self.presented_alternative_statuses
        )

    @property
    def cell_id(self) -> str:
        return "negotiation_study_cell_" + _digest(self.to_record(include_identity=False))[7:]

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "policy": self.policy.value,
            "scenario_id": self.scenario_id,
            "split": self.split.value,
            "seed": self.seed,
            "source_model_digest": self.source_model_digest,
            "original_contract_id": self.original_contract_id,
            "original_hard_constraint_digest": self.original_hard_constraint_digest,
            "effective_contract_id": self.effective_contract_id,
            "explanation_outcome": self.explanation_outcome.value,
            "measured_candidate_ids": list(self.measured_candidate_ids),
            "retained_evidence_ids": list(self.retained_evidence_ids),
            "presented_alternative_ids": list(self.presented_alternative_ids),
            "presented_alternative_statuses": list(self.presented_alternative_statuses),
            "claimed_alternative_count": self.claimed_alternative_count,
            "amendment_attempted": self.amendment_attempted,
            "recovery_target_expected": self.recovery_target_expected,
            "amendment_traceable": self.amendment_traceable,
            "amendment_accurate": self.amendment_accurate,
            "amendment_id": self.amendment_id,
            "amendment_diff_id": self.amendment_diff_id,
            "amendment_application": (
                None if self.amendment_application is None else dict(self.amendment_application)
            ),
            "constraint_preserved": self.constraint_preserved,
            "feasible_target_recovered": self.feasible_target_recovered,
            "misleading_claim": self.misleading_claim,
            "interaction_cost": self.interaction_cost,
            "negative": self.negative,
            "inconclusive": self.inconclusive,
            "shippable": self.shippable,
            "trace_events": list(self.trace_events),
            "error": self.error,
            "silent_constraint_change": self.silent_constraint_change,
        }
        if include_identity:
            record["cell_id"] = self.cell_id
        return record


@dataclass(frozen=True, slots=True)
class NegotiationStudyMetrics:
    """Decision-quality measurements for one policy across retained cells."""

    policy: NegotiationPolicy
    cell_count: int
    constraint_preservation_hits: int
    measured_alternative_grounding_hits: int
    measured_alternative_claims: int
    amendment_attempts: int
    amendment_traceability_hits: int
    amendment_accuracy_hits: int
    recoverable_targets: int
    feasible_target_recovery_hits: int
    misleading_claims: int
    alternative_claims: int
    interaction_cost_total: int
    negative_cells: int
    inconclusive_cells: int
    silent_constraint_changes: int
    errors: int

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    @property
    def constraint_preservation_rate(self) -> float:
        return self._rate(self.constraint_preservation_hits, self.cell_count)

    @property
    def measured_alternative_grounding_rate(self) -> float:
        return self._rate(
            self.measured_alternative_grounding_hits, self.measured_alternative_claims
        )

    @property
    def amendment_traceability_rate(self) -> float:
        return self._rate(self.amendment_traceability_hits, self.amendment_attempts)

    @property
    def amendment_accuracy_rate(self) -> float:
        return self._rate(self.amendment_accuracy_hits, self.amendment_attempts)

    @property
    def feasible_target_recovery_rate(self) -> float:
        return self._rate(self.feasible_target_recovery_hits, self.recoverable_targets)

    @property
    def misleading_claim_rate(self) -> float:
        return self._rate(self.misleading_claims, self.alternative_claims)

    @property
    def mean_interaction_cost(self) -> float:
        return self._rate(self.interaction_cost_total, self.cell_count)

    def to_record(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "cell_count": self.cell_count,
            "constraint_preservation_hits": self.constraint_preservation_hits,
            "constraint_preservation_rate": self.constraint_preservation_rate,
            "measured_alternative_grounding_hits": self.measured_alternative_grounding_hits,
            "measured_alternative_claims": self.measured_alternative_claims,
            "measured_alternative_grounding_rate": self.measured_alternative_grounding_rate,
            "amendment_attempts": self.amendment_attempts,
            "amendment_traceability_hits": self.amendment_traceability_hits,
            "amendment_traceability_rate": self.amendment_traceability_rate,
            "amendment_accuracy_hits": self.amendment_accuracy_hits,
            "amendment_accuracy_rate": self.amendment_accuracy_rate,
            "recoverable_targets": self.recoverable_targets,
            "feasible_target_recovery_hits": self.feasible_target_recovery_hits,
            "feasible_target_recovery_rate": self.feasible_target_recovery_rate,
            "misleading_claims": self.misleading_claims,
            "alternative_claims": self.alternative_claims,
            "misleading_claim_rate": self.misleading_claim_rate,
            "interaction_cost_total": self.interaction_cost_total,
            "mean_interaction_cost": self.mean_interaction_cost,
            "negative_cells": self.negative_cells,
            "inconclusive_cells": self.inconclusive_cells,
            "silent_constraint_changes": self.silent_constraint_changes,
            "errors": self.errors,
        }


@dataclass(frozen=True, slots=True)
class NegotiationStudyRun:
    """Complete retained run and policy measurements."""

    protocol_id: str
    corpus_id: str
    corpus_revision: str
    limits: NegotiationStudyLimits
    policies: tuple[NegotiationPolicy, ...]
    results: tuple[NegotiationStudyResult, ...]
    metrics: tuple[NegotiationStudyMetrics, ...]
    stopped: bool = False
    stop_reason: str | None = None
    schema_version: int = NEGOTIATION_STUDY_SCHEMA_VERSION

    @property
    def run_id(self) -> str:
        return "negotiation_study_run_" + _digest(self.to_record(include_identity=False))[7:]

    @property
    def passed(self) -> bool:
        return not self.stopped and self.policy_passed(NegotiationPolicy.MEASURED_PARETO)

    def policy_passed(self, policy: NegotiationPolicy) -> bool:
        if policy is NegotiationPolicy.PREDICTION_ONLY:
            return False
        metrics = next(item for item in self.metrics if item.policy is policy)
        return (
            metrics.errors == 0
            and metrics.silent_constraint_changes == 0
            and metrics.constraint_preservation_rate == 1.0
            and metrics.misleading_claim_rate == 0.0
            and (
                policy is NegotiationPolicy.REFUSAL_ONLY
                or metrics.measured_alternative_grounding_rate == 1.0
            )
            and (metrics.amendment_attempts == 0 or metrics.amendment_traceability_rate == 1.0)
        )

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "limits": self.limits.to_record(),
            "policies": [item.value for item in self.policies],
            "passed": self.passed,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "results": [item.to_record() for item in self.results],
            "metrics": [item.to_record() for item in self.metrics],
        }
        if include_identity:
            record["run_id"] = self.run_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


def _require_keys(record: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(record) != keys:
        raise NegotiationStudyError(f"{label} has missing or unknown fields")


def _plugin(value: object, label: str) -> ObjectivePluginBinding | None:
    from modelsurgeon.search import ObjectivePluginBinding

    if value is None:
        return None
    record = _object(value, label)
    _require_keys(record, {"name", "plugin_id", "capability", "config_digest", "trust_mode"}, label)
    return ObjectivePluginBinding(
        _text(record["name"], f"{label}.name"),
        _text(record["plugin_id"], f"{label}.plugin_id"),
        _text(record["capability"], f"{label}.capability"),
        _text(record["config_digest"], f"{label}.config_digest"),
        _text(record["trust_mode"], f"{label}.trust_mode"),
    )


def _contract(value: object, label: str) -> ObjectiveContract:
    from modelsurgeon.search import (
        ContractConstraintDirection,
        ContractObjectiveDirection,
        HardConstraint,
        MetricUnit,
        ObjectiveApprovalPolicy,
        ObjectiveContract,
        ObjectiveMode,
        SoftObjective,
    )
    from modelsurgeon.search import (
        ContractObjectiveNormalization as ObjectiveNormalization,
    )

    record = _object(value, label)
    _require_keys(
        record,
        {"schema_version", "contract_id", "mode", "constraints", "objectives", "approval_policy"},
        label,
    )
    constraints: list[HardConstraint] = []
    for index, raw in enumerate(_array(record["constraints"], f"{label}.constraints")):
        item = _object(raw, f"{label}.constraints[{index}]")
        _require_keys(item, {"metric", "direction", "threshold", "unit", "baseline"}, "constraint")
        constraints.append(
            HardConstraint(
                _text(item["metric"], "constraint.metric"),
                ContractConstraintDirection(_text(item["direction"], "constraint.direction")),
                _float(item["threshold"], "constraint.threshold"),
                MetricUnit(_text(item["unit"], "constraint.unit")),
                _text(item["baseline"], "constraint.baseline"),
            )
        )
    objectives: list[SoftObjective] = []
    for index, raw in enumerate(_array(record["objectives"], f"{label}.objectives")):
        item = _object(raw, f"{label}.objectives[{index}]")
        _require_keys(
            item,
            {
                "metric",
                "direction",
                "unit",
                "weight",
                "normalization",
                "baseline",
                "minimum",
                "maximum",
                "plugin",
            },
            "objective",
        )
        objectives.append(
            SoftObjective(
                _text(item["metric"], "objective.metric"),
                ContractObjectiveDirection(_text(item["direction"], "objective.direction")),
                MetricUnit(_text(item["unit"], "objective.unit")),
                _float(item["weight"], "objective.weight"),
                ObjectiveNormalization(_text(item["normalization"], "objective.normalization")),
                None
                if item["baseline"] is None
                else _float(item["baseline"], "objective.baseline"),
                None if item["minimum"] is None else _float(item["minimum"], "objective.minimum"),
                None if item["maximum"] is None else _float(item["maximum"], "objective.maximum"),
                _plugin(item["plugin"], "objective.plugin"),
            )
        )
    approval = _object(record["approval_policy"], f"{label}.approval_policy")
    _require_keys(
        approval,
        {"require_execution_approval", "require_custom_plugin_approval", "allowed_plugin_names"},
        "approval policy",
    )
    contract = ObjectiveContract(
        tuple(constraints),
        tuple(objectives),
        ObjectiveMode(_text(record["mode"], "contract.mode")),
        ObjectiveApprovalPolicy(
            bool(approval["require_execution_approval"]),
            bool(approval["require_custom_plugin_approval"]),
            tuple(
                _text(item, "allowed plugin name")
                for item in _array(approval["allowed_plugin_names"], "allowed plugin names")
            ),
        ),
        _int(record["schema_version"], "contract.schema_version"),
    )
    if _text(record["contract_id"], "contract ID") != contract.contract_id:
        raise NegotiationStudyError(f"{label} contract identity does not match its contents")
    return contract


def _observation(value: object, label: str) -> MetricObservation:
    from modelsurgeon.search import MetricObservation, MetricUnit

    record = _object(value, label)
    _require_keys(
        record, {"metric", "value", "unit", "lower", "upper", "baseline", "evidence"}, label
    )
    return MetricObservation(
        _text(record["metric"], f"{label}.metric"),
        _float(record["value"], f"{label}.value"),
        MetricUnit(_text(record["unit"], f"{label}.unit")),
        None if record["lower"] is None else _float(record["lower"], f"{label}.lower"),
        None if record["upper"] is None else _float(record["upper"], f"{label}.upper"),
        None if record["baseline"] is None else _float(record["baseline"], f"{label}.baseline"),
        _sorted_unique(
            tuple(
                _text(item, f"{label}.evidence")
                for item in _array(record["evidence"], f"{label}.evidence")
            ),
            f"{label}.evidence",
        ),
    )


def _candidate(value: object, label: str) -> FeasibilityCandidateEvidence:
    from modelsurgeon.explain import (
        CandidateDisposition,
        CandidateEvidenceStatus,
        FeasibilityCandidateEvidence,
    )

    record = _object(value, label)
    _require_keys(
        record,
        {
            "schema_version",
            "candidate_id",
            "evidence_id",
            "status",
            "source_model_digest",
            "observations",
            "predicted_observations",
            "disposition",
            "evaluation_id",
            "provenance",
            "resource_usage",
        },
        label,
    )
    raw_resource = record["resource_usage"]
    resource = None
    if raw_resource is not None:
        resource_record = _object(raw_resource, f"{label}.resource_usage")
        _require_keys(
            resource_record,
            {"wall_seconds", "memory_bytes", "output_bytes", "evaluation_count"},
            "resource usage",
        )
        from modelsurgeon.explain.infeasibility import ResourceObservation

        resource = ResourceObservation(
            None
            if resource_record["wall_seconds"] is None
            else _float(resource_record["wall_seconds"], "wall_seconds"),
            None
            if resource_record["memory_bytes"] is None
            else _int(resource_record["memory_bytes"], "memory_bytes"),
            None
            if resource_record["output_bytes"] is None
            else _int(resource_record["output_bytes"], "output_bytes"),
            None
            if resource_record["evaluation_count"] is None
            else _int(resource_record["evaluation_count"], "evaluation_count"),
        )
    return FeasibilityCandidateEvidence(
        _identifier(record["candidate_id"], f"{label}.candidate_id", "candidate_"),
        _identifier(record["evidence_id"], f"{label}.evidence_id", "evidence_"),
        CandidateEvidenceStatus(_text(record["status"], f"{label}.status")),
        _text(record["source_model_digest"], f"{label}.source_model_digest"),
        tuple(
            sorted(
                (
                    _observation(item, f"{label}.observations")
                    for item in _array(record["observations"], f"{label}.observations")
                ),
                key=lambda item: item.metric,
            )
        ),
        tuple(
            sorted(
                (
                    _observation(item, f"{label}.predicted_observations")
                    for item in _array(
                        record["predicted_observations"], f"{label}.predicted_observations"
                    )
                ),
                key=lambda item: item.metric,
            )
        ),
        CandidateDisposition(_text(record["disposition"], f"{label}.disposition")),
        None
        if record["evaluation_id"] is None
        else _identifier(record["evaluation_id"], f"{label}.evaluation_id", "evaluation_"),
        _mapping(record["provenance"], f"{label}.provenance"),
        resource,
    )


def load_negotiation_study(path: Path) -> NegotiationStudyCorpus:
    """Load and validate a versioned fixture campaign protocol."""

    try:
        if path.stat().st_size > MAX_NEGOTIATION_STUDY_BYTES:
            raise NegotiationStudyError("negotiation study exceeds its byte budget")
        root = json.loads(path.read_text(encoding="utf-8"))
    except NegotiationStudyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NegotiationStudyError("negotiation study is not valid UTF-8 JSON") from error
    record = _object(root, "negotiation study")
    _require_keys(
        record,
        {
            "schema_version",
            "protocol_id",
            "corpus_id",
            "corpus_revision",
            "limits",
            "policies",
            "scenarios",
        },
        "negotiation study",
    )
    if _text(record["protocol_id"], "protocol_id") != NEGOTIATION_STUDY_PROTOCOL_ID:
        raise NegotiationStudyError("unsupported negotiation study protocol")
    limits_record = _object(record["limits"], "study limits")
    _require_keys(
        limits_record,
        {"max_scenarios", "fixed_evaluation_count", "max_seeds_per_scenario", "max_output_bytes"},
        "study limits",
    )
    limits = NegotiationStudyLimits(
        _int(limits_record["max_scenarios"], "max_scenarios", minimum=1),
        _int(limits_record["fixed_evaluation_count"], "fixed_evaluation_count", minimum=1),
        _int(limits_record["max_seeds_per_scenario"], "max_seeds_per_scenario", minimum=1),
        _int(limits_record["max_output_bytes"], "max_output_bytes", minimum=1),
    )
    scenarios: list[NegotiationStudyScenario] = []
    for index, raw in enumerate(_array(record["scenarios"], "scenarios")):
        item = _object(raw, f"scenarios[{index}]")
        _require_keys(
            item,
            {
                "scenario_id",
                "split",
                "model_fixture",
                "source_model_digest",
                "campaign_id",
                "session_id",
                "run_id",
                "fixed_evaluation_count",
                "seeds",
                "original_contract",
                "proposed_contract",
                "candidates",
                "kind",
                "target_candidate_id",
                "target_recovery_expected",
                "negative",
                "inconclusive",
            },
            "scenario",
        )
        scenarios.append(
            NegotiationStudyScenario(
                _text(item["scenario_id"], "scenario_id"),
                NegotiationSplit(_text(item["split"], "scenario split")),
                _text(item["model_fixture"], "model_fixture"),
                _text(item["source_model_digest"], "source_model_digest"),
                _text(item["campaign_id"], "campaign_id"),
                _text(item["session_id"], "session_id"),
                _text(item["run_id"], "run_id"),
                _int(item["fixed_evaluation_count"], "fixed_evaluation_count", minimum=1),
                tuple(_int(seed, "seed") for seed in _array(item["seeds"], "seeds")),
                _contract(item["original_contract"], "original_contract"),
                tuple(
                    sorted(
                        (
                            _candidate(candidate, f"scenarios[{index}].candidate")
                            for candidate in _array(item["candidates"], "candidates")
                        ),
                        key=lambda candidate: candidate.candidate_id,
                    )
                ),
                NegotiationScenarioKind(_text(item["kind"], "scenario kind")),
                None
                if item["target_candidate_id"] is None
                else _identifier(item["target_candidate_id"], "target_candidate_id", "candidate_"),
                None
                if item["proposed_contract"] is None
                else _contract(item["proposed_contract"], "proposed_contract"),
                bool(item["target_recovery_expected"]),
                bool(item["negative"]),
                bool(item["inconclusive"]),
            )
        )
    policies = tuple(
        NegotiationPolicy(_text(item, "policy")) for item in _array(record["policies"], "policies")
    )
    return NegotiationStudyCorpus(
        _text(record["corpus_id"], "corpus_id"),
        _text(record["corpus_revision"], "corpus_revision"),
        limits,
        tuple(scenarios),
        policies,
        _text(record["protocol_id"], "protocol_id"),
        _int(record["schema_version"], "schema_version"),
    )


class _SilentConstraintChange(NegotiationStudyError):
    """Internal stop signal; the run must not continue after this event."""


def _seeded_order(scenario: NegotiationStudyScenario, seed: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            (item.candidate_id for item in scenario.candidates),
            key=lambda candidate_id: hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest(),
        )
    )


def _hard_constraint_digest(contract: ObjectiveContract) -> str:
    return _digest([item.to_record() for item in contract.constraints])


def _candidate_is_feasible(
    contract: ObjectiveContract, candidate: FeasibilityCandidateEvidence
) -> bool:
    from modelsurgeon.search.objective_contract import evaluate_contract

    return evaluate_contract(contract, candidate.observations).feasible


def _run_cell(
    policy: NegotiationPolicy,
    scenario: NegotiationStudyScenario,
    seed: int,
    limits: NegotiationStudyLimits,
) -> NegotiationStudyResult:
    from modelsurgeon.explain import (
        CandidateEvidenceStatus,
        CanonicalEvidenceArchive,
        FeasibilityOutcome,
        FeasibilityProvenance,
        ParetoResourceBounds,
        build_feasibility_explanation,
        build_pareto_alternatives,
    )
    from modelsurgeon.search import (
        apply_objective_amendment,
        approve_objective_amendment,
        propose_objective_amendment,
    )

    order = _seeded_order(scenario, seed)
    archive = CanonicalEvidenceArchive.build(scenario.original_contract, scenario.candidates)
    provenance = FeasibilityProvenance(
        scenario.original_contract.contract_id,
        scenario.source_model_digest,
        archive.archive_id,
        evidence_ids=tuple(item.evidence_id for item in archive.candidates),
    )
    explanation = build_feasibility_explanation(
        scenario.original_contract,
        archive,
        provenance=provenance,
    )
    expected_outcome = FeasibilityOutcome(scenario.kind.value)
    error: str | None = None
    if explanation.outcome is not expected_outcome:
        error = f"expected {expected_outcome.value}, observed {explanation.outcome.value}"
    original_hard_digest = _hard_constraint_digest(scenario.original_contract)
    constraint_preserved = explanation.hard_constraints == scenario.original_contract.constraints
    if not constraint_preserved:
        raise _SilentConstraintChange("study stopped: original hard constraints changed")

    measured_ids = tuple(sorted(explanation.measured_candidate_ids))
    retained_ids = tuple(sorted(item.evidence_id for item in scenario.candidates))
    alternative_ids: tuple[str, ...] = ()
    alternative_statuses: tuple[str, ...] = ()
    amendment_attempted = False
    amendment_traceable = False
    amendment_accurate = False
    amendment_id: str | None = None
    amendment_diff_id: str | None = None
    amendment_application: Mapping[str, object] | None = None
    effective_contract = scenario.original_contract
    target_recovered = False
    misleading_claim = False
    trace = [f"evaluate:{scenario.fixed_evaluation_count}", f"seed-order:{','.join(order)}"]
    interaction_cost = 1
    shippable = policy is not NegotiationPolicy.PREDICTION_ONLY

    if policy is NegotiationPolicy.REFUSAL_ONLY:
        if explanation.outcome is FeasibilityOutcome.INFEASIBLE:
            trace.append("refuse:measured-infeasible")
        else:
            trace.append(f"refuse:{explanation.outcome.value}")
    elif policy is NegotiationPolicy.MEASURED_PARETO:
        pareto = build_pareto_alternatives(
            scenario.original_contract,
            archive,
            provenance=provenance,
            resource_bounds=ParetoResourceBounds(
                max_candidates=limits.fixed_evaluation_count,
                max_alternatives=limits.fixed_evaluation_count,
                max_frontier=limits.fixed_evaluation_count,
                max_output_bytes=limits.max_output_bytes,
            ),
        )
        alternative_ids = tuple(item.candidate_id for item in pareto.alternatives)
        alternative_statuses = tuple(item.status.value for item in pareto.alternatives)
        trace.append("pareto:measured:" + ",".join(alternative_ids))
        interaction_cost += 1
        if (
            scenario.proposed_contract is not None
            and explanation.outcome is FeasibilityOutcome.INFEASIBLE
        ):
            amendment_attempted = True
            amendment = propose_objective_amendment(
                scenario.original_contract,
                scenario.proposed_contract,
                rationale="measured fixture near-miss evidence supports an explicit trade-off",
                evidence=explanation,
                operator_id="fixture-operator",
                requested_at="2026-09-07T10:00:00+00:00",
                expires_at="2026-09-07T11:00:00+00:00",
                parent_campaign_id=scenario.campaign_id,
                session_id=scenario.session_id,
                run_id=scenario.run_id,
                operator_context={
                    "protocol_id": NEGOTIATION_STUDY_PROTOCOL_ID,
                    "seed": str(seed),
                },
            )
            amendment_id = amendment.amendment_id
            amendment_diff_id = amendment.diff.diff_id
            approved = approve_objective_amendment(
                amendment,
                operator_id="fixture-operator",
                decided_at="2026-09-07T10:05:00+00:00",
            )
            application = apply_objective_amendment(
                approved,
                current_objective=scenario.original_contract,
                current_campaign_id=scenario.campaign_id,
                applied_at="2026-09-07T10:06:00+00:00",
            )
            effective_contract = scenario.proposed_contract
            amendment_application = application.to_record()
            amendment_traceable = (
                approved.status.value == "approved"
                and application.original_spec_identity == scenario.original_contract.contract_id
                and application.effective_spec_identity == scenario.proposed_contract.contract_id
                and application.evidence_archive_id == explanation.provenance.archive_id
                and application.preserved_evidence_ids == retained_ids
                and application.original_campaign_id == scenario.campaign_id
                and application.downstream_campaign_id != scenario.campaign_id
                and amendment.hard_constraints == scenario.original_contract.constraints
            )
            amendment_accurate = amendment_traceable and amendment.diff.hard_constraints_changed
            trace.extend(
                (
                    f"amendment:proposed:{amendment_id}",
                    f"amendment:approved:{approved.approval_id}",
                    f"amendment:applied:{application.downstream_campaign_id}",
                )
            )
            interaction_cost += 3
            if scenario.target_candidate_id is not None:
                target = next(
                    item
                    for item in scenario.candidates
                    if item.candidate_id == scenario.target_candidate_id
                )
                target_recovered = amendment_traceable and _candidate_is_feasible(
                    effective_contract, target
                )
    else:
        predicted_ids = tuple(
            sorted(
                item.candidate_id
                for item in scenario.candidates
                if item.status is CandidateEvidenceStatus.PREDICTED
            )
        )
        alternative_ids = predicted_ids
        alternative_statuses = tuple("predicted" for _ in predicted_ids)
        trace.append("prediction-only:" + ",".join(predicted_ids))
        interaction_cost += 1 if predicted_ids else 0
        misleading_claim = bool(predicted_ids)
        shippable = False

    if effective_contract.constraints != scenario.original_contract.constraints:
        if not amendment_traceable:
            raise _SilentConstraintChange("study stopped: effective constraints changed silently")
        if amendment_application is None:
            raise _SilentConstraintChange(
                "study stopped: amended contract has no application trace"
            )
    if policy is NegotiationPolicy.MEASURED_PARETO and alternative_ids:
        measured_set = set(measured_ids)
        if not set(alternative_ids) <= measured_set:
            raise _SilentConstraintChange(
                "study stopped: non-measured alternative entered measured Pareto"
            )
    if policy is NegotiationPolicy.PREDICTION_ONLY and any(
        status == "measured" for status in alternative_statuses
    ):
        raise _SilentConstraintChange(
            "study stopped: prediction-only alternative was marked measured"
        )
    if scenario.target_recovery_expected and policy is not NegotiationPolicy.MEASURED_PARETO:
        target_recovered = False
    return NegotiationStudyResult(
        policy,
        scenario.scenario_id,
        scenario.split,
        seed,
        scenario.source_model_digest,
        scenario.original_contract.contract_id,
        original_hard_digest,
        effective_contract.contract_id,
        explanation.outcome,
        measured_ids,
        retained_ids,
        alternative_ids,
        alternative_statuses,
        len(alternative_ids),
        amendment_attempted,
        scenario.target_recovery_expected,
        amendment_traceable,
        amendment_accurate,
        amendment_id,
        amendment_diff_id,
        amendment_application,
        constraint_preserved,
        target_recovered,
        misleading_claim,
        interaction_cost,
        scenario.negative,
        scenario.inconclusive,
        shippable,
        tuple(trace),
        error,
        False,
    )


def _metrics(
    policy: NegotiationPolicy, results: Sequence[NegotiationStudyResult]
) -> NegotiationStudyMetrics:
    selected = tuple(item for item in results if item.policy is policy)
    return NegotiationStudyMetrics(
        policy,
        len(selected),
        sum(item.constraint_preserved for item in selected),
        sum(
            item.claimed_alternative_count
            for item in selected
            if item.measured_alternative_grounding
        ),
        sum(
            item.claimed_alternative_count
            for item in selected
            if item.policy is not NegotiationPolicy.PREDICTION_ONLY
        ),
        sum(item.amendment_attempted for item in selected),
        sum(item.amendment_traceable for item in selected if item.amendment_attempted),
        sum(item.amendment_accurate for item in selected if item.amendment_attempted),
        sum(item.recovery_target_expected for item in selected),
        sum(item.feasible_target_recovered for item in selected),
        sum(item.misleading_claim for item in selected),
        sum(item.claimed_alternative_count for item in selected),
        sum(item.interaction_cost for item in selected),
        sum(item.negative for item in selected),
        sum(item.inconclusive for item in selected),
        sum(item.silent_constraint_change for item in selected),
        sum(item.error is not None for item in selected),
    )


def run_negotiation_study(
    corpus: NegotiationStudyCorpus,
    policies: Sequence[NegotiationPolicy] | None = None,
) -> NegotiationStudyRun:
    """Replay every bounded policy/scenario/seed cell in canonical order."""

    from modelsurgeon.explain import FeasibilityOutcome

    selected_values = corpus.policies if policies is None else tuple(policies)
    selected = tuple(sorted(set(selected_values), key=lambda item: item.value))
    if not selected or any(not isinstance(item, NegotiationPolicy) for item in selected):
        raise NegotiationStudyError("study requires known policies")
    results: list[NegotiationStudyResult] = []
    for policy in selected:
        for scenario in corpus.scenarios:
            for seed in scenario.seeds:
                try:
                    results.append(_run_cell(policy, scenario, seed, corpus.limits))
                except _SilentConstraintChange as error:
                    raise NegotiationStudyError(str(error)) from error
                except Exception as error:
                    results.append(
                        NegotiationStudyResult(
                            policy,
                            scenario.scenario_id,
                            scenario.split,
                            seed,
                            scenario.source_model_digest,
                            scenario.original_contract.contract_id,
                            _hard_constraint_digest(scenario.original_contract),
                            scenario.original_contract.contract_id,
                            FeasibilityOutcome.UNKNOWN,
                            (),
                            tuple(sorted(item.evidence_id for item in scenario.candidates)),
                            (),
                            (),
                            0,
                            False,
                            scenario.target_recovery_expected,
                            False,
                            False,
                            None,
                            None,
                            None,
                            True,
                            False,
                            False,
                            1,
                            scenario.negative,
                            scenario.inconclusive,
                            policy is not NegotiationPolicy.PREDICTION_ONLY,
                            (f"error:{type(error).__name__}",),
                            str(error),
                            False,
                        )
                    )
    retained = tuple(results)
    metrics = tuple(_metrics(policy, retained) for policy in selected)
    return NegotiationStudyRun(
        corpus.protocol_id,
        corpus.corpus_id,
        corpus.corpus_revision,
        corpus.limits,
        selected,
        retained,
        metrics,
    )


def load_and_run_negotiation_study(path: Path) -> NegotiationStudyRun:
    """Load a fixture protocol and replay it."""

    return run_negotiation_study(load_negotiation_study(path))


__all__ = [
    "MAX_NEGOTIATION_STUDY_BYTES",
    "NEGOTIATION_STUDY_PROTOCOL_ID",
    "NEGOTIATION_STUDY_SCHEMA_VERSION",
    "NegotiationPolicy",
    "NegotiationScenarioKind",
    "NegotiationSplit",
    "NegotiationStudyCorpus",
    "NegotiationStudyError",
    "NegotiationStudyLimits",
    "NegotiationStudyMetrics",
    "NegotiationStudyResult",
    "NegotiationStudyRun",
    "NegotiationStudyScenario",
    "load_and_run_negotiation_study",
    "load_negotiation_study",
    "run_negotiation_study",
]
