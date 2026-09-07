"""Bounded, calibrated surrogate acquisition for complete architecture states."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from enum import StrEnum

from modelsurgeon.config import ObjectiveDirection, OptimizeMetric
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.architecture_policies import (
    ArchitectureEvidenceStatus,
    CompleteArchitectureState,
)
from modelsurgeon.search.constraints import ConstraintSet
from modelsurgeon.search.objectives import ObjectiveSet
from modelsurgeon.search.pareto import (
    ParetoCandidate,
    ParetoObjectiveValue,
    conservatively_dominates,
)

SURROGATE_SCHEMA_VERSION = 1


class SurrogateError(ValueError):
    """Raised when surrogate evidence, budgets, or persisted records are unsafe."""


class AcquisitionKind(StrEnum):
    EXPECTED_IMPROVEMENT = "expected_improvement"
    HYPERVOLUME_IMPROVEMENT = "hypervolume_improvement"


class SurrogateFitStatus(StrEnum):
    FITTED = "fitted"
    INSUFFICIENT_DATA = "insufficient_data"
    UNCALIBRATED = "uncalibrated"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNSUPPORTED = "unsupported"


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise SurrogateError(f"{label} must be finite")
    return value


def _stable_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    digest = hashlib.sha256(canonical_identity_json(value).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _tie_rank(seed: int, index: int, candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{index}:{candidate_id}".encode()).digest(), "big")


@dataclass(frozen=True, slots=True)
class SurrogateBudget:
    max_training_examples: int = 256
    max_fit_operations: int = 100_000
    max_model_bytes: int = 4_000_000
    max_fit_time_ms: int = 5_000

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_training_examples, "training example budget"),
            (self.max_fit_operations, "fit operation budget"),
            (self.max_model_bytes, "model memory budget"),
            (self.max_fit_time_ms, "fit time budget"),
        ):
            if isinstance(value, bool) or value <= 0:
                raise SurrogateError(f"{label} must be positive")

    def to_record(self) -> dict[str, int]:
        return {
            "max_training_examples": self.max_training_examples,
            "max_fit_operations": self.max_fit_operations,
            "max_model_bytes": self.max_model_bytes,
            "max_fit_time_ms": self.max_fit_time_ms,
        }


@dataclass(frozen=True, slots=True)
class SurrogateConfig:
    acquisition: AcquisitionKind
    evaluation_budget: int
    batch_size: int = 1
    minimum_training_examples: int = 4
    calibration_fraction: float = 0.25
    exploration_weight: float = 1.0
    minimum_feasible_probability: float = 0.5
    seed: int = 0
    budget: SurrogateBudget = SurrogateBudget()

    def __post_init__(self) -> None:
        for value, label in (
            (self.evaluation_budget, "evaluation budget"),
            (self.batch_size, "batch size"),
            (self.minimum_training_examples, "minimum training examples"),
        ):
            if isinstance(value, bool) or value <= 0:
                raise SurrogateError(f"{label} must be positive")
        if not 0 < self.calibration_fraction < 1:
            raise SurrogateError("calibration fraction must be within (0, 1)")
        if not math.isfinite(self.exploration_weight) or self.exploration_weight < 0:
            raise SurrogateError("exploration weight must be finite and non-negative")
        if not 0 <= self.minimum_feasible_probability <= 1:
            raise SurrogateError("minimum feasible probability must be within [0, 1]")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise SurrogateError("surrogate seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "acquisition": self.acquisition.value,
            "evaluation_budget": self.evaluation_budget,
            "batch_size": self.batch_size,
            "minimum_training_examples": self.minimum_training_examples,
            "calibration_fraction": self.calibration_fraction,
            "exploration_weight": self.exploration_weight,
            "minimum_feasible_probability": self.minimum_feasible_probability,
            "seed": self.seed,
            "budget": self.budget.to_record(),
        }


@dataclass(frozen=True, slots=True)
class SurrogateTrainingExample:
    candidate_id: str
    features: tuple[float, ...]
    objectives: tuple[ParetoObjectiveValue, ...]
    feasible: bool

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.features:
            raise SurrogateError("surrogate examples require an ID and feature vector")
        if any(not math.isfinite(value) for value in self.features):
            raise SurrogateError("surrogate features must be finite")
        if not self.objectives:
            raise SurrogateError("surrogate examples require objective values")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "features": list(self.features),
            "objectives": [item.to_record() for item in self.objectives],
            "feasible": self.feasible,
        }


@dataclass(frozen=True, slots=True)
class SurrogatePrediction:
    candidate_id: str
    objectives: tuple[ParetoObjectiveValue, ...]
    feasible_probability: float
    distance: float
    uncertainty: float

    def __post_init__(self) -> None:
        if not 0 <= self.feasible_probability <= 1:
            raise SurrogateError("surrogate feasibility probability must be within [0, 1]")
        if self.distance < 0 or self.uncertainty < 0:
            raise SurrogateError("surrogate distance and uncertainty cannot be negative")
        _finite(self.distance, "surrogate distance")
        _finite(self.uncertainty, "surrogate uncertainty")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "objectives": [item.to_record() for item in self.objectives],
            "feasible_probability": self.feasible_probability,
            "distance": self.distance,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class SurrogateModel:
    """RBF-compatible mixed-space surrogate with calibrated residual radius."""

    model_id: str
    feature_schema: tuple[str, ...]
    feature_mins: tuple[float, ...]
    feature_maxs: tuple[float, ...]
    examples: tuple[SurrogateTrainingExample, ...]
    objective_metrics: tuple[OptimizeMetric, ...]
    length_scale: float
    residual_radius: float
    feasibility_rate: float

    def __post_init__(self) -> None:
        width = len(self.feature_schema)
        if width == 0 or len(self.feature_mins) != width or len(self.feature_maxs) != width:
            raise SurrogateError("surrogate feature metadata must align")
        if not self.examples or not self.objective_metrics:
            raise SurrogateError("surrogate models require training examples and targets")
        if self.length_scale <= 0 or self.residual_radius < 0:
            raise SurrogateError("surrogate scales must be positive and finite")
        _finite(self.length_scale, "surrogate length scale")
        _finite(self.residual_radius, "surrogate residual radius")
        if not 0 <= self.feasibility_rate <= 1:
            raise SurrogateError("surrogate feasibility rate must be within [0, 1]")

    @property
    def estimated_bytes(self) -> int:
        return (len(self.examples) * len(self.feature_schema) * 8) + (len(self.examples) * 64)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "feature_schema": list(self.feature_schema),
            "feature_mins": list(self.feature_mins),
            "feature_maxs": list(self.feature_maxs),
            "examples": [item.to_record() for item in self.examples],
            "objective_metrics": [item.value for item in self.objective_metrics],
            "length_scale": self.length_scale,
            "residual_radius": self.residual_radius,
            "feasibility_rate": self.feasibility_rate,
        }

    @classmethod
    def from_record(cls, record: object) -> SurrogateModel:
        if not isinstance(record, dict):
            raise SurrogateError("surrogate model must be an object")
        required = {
            "schema_version",
            "model_id",
            "feature_schema",
            "feature_mins",
            "feature_maxs",
            "examples",
            "objective_metrics",
            "length_scale",
            "residual_radius",
            "feasibility_rate",
        }
        if set(record) != required or record["schema_version"] != SURROGATE_SCHEMA_VERSION:
            raise SurrogateError("unsupported or malformed surrogate model")
        schema = record["feature_schema"]
        mins = record["feature_mins"]
        maxs = record["feature_maxs"]
        metrics = record["objective_metrics"]
        raw_examples = record["examples"]
        if (
            not isinstance(record["model_id"], str)
            or not isinstance(schema, list)
            or not all(isinstance(item, str) for item in schema)
            or not isinstance(mins, list)
            or not isinstance(maxs, list)
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in mins
            )
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in maxs
            )
            or not isinstance(metrics, list)
            or not all(isinstance(item, str) for item in metrics)
            or not isinstance(raw_examples, list)
        ):
            raise SurrogateError("surrogate model fields have invalid types")
        examples: list[SurrogateTrainingExample] = []
        for raw in raw_examples:
            if not isinstance(raw, dict) or set(raw) != {
                "candidate_id",
                "features",
                "objectives",
                "feasible",
            }:
                raise SurrogateError("surrogate example is malformed")
            features = raw["features"]
            objectives = raw["objectives"]
            if (
                not isinstance(features, list)
                or not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in features
                )
                or not isinstance(objectives, list)
                or not isinstance(raw["feasible"], bool)
            ):
                raise SurrogateError("surrogate example fields have invalid types")
            values: list[ParetoObjectiveValue] = []
            for objective in objectives:
                if not isinstance(objective, dict) or set(objective) != {
                    "metric",
                    "estimate",
                    "confidence_low",
                    "confidence_high",
                }:
                    raise SurrogateError("surrogate objective record is malformed")
                values.append(
                    ParetoObjectiveValue(
                        OptimizeMetric(str(objective["metric"])),
                        float(objective["estimate"]),
                        None
                        if objective["confidence_low"] is None
                        else float(objective["confidence_low"]),
                        None
                        if objective["confidence_high"] is None
                        else float(objective["confidence_high"]),
                    )
                )
            candidate_id = raw["candidate_id"]
            if not isinstance(candidate_id, str):
                raise SurrogateError("surrogate example ID must be text")
            examples.append(
                SurrogateTrainingExample(
                    candidate_id,
                    tuple(float(item) for item in features),
                    tuple(values),
                    raw["feasible"],
                )
            )
        return cls(
            record["model_id"],
            tuple(schema),
            tuple(float(item) for item in mins),
            tuple(float(item) for item in maxs),
            tuple(examples),
            tuple(OptimizeMetric(item) for item in metrics),
            float(record["length_scale"]),
            float(record["residual_radius"]),
            float(record["feasibility_rate"]),
        )

    def _normalize(self, features: tuple[float, ...]) -> tuple[float, ...]:
        if len(features) != len(self.feature_schema):
            raise SurrogateError("candidate feature width does not match surrogate")
        return tuple(
            0.5
            if maximum == minimum
            else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
            for value, minimum, maximum in zip(
                features, self.feature_mins, self.feature_maxs, strict=True
            )
        )

    def predict(self, candidate_id: str, features: tuple[float, ...]) -> SurrogatePrediction:
        normalized = self._normalize(features)
        weighted: list[tuple[float, SurrogateTrainingExample]] = []
        for example in self.examples:
            distance = math.sqrt(
                math.fsum(
                    (left - right) ** 2
                    for left, right in zip(normalized, example.features, strict=True)
                )
            )
            weight = math.exp(-(distance**2) / (2 * self.length_scale**2))
            weighted.append((weight, example))
        weight_sum = math.fsum(weight for weight, _ in weighted)
        if weight_sum <= 0:
            raise SurrogateError("surrogate kernel has no usable weight")
        objectives: list[ParetoObjectiveValue] = []
        uncertainty = self.residual_radius + (1.0 / math.sqrt(weight_sum))
        for metric in self.objective_metrics:
            values = [
                next(item.estimate for item in example.objectives if item.metric is metric)
                for _, example in weighted
            ]
            mean = (
                math.fsum(
                    weight * value for (weight, _), value in zip(weighted, values, strict=True)
                )
                / weight_sum
            )
            radius = max(self.residual_radius, uncertainty * max(1.0, abs(mean)))
            objectives.append(ParetoObjectiveValue(metric, mean, mean - radius, mean + radius))
        feasible = (
            math.fsum(weight * float(example.feasible) for weight, _ in weighted) / weight_sum
        )
        nearest = min(
            math.sqrt(
                math.fsum(
                    (left - right) ** 2
                    for left, right in zip(normalized, example.features, strict=True)
                )
            )
            for _, example in weighted
        )
        return SurrogatePrediction(candidate_id, tuple(objectives), feasible, nearest, uncertainty)


@dataclass(frozen=True, slots=True)
class SurrogateFit:
    status: SurrogateFitStatus
    model: SurrogateModel | None
    measured_candidate_ids: tuple[str, ...]
    retained_outcome_ids: tuple[str, ...]
    held_out_count: int
    calibration_rmse: float | None
    calibration_coverage: float | None
    calibration_feasibility_brier: float | None
    reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "status": self.status.value,
            "model": None if self.model is None else self.model.to_record(),
            "measured_candidate_ids": list(self.measured_candidate_ids),
            "retained_outcome_ids": list(self.retained_outcome_ids),
            "held_out_count": self.held_out_count,
            "calibration_rmse": self.calibration_rmse,
            "calibration_coverage": self.calibration_coverage,
            "calibration_feasibility_brier": self.calibration_feasibility_brier,
            "reason": self.reason,
        }

    @classmethod
    def from_record(cls, record: object) -> SurrogateFit:
        if not isinstance(record, dict):
            raise SurrogateError("surrogate fit must be an object")
        required = {
            "schema_version",
            "status",
            "model",
            "measured_candidate_ids",
            "retained_outcome_ids",
            "held_out_count",
            "calibration_rmse",
            "calibration_coverage",
            "calibration_feasibility_brier",
            "reason",
        }
        if set(record) != required or record["schema_version"] != SURROGATE_SCHEMA_VERSION:
            raise SurrogateError("unsupported or malformed surrogate fit")
        measured = record["measured_candidate_ids"]
        retained = record["retained_outcome_ids"]
        if (
            not isinstance(measured, list)
            or not all(isinstance(item, str) for item in measured)
            or not isinstance(retained, list)
            or not all(isinstance(item, str) for item in retained)
            or not isinstance(record["status"], str)
            or not isinstance(record["held_out_count"], int)
        ):
            raise SurrogateError("surrogate fit fields have invalid types")
        model = None if record["model"] is None else SurrogateModel.from_record(record["model"])
        optional_numbers: list[float | None] = []
        for key in ("calibration_rmse", "calibration_coverage", "calibration_feasibility_brier"):
            value = record[key]
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise SurrogateError(f"surrogate {key} must be numeric or null")
            optional_numbers.append(None if value is None else float(value))
        reason = record["reason"]
        if reason is not None and not isinstance(reason, str):
            raise SurrogateError("surrogate fit reason must be text or null")
        return cls(
            SurrogateFitStatus(record["status"]),
            model,
            tuple(measured),
            tuple(retained),
            record["held_out_count"],
            optional_numbers[0],
            optional_numbers[1],
            optional_numbers[2],
            reason,
        )


def _feature_values(
    state: CompleteArchitectureState,
    axes: tuple[str, ...],
) -> tuple[float, ...]:
    assignments = {axis.value: value for axis, value in state.candidate.assignments}
    return (
        *(_stable_value(assignments.get(axis)) for axis in axes),
        _stable_value(state.candidate.hardware_profile_id),
        float(state.candidate.parameter_count),
        float(state.candidate.storage_bytes),
        state.candidate.source_distance,
    )


def fit_surrogate(
    states: tuple[CompleteArchitectureState, ...],
    objectives: ObjectiveSet,
    constraints: ConstraintSet,
    config: SurrogateConfig,
) -> SurrogateFit:
    """Fit a bounded kernel surrogate and calibrate it on deterministic held-out states."""

    if not states:
        raise SurrogateError("surrogate fitting requires architecture states")
    unique = {state.candidate_id: state for state in states}
    if len(unique) != len(states):
        raise SurrogateError("surrogate states must have unique candidate IDs")
    axes = tuple(
        sorted({axis.value for state in states for axis, _ in state.candidate.assignments})
    )
    feature_schema = (
        *tuple(f"axis:{axis}" for axis in axes),
        "hardware_profile",
        "parameters",
        "storage",
        "source_distance",
    )
    features = {state.candidate_id: _feature_values(state, axes) for state in states}
    measured_all = tuple(
        state for state in states if state.status is ArchitectureEvidenceStatus.MEASURED
    )
    ordered_all = tuple(
        sorted(
            measured_all,
            key=lambda state: (_tie_rank(config.seed, 0, state.candidate_id), state.candidate_id),
        )
    )
    measured = ordered_all[: config.budget.max_training_examples]
    retained = tuple(
        state.candidate_id
        for state in states
        if state.status is not ArchitectureEvidenceStatus.MEASURED
        or state.candidate_id not in {item.candidate_id for item in measured}
    )
    metrics: tuple[OptimizeMetric, ...] = tuple(term.metric for term in objectives.terms)
    for state in measured:
        if not set(metrics) <= {item.metric for item in state.objectives}:
            raise SurrogateError("surrogate measured state is missing an objective")
    if len(measured) < config.minimum_training_examples:
        return SurrogateFit(
            SurrogateFitStatus.INSUFFICIENT_DATA,
            None,
            tuple(state.candidate_id for state in measured),
            retained,
            0,
            None,
            None,
            None,
            "minimum measured training examples not met",
        )
    ordered = measured
    held_out_count = max(1, math.floor(len(ordered) * config.calibration_fraction))
    calibration = tuple(ordered[:held_out_count])
    training = tuple(ordered[held_out_count:])
    if len(training) < 2:
        return SurrogateFit(
            SurrogateFitStatus.UNCALIBRATED,
            None,
            tuple(state.candidate_id for state in measured),
            retained,
            held_out_count,
            None,
            None,
            None,
            "calibration split leaves too few training examples",
        )
    width = len(feature_schema)
    if len(training) * width > config.budget.max_fit_operations:
        return SurrogateFit(
            SurrogateFitStatus.BUDGET_EXCEEDED,
            None,
            tuple(state.candidate_id for state in measured),
            retained,
            held_out_count,
            None,
            None,
            None,
            "fit operation budget exceeded",
        )
    estimated_bytes = len(training) * width * 8 + len(training) * 64
    if estimated_bytes > config.budget.max_model_bytes:
        return SurrogateFit(
            SurrogateFitStatus.BUDGET_EXCEEDED,
            None,
            tuple(state.candidate_id for state in measured),
            retained,
            held_out_count,
            None,
            None,
            None,
            "fit memory budget exceeded",
        )
    started = time.perf_counter()
    raw_features = tuple(features[state.candidate_id] for state in training)
    mins = tuple(min(row[index] for row in raw_features) for index in range(width))
    maxs = tuple(max(row[index] for row in raw_features) for index in range(width))
    normalized = tuple(
        tuple(
            0.5
            if maximum == minimum
            else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
            for value, minimum, maximum in zip(row, mins, maxs, strict=True)
        )
        for row in raw_features
    )
    examples = tuple(
        SurrogateTrainingExample(
            state.candidate_id,
            vector,
            state.objectives,
            constraints.evaluate(state.constraints).passed,
        )
        for state, vector in zip(training, normalized, strict=True)
    )
    radius = 0.05
    model_identity = {
        "schema_version": SURROGATE_SCHEMA_VERSION,
        "features": feature_schema,
        "examples": [item.to_record() for item in examples],
        "metrics": [metric.value for metric in metrics],
        "seed": config.seed,
    }
    model = SurrogateModel(
        f"surrogate_{hashlib.sha256(canonical_identity_json(model_identity).encode()).hexdigest()}",
        feature_schema,
        mins,
        maxs,
        examples,
        metrics,
        0.35,
        radius,
        math.fsum(float(item.feasible) for item in examples) / len(examples),
    )
    errors: list[float] = []
    feasibility_errors: list[float] = []
    covered = 0
    for state in calibration:
        if (time.perf_counter() - started) * 1000 > config.budget.max_fit_time_ms:
            return SurrogateFit(
                SurrogateFitStatus.BUDGET_EXCEEDED,
                None,
                tuple(item.candidate_id for item in measured),
                retained,
                held_out_count,
                None,
                None,
                None,
                "fit time budget exceeded",
            )
        prediction = model.predict(state.candidate_id, features[state.candidate_id])
        actual = {item.metric: item.estimate for item in state.objectives}
        feasibility_errors.append(
            (
                prediction.feasible_probability
                - float(constraints.evaluate(state.constraints).passed)
            )
            ** 2
        )
        errors.extend(
            prediction_value.estimate - actual[metric]
            for prediction_value, metric in zip(prediction.objectives, metrics, strict=True)
        )
        covered_state = True
        for prediction_value, metric in zip(prediction.objectives, metrics, strict=True):
            if prediction_value.confidence_low is None or prediction_value.confidence_high is None:
                covered_state = False
                break
            if (
                not prediction_value.confidence_low
                <= actual[metric]
                <= prediction_value.confidence_high
            ):
                covered_state = False
                break
        if covered_state:
            covered += 1
    rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    model = replace(model, residual_radius=max(radius, rmse))
    return SurrogateFit(
        SurrogateFitStatus.FITTED,
        model,
        tuple(item.candidate_id for item in measured),
        retained,
        held_out_count,
        rmse,
        covered / held_out_count,
        math.fsum(feasibility_errors) / held_out_count,
        None,
    )


@dataclass(frozen=True, slots=True)
class SurrogatePolicyState:
    policy_id: str
    selected_candidate_ids: tuple[str, ...] = ()
    decision_index: int = 0
    fit_id: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.startswith("surrogate_policy_"):
            raise SurrogateError("surrogate state requires a canonical policy ID")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise SurrogateError("surrogate selected candidate IDs must be unique")
        if self.decision_index < 0:
            raise SurrogateError("surrogate decision index cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "decision_index": self.decision_index,
            "fit_id": self.fit_id,
        }

    @classmethod
    def from_record(cls, record: object) -> SurrogatePolicyState:
        if not isinstance(record, dict):
            raise SurrogateError("surrogate state must be an object")
        required = {
            "schema_version",
            "policy_id",
            "selected_candidate_ids",
            "decision_index",
            "fit_id",
        }
        if set(record) != required or record["schema_version"] != SURROGATE_SCHEMA_VERSION:
            raise SurrogateError("unsupported or malformed surrogate state")
        ids = record["selected_candidate_ids"]
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise SurrogateError("surrogate selected IDs must be a string list")
        if not isinstance(record["policy_id"], str) or not isinstance(
            record["decision_index"], int
        ):
            raise SurrogateError("surrogate state types are invalid")
        fit_id = record["fit_id"]
        if fit_id is not None and not isinstance(fit_id, str):
            raise SurrogateError("surrogate fit ID must be text or null")
        return cls(record["policy_id"], tuple(ids), record["decision_index"], fit_id)


@dataclass(frozen=True, slots=True)
class SurrogatePolicyDecision:
    candidate_id: str
    selected: bool
    reason: str
    acquisition: float | None
    prediction: SurrogatePrediction | None
    status: ArchitectureEvidenceStatus
    constraint_record: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "selected": self.selected,
            "reason": self.reason,
            "acquisition": self.acquisition,
            "prediction": None if self.prediction is None else self.prediction.to_record(),
            "status": self.status.value,
            "constraints": self.constraint_record,
        }


@dataclass(frozen=True, slots=True)
class SurrogateSelection:
    decisions: tuple[SurrogatePolicyDecision, ...]
    next_state: SurrogatePolicyState
    fit: SurrogateFit
    budget_exhausted: bool

    @property
    def selected(self) -> tuple[SurrogatePolicyDecision, ...]:
        return tuple(item for item in self.decisions if item.selected)

    def to_record(self) -> dict[str, object]:
        return {
            "decisions": [item.to_record() for item in self.decisions],
            "next_state": self.next_state.to_record(),
            "fit": self.fit.to_record(),
            "budget_exhausted": self.budget_exhausted,
        }


def _distance(left: CompleteArchitectureState, right: CompleteArchitectureState) -> int:
    left_values = {axis.value: value for axis, value in left.candidate.assignments}
    right_values = {axis.value: value for axis, value in right.candidate.assignments}
    axes = set(left_values) | set(right_values)
    return sum(left_values.get(axis) != right_values.get(axis) for axis in axes)


class SurrogateArchitecturePolicy:
    """Calibrated expected-improvement or hypervolume-improvement acquisition."""

    def __init__(
        self,
        config: SurrogateConfig,
        objectives: ObjectiveSet,
        constraints: ConstraintSet,
    ) -> None:
        self.config = config
        self.objectives = objectives
        self.constraints = constraints

    @property
    def policy_id(self) -> str:
        identity = canonical_identity_json(
            {
                "config": self.config.to_record(),
                "objectives": self.objectives.objective_set_id,
                "constraints": self.constraints.constraint_set_id,
            }
        )
        return f"surrogate_policy_{hashlib.sha256(identity.encode()).hexdigest()}"

    def _utility(self, values: tuple[ParetoObjectiveValue, ...]) -> float:
        by_metric = {item.metric: item.estimate for item in values}
        total = 0.0
        weight = 0.0
        for term in self.objectives.terms:
            value = by_metric.get(term.metric)
            if value is None:
                return -math.inf
            direction = 1.0 if term.direction is ObjectiveDirection.MAXIMIZE else -1.0
            total += direction * term.weight * value
            weight += term.weight
        return total / weight

    def _acquisition(
        self,
        prediction: SurrogatePrediction,
        frontier: tuple[ParetoCandidate, ...],
    ) -> float:
        candidate = ParetoCandidate(prediction.candidate_id, prediction.objectives, {})
        if any(conservatively_dominates(other, candidate, self.objectives) for other in frontier):
            return 0.0
        if self.config.acquisition is AcquisitionKind.EXPECTED_IMPROVEMENT:
            best = max((self._utility(item.objectives) for item in frontier), default=0.0)
            return max(0.0, self._utility(prediction.objectives) - best) + (
                self.config.exploration_weight * prediction.uncertainty
            )
        improvement = self._utility(prediction.objectives)
        if frontier:
            reference = min(self._utility(item.objectives) for item in frontier)
            improvement = max(0.0, improvement - reference)
        return improvement + self.config.exploration_weight * prediction.uncertainty

    def select(
        self,
        candidates: tuple[CompleteArchitectureState, ...],
        fit: SurrogateFit,
        state: SurrogatePolicyState | None = None,
    ) -> SurrogateSelection:
        current = state or SurrogatePolicyState(self.policy_id)
        if current.policy_id != self.policy_id:
            raise SurrogateError("surrogate state belongs to another policy")
        if len(current.selected_candidate_ids) >= self.config.evaluation_budget:
            return SurrogateSelection((), current, fit, True)
        pool = {candidate.candidate_id: candidate for candidate in candidates}
        if len(pool) != len(candidates) or not pool:
            raise SurrogateError("surrogate candidate pool IDs must be unique and non-empty")
        if (
            current.fit_id is not None
            and fit.model is not None
            and current.fit_id != fit.model.model_id
        ):
            raise SurrogateError("surrogate resume state does not match fitted model")
        frontier = tuple(
            ParetoCandidate(
                example.candidate_id,
                example.objectives,
                {"candidate_id": example.candidate_id},
            )
            for example in (() if fit.model is None else fit.model.examples)
        )
        decisions: list[SurrogatePolicyDecision] = []
        scored: list[tuple[float, int, CompleteArchitectureState, SurrogatePrediction]] = []
        selected_ids = set(current.selected_candidate_ids)
        for candidate in candidates:
            constraint_record = self.constraints.evaluate(candidate.constraints).to_record()
            if candidate.candidate_id in selected_ids:
                decisions.append(
                    SurrogatePolicyDecision(
                        candidate.candidate_id,
                        False,
                        "already_selected",
                        None,
                        None,
                        candidate.status,
                        constraint_record,
                    )
                )
                continue
            if candidate.status is not ArchitectureEvidenceStatus.PREDICTED:
                decisions.append(
                    SurrogatePolicyDecision(
                        candidate.candidate_id,
                        False,
                        f"retained_{candidate.status.value}_evidence",
                        None,
                        None,
                        candidate.status,
                        constraint_record,
                    )
                )
                continue
            if not self.constraints.evaluate(candidate.constraints).passed:
                decisions.append(
                    SurrogatePolicyDecision(
                        candidate.candidate_id,
                        False,
                        "predicted_constraint_violation",
                        None,
                        None,
                        candidate.status,
                        constraint_record,
                    )
                )
                continue
            if fit.model is None:
                prediction = SurrogatePrediction(
                    candidate.candidate_id,
                    candidate.objectives,
                    1.0,
                    0.0,
                    0.0,
                )
                acquisition = self._acquisition(prediction, frontier)
                reason = f"fallback_{fit.status.value}"
            else:
                axes = tuple(
                    item.removeprefix("axis:")
                    for item in fit.model.feature_schema
                    if item.startswith("axis:")
                )
                prediction = fit.model.predict(
                    candidate.candidate_id,
                    _feature_values(candidate, axes),
                )
                acquisition = self._acquisition(prediction, frontier)
            if (
                fit.model is not None
                and prediction.feasible_probability < self.config.minimum_feasible_probability
            ):
                decisions.append(
                    SurrogatePolicyDecision(
                        candidate.candidate_id,
                        False,
                        "surrogate_feasibility_below_threshold",
                        acquisition,
                        prediction,
                        candidate.status,
                        constraint_record,
                    )
                )
                continue
            scored.append(
                (
                    acquisition,
                    _tie_rank(self.config.seed, current.decision_index, candidate.candidate_id),
                    candidate,
                    prediction,
                )
            )
        remaining = self.config.evaluation_budget - len(selected_ids)
        limit = min(self.config.batch_size, remaining)
        chosen: list[tuple[float, int, CompleteArchitectureState, SurrogatePrediction]] = []
        pool_scored = list(scored)
        while pool_scored and len(chosen) < limit:
            pool_scored.sort(
                key=lambda item: (
                    -item[0],
                    -min((_distance(item[2], previous[2]) for previous in chosen), default=0),
                    item[1],
                    item[2].candidate_id,
                )
            )
            chosen.append(pool_scored.pop(0))
        chosen_ids = {item[2].candidate_id for item in chosen}
        reason = (
            "surrogate_acquisition" if fit.model is not None else f"fallback_{fit.status.value}"
        )
        for acquisition, _, candidate, prediction in scored:
            constraint_record = self.constraints.evaluate(candidate.constraints).to_record()
            decisions.append(
                SurrogatePolicyDecision(
                    candidate.candidate_id,
                    candidate.candidate_id in chosen_ids,
                    reason if candidate.candidate_id in chosen_ids else "outside_batch_cutoff",
                    acquisition,
                    prediction,
                    candidate.status,
                    constraint_record,
                )
            )
        decisions.sort(key=lambda item: item.candidate_id)
        next_state = SurrogatePolicyState(
            self.policy_id,
            (*current.selected_candidate_ids, *[item[2].candidate_id for item in chosen]),
            current.decision_index + 1,
            None if fit.model is None else fit.model.model_id,
        )
        return SurrogateSelection(
            tuple(decisions),
            next_state,
            fit,
            len(next_state.selected_candidate_ids) >= self.config.evaluation_budget or not scored,
        )


ArchitectureSurrogatePolicy = SurrogateArchitecturePolicy
BayesianArchitecturePolicy = SurrogateArchitecturePolicy
