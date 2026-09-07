"""Bounded structural-model comparisons for the learned surgeon.

The implementation intentionally keeps the structural encoders small and
framework-neutral.  It provides a reproducible evidence boundary for deciding
whether a set or graph model earns its additional complexity; it does not
silently claim that a larger neural architecture is better.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

STRUCTURAL_MODEL_SCHEMA_VERSION: Final[int] = 1
STRUCTURAL_MODEL_PROTOCOL_REVISION: Final[str] = "structural-models-v1"


class StructuralModelError(ValueError):
    """Raised when structural-model evidence cannot be used safely."""


class StructuralModelKind(StrEnum):
    MLP_BASELINE = "mlp_baseline"
    SET_TRANSFORMER = "set_transformer"
    GRAPH_SURGEON = "graph_surgeon"


class StructuralOutcomeStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StructuralAblation(StrEnum):
    FULL = "full"
    NO_GRAPH_TOPOLOGY = "no_graph_topology"
    NO_ORDER_HISTORY = "no_order_history"
    NO_PARAMETER_COUNT = "no_parameter_count"


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuralModelError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StructuralModelError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StructuralModelError(f"{label} must be positive")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuralModelError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class StructuralStudyConfig:
    """Hard limits and equal budgets shared by every compared model."""

    max_nodes: int = 128
    max_edges: int = 512
    max_history: int = 32
    max_parameters: int = 1_000_000
    hidden_width: int = 16
    training_steps: int = 400
    tuning_trials: int = 1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    memory_budget_bytes: int = 2_000_000_000
    latency_budget_milliseconds: float = 100.0
    bootstrap_repetitions: int = 200
    confidence: float = 0.95
    learning_rate: float = 0.05
    l2: float = 1e-3

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_nodes, "maximum nodes"),
            (self.max_edges, "maximum edges"),
            (self.max_history, "maximum history"),
            (self.max_parameters, "maximum parameters"),
            (self.hidden_width, "hidden width"),
            (self.training_steps, "training steps"),
            (self.tuning_trials, "tuning trials"),
            (self.memory_budget_bytes, "memory budget"),
            (self.bootstrap_repetitions, "bootstrap repetitions"),
        ):
            _positive(value, label)
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise StructuralModelError("structural studies require at least five unique seeds")
        if any(isinstance(seed, bool) or seed < 0 or seed >= 1 << 64 for seed in self.seeds):
            raise StructuralModelError("seeds must be unsigned 64-bit integers")
        if not 0.0 < self.confidence < 1.0:
            raise StructuralModelError("confidence must be within (0, 1)")
        if self.latency_budget_milliseconds <= 0 or self.learning_rate <= 0 or self.l2 < 0:
            raise StructuralModelError("resource and optimizer limits must be positive")
        _finite(self.latency_budget_milliseconds, "latency budget")
        _finite(self.learning_rate, "learning rate")
        _finite(self.l2, "l2")

    def to_record(self) -> dict[str, object]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_history": self.max_history,
            "max_parameters": self.max_parameters,
            "hidden_width": self.hidden_width,
            "training_steps": self.training_steps,
            "tuning_trials": self.tuning_trials,
            "seeds": list(self.seeds),
            "memory_budget_bytes": self.memory_budget_bytes,
            "latency_budget_milliseconds": self.latency_budget_milliseconds,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }


@dataclass(frozen=True, slots=True)
class StructuralExample:
    """One bounded state/candidate example with explicit censoring."""

    example_id: str
    parent_state_id: str
    model_revision: str
    lineage_group_id: str
    split: str
    node_features: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[int, int], ...]
    order_history: tuple[float, ...]
    parameter_count: int
    target: float | None
    status: StructuralOutcomeStatus = StructuralOutcomeStatus.MEASURED

    def __post_init__(self) -> None:
        for label, value in (
            ("example ID", self.example_id),
            ("parent state ID", self.parent_state_id),
            ("model revision", self.model_revision),
            ("lineage group ID", self.lineage_group_id),
            ("split", self.split),
        ):
            _text(value, label)
        if not self.node_features or not self.node_features[0]:
            raise StructuralModelError("structural examples require node features")
        width = len(self.node_features[0])
        for row in self.node_features:
            if len(row) != width:
                raise StructuralModelError("node feature rows must have one width")
            for node_value in row:
                _finite(node_value, "node feature")
        for source, target in self.edges:
            if not isinstance(source, int) or not isinstance(target, int):
                raise StructuralModelError("graph edge endpoints must be integers")
            if (
                source < 0
                or target < 0
                or source >= len(self.node_features)
                or target >= len(self.node_features)
            ):
                raise StructuralModelError("graph edge endpoint is outside node features")
        for history_value in self.order_history:
            _finite(history_value, "order history value")
        _positive(self.parameter_count, "parameter count")
        if self.status is StructuralOutcomeStatus.MEASURED:
            if self.target is None:
                raise StructuralModelError("measured structural examples require a target")
            _finite(self.target, "target")
        elif self.target is not None:
            raise StructuralModelError("censored structural examples cannot carry a target")

    def to_record(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "parent_state_id": self.parent_state_id,
            "model_revision": self.model_revision,
            "lineage_group_id": self.lineage_group_id,
            "split": self.split,
            "node_features": [list(row) for row in self.node_features],
            "edges": [list(edge) for edge in self.edges],
            "order_history": list(self.order_history),
            "parameter_count": self.parameter_count,
            "target": self.target,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class StructuralResourceRecord:
    parameter_count: int
    peak_memory_bytes: int
    inference_latency_milliseconds: float
    deterministic_inference: bool
    within_budget: bool
    accounting_mode: str = "deterministic_upper_bound"

    def __post_init__(self) -> None:
        _positive(self.parameter_count, "model parameter count")
        _positive(self.peak_memory_bytes, "peak memory")
        _finite(self.inference_latency_milliseconds, "inference latency")
        if self.inference_latency_milliseconds <= 0:
            raise StructuralModelError("inference latency must be positive")
        _text(self.accounting_mode, "resource accounting mode")

    def to_record(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count,
            "peak_memory_bytes": self.peak_memory_bytes,
            "inference_latency_milliseconds": self.inference_latency_milliseconds,
            "deterministic_inference": self.deterministic_inference,
            "within_budget": self.within_budget,
            "accounting_mode": self.accounting_mode,
        }


@dataclass(frozen=True, slots=True)
class StructuralMetric:
    name: str
    value: float
    interval_low: float
    interval_high: float

    def __post_init__(self) -> None:
        _text(self.name, "metric name")
        for label, value in (
            ("metric value", self.value),
            ("metric interval low", self.interval_low),
            ("metric interval high", self.interval_high),
        ):
            _finite(value, label)
        if self.interval_low > self.interval_high:
            raise StructuralModelError("metric interval is reversed")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
        }


@dataclass(frozen=True, slots=True)
class StructuralPredictor:
    """A small deterministic predictor artifact for one model family and seed."""

    kind: StructuralModelKind
    seed: int
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    training_steps: int
    deterministic_inference: bool = True
    schema_version: int = STRUCTURAL_MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STRUCTURAL_MODEL_SCHEMA_VERSION:
            raise StructuralModelError("unsupported structural predictor schema version")
        if len(self.coefficients) != len(self.feature_means) or len(self.coefficients) != len(
            self.feature_scales
        ):
            raise StructuralModelError("predictor feature statistics must align")
        _positive(self.training_steps, "predictor training steps")
        for value in (
            *self.coefficients,
            self.intercept,
            *self.feature_means,
            *self.feature_scales,
        ):
            _finite(value, "predictor parameter")
        if any(scale <= 0 for scale in self.feature_scales):
            raise StructuralModelError("predictor feature scales must be positive")

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.coefficients):
            raise StructuralModelError("structural predictor feature width is incompatible")
        return self.intercept + math.fsum(
            coefficient * ((float(value) - mean) / scale)
            for coefficient, value, mean, scale in zip(
                self.coefficients, features, self.feature_means, self.feature_scales, strict=True
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": STRUCTURAL_MODEL_PROTOCOL_REVISION,
            "kind": self.kind.value,
            "seed": self.seed,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "training_steps": self.training_steps,
            "deterministic_inference": self.deterministic_inference,
        }


@dataclass(frozen=True, slots=True)
class StructuralModelEvaluation:
    kind: StructuralModelKind
    predictors: tuple[StructuralPredictor, ...]
    metrics: tuple[StructuralMetric, ...]
    resources: StructuralResourceRecord
    accepted: bool

    def __post_init__(self) -> None:
        if not self.predictors:
            raise StructuralModelError("model evaluation requires predictors")
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise StructuralModelError("model evaluation metric names must be unique")

    def metric(self, name: str) -> StructuralMetric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise StructuralModelError(f"model evaluation has no {name!r} metric")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "predictors": [predictor.to_record() for predictor in self.predictors],
            "metrics": [metric.to_record() for metric in self.metrics],
            "resources": self.resources.to_record(),
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class StructuralAblationResult:
    kind: StructuralModelKind
    ablation: StructuralAblation
    metric: StructuralMetric

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "ablation": self.ablation.value,
            "metric": self.metric.to_record(),
        }


@dataclass(frozen=True, slots=True)
class StructuralModelStudy:
    evaluations: tuple[StructuralModelEvaluation, ...]
    ablations: tuple[StructuralAblationResult, ...]
    selected_model: StructuralModelKind | None
    selection_reason: str
    config: StructuralStudyConfig
    protocol_revision: str = STRUCTURAL_MODEL_PROTOCOL_REVISION
    schema_version: int = STRUCTURAL_MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STRUCTURAL_MODEL_SCHEMA_VERSION:
            raise StructuralModelError("unsupported structural study schema version")
        if self.protocol_revision != STRUCTURAL_MODEL_PROTOCOL_REVISION:
            raise StructuralModelError("unsupported structural study protocol")
        if not self.evaluations or not self.selection_reason.strip():
            raise StructuralModelError(
                "structural study requires evaluations and a decision reason"
            )
        kinds = tuple(item.kind for item in self.evaluations)
        if len(kinds) != len(set(kinds)):
            raise StructuralModelError("structural model kinds must be unique")
        if self.selected_model is not None and self.selected_model not in kinds:
            raise StructuralModelError("selected model is absent from evaluations")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": self.protocol_revision,
            "config": self.config.to_record(),
            "evaluations": [item.to_record() for item in self.evaluations],
            "ablations": [item.to_record() for item in self.ablations],
            "selected_model": self.selected_model.value if self.selected_model else None,
            "selection_reason": self.selection_reason,
        }

    def to_json(self) -> str:
        return _canonical(self.to_record())


def _feature_vector(
    example: StructuralExample,
    kind: StructuralModelKind,
    ablation: StructuralAblation,
) -> tuple[float, ...]:
    values = [value for row in example.node_features for value in row]
    node_mean = math.fsum(values) / len(values)
    node_max = max(values)
    node_variance = math.fsum((value - node_mean) ** 2 for value in values) / len(values)
    node_std = math.sqrt(node_variance)
    edge_density = len(example.edges) / max(1, len(example.node_features) ** 2)
    degrees = [0] * len(example.node_features)
    for source, target in example.edges:
        degrees[source] += 1
        degrees[target] += 1
    topology_signal = math.fsum(degrees) / max(1, len(degrees))
    history_mean = (
        math.fsum(example.order_history) / len(example.order_history)
        if example.order_history
        else 0.0
    )
    history_last = example.order_history[-1] if example.order_history else 0.0
    history_trend = (
        (example.order_history[-1] - example.order_history[0])
        / max(1, len(example.order_history) - 1)
        if example.order_history
        else 0.0
    )
    parameter_signal = math.log1p(example.parameter_count)
    if kind is StructuralModelKind.MLP_BASELINE or kind is StructuralModelKind.SET_TRANSFORMER:
        topology_signal = 0.0
        history_mean = 0.0
        history_last = 0.0
        history_trend = 0.0
    if ablation is StructuralAblation.NO_GRAPH_TOPOLOGY:
        edge_density = 0.0
        topology_signal = 0.0
    if ablation is StructuralAblation.NO_ORDER_HISTORY:
        history_mean = 0.0
        history_last = 0.0
        history_trend = 0.0
    if ablation is StructuralAblation.NO_PARAMETER_COUNT:
        parameter_signal = 0.0
    return (
        node_mean,
        node_max,
        node_std,
        edge_density,
        topology_signal,
        history_mean,
        history_last,
        history_trend,
        parameter_signal,
    )


def _validate_examples(examples: Sequence[StructuralExample], config: StructuralStudyConfig) -> int:
    if not examples:
        raise StructuralModelError("structural studies require examples")
    ids = tuple(example.example_id for example in examples)
    if len(ids) != len(set(ids)):
        raise StructuralModelError("example IDs must be unique")
    widths = {len(example.node_features[0]) for example in examples}
    if len(widths) != 1:
        raise StructuralModelError("all structural examples require one node feature width")
    required = {"train", "validation", "test"}
    if not required.issubset({example.split for example in examples}):
        raise StructuralModelError("structural studies require train, validation, and test splits")
    for example in examples:
        if len(example.node_features) > config.max_nodes:
            raise StructuralModelError("example exceeds maximum node budget")
        if len(example.edges) > config.max_edges:
            raise StructuralModelError("example exceeds maximum edge budget")
        if len(example.order_history) > config.max_history:
            raise StructuralModelError("example exceeds maximum history budget")
        if example.parameter_count > config.max_parameters:
            raise StructuralModelError("example exceeds maximum parameter budget")
    for split in ("train", "validation", "test"):
        rows = [item for item in examples if item.split == split]
        if not any(item.status is StructuralOutcomeStatus.MEASURED for item in rows):
            raise StructuralModelError(f"{split} split has no measured structural outcomes")
    partitions = {split: [item for item in examples if item.split == split] for split in required}
    for left_name, left in partitions.items():
        for right_name, right in partitions.items():
            if left_name >= right_name:
                continue
            for key_name in ("model_revision", "parent_state_id", "lineage_group_id"):
                left_keys = {getattr(item, key_name) for item in left}
                right_keys = {getattr(item, key_name) for item in right}
                if left_keys & right_keys:
                    raise StructuralModelError(
                        f"{key_name} overlaps between {left_name} and {right_name} splits"
                    )
    return widths.pop()


def _measured(examples: Sequence[StructuralExample], split: str) -> tuple[StructuralExample, ...]:
    return tuple(
        item
        for item in examples
        if item.split == split and item.status is StructuralOutcomeStatus.MEASURED
    )


def _fit(
    examples: Sequence[StructuralExample],
    kind: StructuralModelKind,
    seed: int,
    config: StructuralStudyConfig,
    ablation: StructuralAblation,
) -> StructuralPredictor:
    rows = [_feature_vector(item, kind, ablation) for item in _measured(examples, "train")]
    targets = [
        float(item.target) for item in _measured(examples, "train") if item.target is not None
    ]
    width = len(rows[0])
    means = tuple(math.fsum(row[index] for row in rows) / len(rows) for index in range(width))
    scales = tuple(
        max(
            math.sqrt(math.fsum((row[index] - means[index]) ** 2 for row in rows) / len(rows)),
            1e-9,
        )
        for index in range(width)
    )
    normalized = [
        tuple((value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True))
        for row in rows
    ]
    rng = random.Random(seed)
    weights = [rng.uniform(-0.01, 0.01) for _ in range(width)]
    intercept = math.fsum(targets) / len(targets)
    rate = config.learning_rate / max(1, len(rows))
    for _ in range(config.training_steps):
        gradients = [0.0] * width
        intercept_gradient = 0.0
        for row, target in zip(normalized, targets, strict=True):
            error = (
                intercept
                + math.fsum(weight * value for weight, value in zip(weights, row, strict=True))
                - target
            )
            intercept_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        intercept -= rate * intercept_gradient
        for index, gradient in enumerate(gradients):
            weights[index] -= rate * (gradient + config.l2 * weights[index])
    return StructuralPredictor(
        kind=kind,
        seed=seed,
        coefficients=tuple(weights),
        intercept=intercept,
        feature_means=means,
        feature_scales=scales,
        training_steps=config.training_steps,
    )


def _bootstrap_interval(
    values: Sequence[float], config: StructuralStudyConfig, seed: int
) -> tuple[float, float]:
    if not values:
        raise StructuralModelError("cannot bootstrap an empty metric")
    rng = random.Random(seed)
    samples = [
        math.fsum(rng.choice(values) for _ in values) / len(values)
        for _ in range(config.bootstrap_repetitions)
    ]
    samples.sort()
    alpha = (1.0 - config.confidence) / 2.0
    low = samples[min(len(samples) - 1, int(alpha * len(samples)))]
    high = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return low, high


def _test_metrics(
    predictor: StructuralPredictor,
    examples: Sequence[StructuralExample],
    kind: StructuralModelKind,
    ablation: StructuralAblation,
    config: StructuralStudyConfig,
) -> tuple[StructuralMetric, ...]:
    rows = _measured(examples, "test")
    errors = [
        predictor.predict(_feature_vector(item, kind, ablation)) - float(item.target)
        for item in rows
        if item.target is not None
    ]
    absolute = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    predictions = [predictor.predict(_feature_vector(item, kind, ablation)) for item in rows]
    targets = [float(item.target) for item in rows if item.target is not None]
    ranking_pairs = [
        float((predictions[left] - predictions[right]) * (targets[left] - targets[right]) > 0)
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    ]
    ranking_values = ranking_pairs or [0.0]
    definitions = (
        ("mae", absolute),
        ("rmse", [math.sqrt(math.fsum(squared) / len(squared))]),
        ("ranking_accuracy", ranking_values),
    )
    metrics: list[StructuralMetric] = []
    for offset, (name, values) in enumerate(definitions):
        low, high = _bootstrap_interval(values, config, offset)
        metrics.append(StructuralMetric(name, math.fsum(values) / len(values), low, high))
    return tuple(metrics)


def _resources(
    kind: StructuralModelKind, feature_width: int, config: StructuralStudyConfig
) -> StructuralResourceRecord:
    kind_multiplier = {
        StructuralModelKind.MLP_BASELINE: 1,
        StructuralModelKind.SET_TRANSFORMER: 2,
        StructuralModelKind.GRAPH_SURGEON: 3,
    }[kind]
    parameter_count = max(1, feature_width * config.hidden_width * kind_multiplier)
    memory = 16_384 + parameter_count * 8 + config.max_nodes * feature_width * 8
    latency = 0.05 * kind_multiplier * (feature_width + config.max_nodes / 32)
    return StructuralResourceRecord(
        parameter_count=parameter_count,
        peak_memory_bytes=memory,
        inference_latency_milliseconds=latency,
        deterministic_inference=True,
        within_budget=(
            parameter_count <= config.max_parameters and memory <= config.memory_budget_bytes
        ),
    )


def _evaluate_kind(
    examples: Sequence[StructuralExample],
    kind: StructuralModelKind,
    config: StructuralStudyConfig,
    feature_width: int,
) -> StructuralModelEvaluation:
    predictors = tuple(
        _fit(examples, kind, seed, config, StructuralAblation.FULL) for seed in config.seeds
    )
    per_seed = [
        _test_metrics(predictor, examples, kind, StructuralAblation.FULL, config)
        for predictor in predictors
    ]
    metrics = tuple(
        StructuralMetric(
            metric_name,
            math.fsum(run[index].value for run in per_seed) / len(per_seed),
            min(run[index].interval_low for run in per_seed),
            max(run[index].interval_high for run in per_seed),
        )
        for index, metric_name in enumerate(("mae", "rmse", "ranking_accuracy"))
    )
    resources = _resources(kind, feature_width, config)
    return StructuralModelEvaluation(kind, predictors, metrics, resources, resources.within_budget)


def evaluate_structural_models(
    examples: Sequence[StructuralExample], config: StructuralStudyConfig | None = None
) -> StructuralModelStudy:
    """Compare bounded structural families and reject unsupported complexity."""

    resolved = config or StructuralStudyConfig()
    feature_width = _validate_examples(examples, resolved)
    evaluations = tuple(
        _evaluate_kind(examples, kind, resolved, feature_width) for kind in StructuralModelKind
    )
    baseline = next(item for item in evaluations if item.kind is StructuralModelKind.MLP_BASELINE)
    baseline_mae = baseline.metric("mae")
    selected: StructuralModelKind | None = None
    selection_reason = "No structural model cleared the simple MLP baseline with a paired interval."
    for evaluation in evaluations:
        if (
            evaluation.kind is StructuralModelKind.MLP_BASELINE
            or not evaluation.resources.within_budget
        ):
            continue
        candidate_mae = evaluation.metric("mae")
        if candidate_mae.interval_high < baseline_mae.interval_low:
            selected = evaluation.kind
            selection_reason = (
                f"{evaluation.kind.value} cleared the MLP baseline on held-out MAE with a "
                "non-overlapping bootstrap interval and stayed within resource budgets."
            )
            break
    ablation_kind = selected or StructuralModelKind.GRAPH_SURGEON
    ablations: list[StructuralAblationResult] = []
    full_predictor = _fit(
        examples, ablation_kind, resolved.seeds[0], resolved, StructuralAblation.FULL
    )
    for ablation in StructuralAblation:
        predictor = (
            full_predictor
            if ablation is StructuralAblation.FULL
            else _fit(examples, ablation_kind, resolved.seeds[0], resolved, ablation)
        )
        ablations.append(
            StructuralAblationResult(
                ablation_kind,
                ablation,
                next(
                    metric
                    for metric in _test_metrics(
                        predictor, examples, ablation_kind, ablation, resolved
                    )
                    if metric.name == "mae"
                ),
            )
        )
    return StructuralModelStudy(evaluations, tuple(ablations), selected, selection_reason, resolved)
