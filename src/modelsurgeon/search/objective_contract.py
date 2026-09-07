"""Versioned hard-constraint and soft-objective contract for autonomous search."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.config import OptimizeMetric, Settings
from modelsurgeon.plugins import PluginCapabilityCard, PluginKind

OBJECTIVE_CONTRACT_SCHEMA_VERSION = 1


class ObjectiveContractError(ValueError):
    """Raised when an objective contract is ambiguous or unsafe."""


class ContractMetric(StrEnum):
    QUALITY = "quality"
    PERPLEXITY = "perplexity"
    LATENCY_GAIN = "latency_gain"
    LATENCY = "latency"
    PARAMETER_COUNT = "parameter_count"
    PEAK_RAM = "peak_ram"
    PEAK_VRAM = "peak_vram"
    FILE_SIZE = "file_size"
    OPTIMIZATION_TIME = "optimization_time"
    REPAIR_COST = "repair_cost"
    ENERGY = "energy"


class MetricUnit(StrEnum):
    RATIO = "ratio"
    PERPLEXITY_POINTS = "perplexity_points"
    COUNT = "count"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"
    SECONDS = "seconds"
    CURRENCY = "currency"
    JOULES = "joules"


class ConstraintDirection(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveMode(StrEnum):
    WEIGHTED = "weighted"
    LEXICOGRAPHIC = "lexicographic"
    PARETO = "pareto"


class ObjectiveNormalization(StrEnum):
    IDENTITY = "identity"
    BASELINE_RATIO = "baseline_ratio"
    MIN_MAX = "min_max"


class ContractOutcome(StrEnum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


_UNITS: dict[str, MetricUnit] = {
    ContractMetric.QUALITY: MetricUnit.RATIO,
    ContractMetric.PERPLEXITY: MetricUnit.PERPLEXITY_POINTS,
    ContractMetric.LATENCY_GAIN: MetricUnit.RATIO,
    ContractMetric.LATENCY: MetricUnit.MILLISECONDS,
    ContractMetric.PARAMETER_COUNT: MetricUnit.COUNT,
    ContractMetric.PEAK_RAM: MetricUnit.BYTES,
    ContractMetric.PEAK_VRAM: MetricUnit.BYTES,
    ContractMetric.FILE_SIZE: MetricUnit.BYTES,
    ContractMetric.OPTIMIZATION_TIME: MetricUnit.SECONDS,
    ContractMetric.REPAIR_COST: MetricUnit.CURRENCY,
    ContractMetric.ENERGY: MetricUnit.JOULES,
}


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObjectiveContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ObjectiveContractError(f"{label} must be finite and valid")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ObjectivePluginBinding:
    name: str
    plugin_id: str
    capability: str
    config_digest: str
    trust_mode: str = "subprocess"

    def __post_init__(self) -> None:
        fields = (self.name, self.plugin_id, self.capability, self.config_digest)
        if not all(value.strip() for value in fields):
            raise ObjectiveContractError("objective plugin binding fields are required")
        if self.trust_mode not in {"subprocess", "trusted_in_process"}:
            raise ObjectiveContractError("objective plugin trust mode is unsupported")

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "plugin_id": self.plugin_id,
            "capability": self.capability,
            "config_digest": self.config_digest,
            "trust_mode": self.trust_mode,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveApprovalPolicy:
    require_execution_approval: bool = True
    require_custom_plugin_approval: bool = True
    allowed_plugin_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.allowed_plugin_names != tuple(sorted(set(self.allowed_plugin_names))):
            raise ObjectiveContractError("allowed plugin names must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "require_execution_approval": self.require_execution_approval,
            "require_custom_plugin_approval": self.require_custom_plugin_approval,
            "allowed_plugin_names": list(self.allowed_plugin_names),
        }


@dataclass(frozen=True, slots=True)
class HardConstraint:
    metric: str
    direction: ConstraintDirection
    threshold: float
    unit: MetricUnit
    baseline: str = "absolute"

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ObjectiveContractError("hard constraints require a metric")
        _finite(self.threshold, "constraint threshold", nonnegative=True)
        expected = _UNITS.get(self.metric)
        if expected is not None and expected is not self.unit:
            raise ObjectiveContractError(f"{self.metric} requires unit {expected.value}")
        if not self.baseline.strip():
            raise ObjectiveContractError("constraint baseline is required")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "threshold": self.threshold,
            "unit": self.unit.value,
            "baseline": self.baseline,
        }


@dataclass(frozen=True, slots=True)
class SoftObjective:
    metric: str
    direction: ObjectiveDirection
    unit: MetricUnit
    weight: float = 1.0
    normalization: ObjectiveNormalization = ObjectiveNormalization.BASELINE_RATIO
    baseline: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    plugin: ObjectivePluginBinding | None = None

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ObjectiveContractError("soft objectives require a metric")
        expected = _UNITS.get(self.metric)
        if expected is not None and expected is not self.unit:
            raise ObjectiveContractError(f"{self.metric} requires unit {expected.value}")
        _finite(self.weight, "objective weight", nonnegative=True)
        if self.weight <= 0:
            raise ObjectiveContractError("objective weight must be positive")
        if self.normalization is ObjectiveNormalization.BASELINE_RATIO:
            if self.baseline is not None and self.baseline == 0:
                raise ObjectiveContractError("baseline ratio requires a non-zero baseline")
        elif self.normalization is ObjectiveNormalization.MIN_MAX:
            if self.minimum is None or self.maximum is None or self.minimum >= self.maximum:
                raise ObjectiveContractError("min-max objective requires ordered bounds")
        elif self.baseline is not None or self.minimum is not None or self.maximum is not None:
            raise ObjectiveContractError("normalization bounds do not match normalization mode")
        if self.metric not in _UNITS and self.plugin is None:
            raise ObjectiveContractError("custom objectives require a plugin binding")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "unit": self.unit.value,
            "weight": self.weight,
            "normalization": self.normalization.value,
            "baseline": self.baseline,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "plugin": None if self.plugin is None else self.plugin.to_record(),
        }


@dataclass(frozen=True, slots=True)
class MetricObservation:
    metric: str
    value: float
    unit: MetricUnit
    lower: float | None = None
    upper: float | None = None
    baseline: float | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite(self.value, "metric observation")
        expected = _UNITS.get(self.metric)
        if expected is not None and expected is not self.unit:
            raise ObjectiveContractError(
                f"{self.metric} observation requires unit {expected.value}"
            )
        if self.lower is not None and self.lower > self.value:
            raise ObjectiveContractError("observation lower bound cannot exceed value")
        if self.upper is not None and self.upper < self.value:
            raise ObjectiveContractError("observation upper bound cannot be below value")
        for bound in (self.lower, self.upper, self.baseline):
            if bound is not None:
                _finite(bound, "observation bound")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ObjectiveContractError("observation evidence must be sorted and unique")

    def conservative(self, direction: ConstraintDirection | ObjectiveDirection) -> float:
        if direction in {ConstraintDirection.MINIMUM, ObjectiveDirection.MAXIMIZE}:
            return self.value if self.lower is None else self.lower
        return self.value if self.upper is None else self.upper

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit.value,
            "lower": self.lower,
            "upper": self.upper,
            "baseline": self.baseline,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    constraint: HardConstraint
    observed: float | None
    passed: bool
    reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "observed": self.observed,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ContractEvaluation:
    outcome: ContractOutcome
    constraint_results: tuple[ConstraintResult, ...]
    objective_values: tuple[float, ...] | None
    score: float | None
    reason: str

    @property
    def feasible(self) -> bool:
        return self.outcome is ContractOutcome.FEASIBLE

    @property
    def comparable(self) -> bool:
        return self.objective_values is not None

    def to_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "constraint_results": [item.to_record() for item in self.constraint_results],
            "objective_values": (
                None if self.objective_values is None else list(self.objective_values)
            ),
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveContract:
    constraints: tuple[HardConstraint, ...]
    objectives: tuple[SoftObjective, ...]
    mode: ObjectiveMode = ObjectiveMode.WEIGHTED
    approval_policy: ObjectiveApprovalPolicy = ObjectiveApprovalPolicy()
    schema_version: int = OBJECTIVE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBJECTIVE_CONTRACT_SCHEMA_VERSION:
            raise ObjectiveContractError("unsupported objective contract schema")
        if not self.constraints or not self.objectives:
            raise ObjectiveContractError("objective contracts require hard and soft terms")
        constraint_metrics = [item.metric for item in self.constraints]
        objective_metrics = [item.metric for item in self.objectives]
        if len(constraint_metrics) != len(set(constraint_metrics)):
            raise ObjectiveContractError("hard constraint metrics must be unique")
        if len(objective_metrics) != len(set(objective_metrics)):
            raise ObjectiveContractError("soft objective metrics must be unique")
        object.__setattr__(
            self, "constraints", tuple(sorted(self.constraints, key=lambda item: item.metric))
        )
        object.__setattr__(
            self, "objectives", tuple(sorted(self.objectives, key=lambda item: item.metric))
        )

    @property
    def contract_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"objective_contract_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "constraints": [item.to_record() for item in self.constraints],
            "objectives": [item.to_record() for item in self.objectives],
            "approval_policy": self.approval_policy.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "mode": self.mode.value,
            "constraints": [item.to_record() for item in self.constraints],
            "objectives": [item.to_record() for item in self.objectives],
            "approval_policy": self.approval_policy.to_record(),
        }


def validate_plugin_objective(objective: SoftObjective, card: PluginCapabilityCard) -> None:
    """Require custom objective capability/trust declarations before execution."""

    if objective.plugin is None:
        return
    if (
        card.kind is not PluginKind.OBJECTIVE
        or card.name != objective.plugin.name
        or card.plugin_id != objective.plugin.plugin_id
    ):
        raise ObjectiveContractError("objective plugin card does not match binding")
    if objective.plugin.capability not in card.capabilities:
        raise ObjectiveContractError("objective plugin capability is not declared")
    if objective.plugin.trust_mode not in {mode.value for mode in card.trust_modes}:
        raise ObjectiveContractError("objective plugin trust mode is not declared")


def evaluate_contract(
    contract: ObjectiveContract,
    observations: Sequence[MetricObservation],
    *,
    plugin_cards: Mapping[str, PluginCapabilityCard] | None = None,
) -> ContractEvaluation:
    """Evaluate hard constraints first; missing evidence can never be rewarded."""

    by_metric: dict[str, MetricObservation] = {}
    for observation in observations:
        if observation.metric in by_metric:
            raise ObjectiveContractError("metric observations must be unique")
        by_metric[observation.metric] = observation
    results: list[ConstraintResult] = []
    for constraint in contract.constraints:
        constraint_observation = by_metric.get(constraint.metric)
        if constraint_observation is None:
            results.append(ConstraintResult(constraint, None, False, "missing_observation"))
            continue
        observed = constraint_observation.conservative(constraint.direction)
        passed = (
            observed >= constraint.threshold
            if constraint.direction is ConstraintDirection.MINIMUM
            else observed <= constraint.threshold
        )
        results.append(
            ConstraintResult(
                constraint, observed, passed, None if passed else "threshold_violation"
            )
        )
    if not all(item.passed for item in results):
        return ContractEvaluation(
            ContractOutcome.INFEASIBLE,
            tuple(results),
            None,
            None,
            "hard constraints fail closed on missing or violating evidence",
        )
    for objective in contract.objectives:
        if objective.plugin is None:
            continue
        if (
            contract.approval_policy.allowed_plugin_names
            and objective.plugin.name not in contract.approval_policy.allowed_plugin_names
        ):
            return ContractEvaluation(
                ContractOutcome.UNSUPPORTED,
                tuple(results),
                None,
                None,
                "objective plugin is not approved by the contract policy",
            )
        card = None if plugin_cards is None else plugin_cards.get(objective.plugin.name)
        if card is None:
            return ContractEvaluation(
                ContractOutcome.UNSUPPORTED,
                tuple(results),
                None,
                None,
                "objective plugin capability card is unavailable",
            )
        try:
            validate_plugin_objective(objective, card)
        except ObjectiveContractError as error:
            return ContractEvaluation(
                ContractOutcome.UNSUPPORTED,
                tuple(results),
                None,
                None,
                str(error),
            )
    values: list[float] = []
    for objective in contract.objectives:
        objective_observation = by_metric.get(objective.metric)
        if objective_observation is None:
            return ContractEvaluation(
                ContractOutcome.UNKNOWN,
                tuple(results),
                None,
                None,
                f"objective evidence is incomplete: {objective.metric}",
            )
        selected = objective_observation.conservative(objective.direction)
        if objective.normalization is ObjectiveNormalization.IDENTITY:
            normalized = selected
        elif objective.normalization is ObjectiveNormalization.BASELINE_RATIO:
            baseline = objective.baseline or objective_observation.baseline
            if baseline is None or baseline == 0:
                return ContractEvaluation(
                    ContractOutcome.UNKNOWN,
                    tuple(results),
                    None,
                    None,
                    f"objective baseline is missing: {objective.metric}",
                )
            normalized = selected / baseline
        else:
            assert objective.minimum is not None and objective.maximum is not None
            normalized = (selected - objective.minimum) / (objective.maximum - objective.minimum)
        values.append(
            normalized if objective.direction is ObjectiveDirection.MAXIMIZE else -normalized
        )
    if contract.mode is ObjectiveMode.WEIGHTED:
        total_weight = math.fsum(item.weight for item in contract.objectives)
        score = (
            math.fsum(
                value * objective.weight
                for value, objective in zip(values, contract.objectives, strict=True)
            )
            / total_weight
        )
    elif contract.mode is ObjectiveMode.LEXICOGRAPHIC:
        score = values[0]
    else:
        score = None
    return ContractEvaluation(
        ContractOutcome.FEASIBLE,
        tuple(results),
        tuple(values),
        score,
        "all hard and soft evidence is comparable",
    )


def contracts_dominate(left: ContractEvaluation, right: ContractEvaluation) -> bool:
    """Return conservative Pareto dominance; infeasible/incomparable points never dominate."""

    if not left.feasible or not right.feasible or not left.comparable or not right.comparable:
        return False
    assert left.objective_values is not None and right.objective_values is not None
    return all(
        a >= b for a, b in zip(left.objective_values, right.objective_values, strict=True)
    ) and any(a > b for a, b in zip(left.objective_values, right.objective_values, strict=True))


def _legacy_metric(metric: OptimizeMetric) -> tuple[ContractMetric, MetricUnit]:
    mapping = {
        OptimizeMetric.QUALITY: (ContractMetric.QUALITY, MetricUnit.RATIO),
        OptimizeMetric.PERPLEXITY: (
            ContractMetric.PERPLEXITY,
            MetricUnit.PERPLEXITY_POINTS,
        ),
        OptimizeMetric.PARAMETER_COUNT: (ContractMetric.PARAMETER_COUNT, MetricUnit.COUNT),
        OptimizeMetric.LATENCY: (ContractMetric.LATENCY, MetricUnit.MILLISECONDS),
        OptimizeMetric.MEMORY: (ContractMetric.PEAK_RAM, MetricUnit.BYTES),
        OptimizeMetric.DISK_SIZE: (ContractMetric.FILE_SIZE, MetricUnit.BYTES),
    }
    return mapping[metric]


def contract_from_settings(settings: Settings) -> ObjectiveContract:
    """Translate legacy Settings into the versioned contract without losing meaning."""

    constraints = [
        HardConstraint(
            ContractMetric.QUALITY,
            ConstraintDirection.MINIMUM,
            settings.constraints.min_quality_retention_ratio,
            MetricUnit.RATIO,
            "immutable_source",
        )
    ]
    if settings.constraints.max_ram_bytes is not None:
        constraints.append(
            HardConstraint(
                ContractMetric.PEAK_RAM,
                ConstraintDirection.MAXIMUM,
                float(settings.constraints.max_ram_bytes),
                MetricUnit.BYTES,
            )
        )
    if settings.constraints.max_vram_bytes is not None:
        constraints.append(
            HardConstraint(
                ContractMetric.PEAK_VRAM,
                ConstraintDirection.MAXIMUM,
                float(settings.constraints.max_vram_bytes),
                MetricUnit.BYTES,
            )
        )
    if settings.constraints.max_perplexity_delta is not None:
        constraints.append(
            HardConstraint(
                ContractMetric.PERPLEXITY,
                ConstraintDirection.MAXIMUM,
                settings.constraints.max_perplexity_delta,
                MetricUnit.PERPLEXITY_POINTS,
                "immutable_source",
            )
        )
    if settings.constraints.min_latency_gain_ratio is not None:
        constraints.append(
            HardConstraint(
                ContractMetric.LATENCY_GAIN,
                ConstraintDirection.MINIMUM,
                settings.constraints.min_latency_gain_ratio,
                MetricUnit.RATIO,
                "immutable_source",
            )
        )
    if settings.constraints.max_disk_bytes is not None:
        constraints.append(
            HardConstraint(
                ContractMetric.FILE_SIZE,
                ConstraintDirection.MAXIMUM,
                float(settings.constraints.max_disk_bytes),
                MetricUnit.BYTES,
            )
        )
    if settings.objective.terms is None:
        terms = tuple(
            (
                metric,
                ObjectiveDirection.MAXIMIZE
                if metric is OptimizeMetric.QUALITY
                else ObjectiveDirection.MINIMIZE,
            )
            for metric in settings.objective.optimize
        )
        objectives = tuple(
            SoftObjective(_legacy_metric(metric)[0], direction, _legacy_metric(metric)[1])
            for metric, direction in terms
        )
    else:
        objectives = tuple(
            SoftObjective(
                _legacy_metric(config.metric)[0],
                ObjectiveDirection(config.direction.value),
                _legacy_metric(config.metric)[1],
                config.weight,
                ObjectiveNormalization(config.normalization.value),
                minimum=config.minimum,
                maximum=config.maximum,
            )
            for config in settings.objective.terms
        )
    return ObjectiveContract(tuple(constraints), objectives)


__all__ = [
    "OBJECTIVE_CONTRACT_SCHEMA_VERSION",
    "ConstraintDirection",
    "ContractEvaluation",
    "ContractMetric",
    "ContractOutcome",
    "HardConstraint",
    "MetricObservation",
    "MetricUnit",
    "ObjectiveApprovalPolicy",
    "ObjectiveContract",
    "ObjectiveContractError",
    "ObjectiveDirection",
    "ObjectiveMode",
    "ObjectiveNormalization",
    "ObjectivePluginBinding",
    "SoftObjective",
    "contract_from_settings",
    "contracts_dominate",
    "evaluate_contract",
    "validate_plugin_objective",
]
