"""Bounded, claim-safe comparison of architecture-search frontiers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

ARCHITECTURE_PARETO_STUDY_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArchitectureParetoStudyError(ValueError):
    """Raised when architecture study evidence or claims are incomplete."""


class ArchitectureStudyOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ArchitectureStudyClaim(StrEnum):
    POSITIVE = "positive"
    NEGATIVE_RESULT = "negative_result"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureParetoStudyError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise ArchitectureParetoStudyError(f"{label} must be a lowercase SHA-256")
    return result


def _interval(value: float, lower: float, upper: float, label: str) -> None:
    for item, name in ((value, label), (lower, f"{label} lower"), (upper, f"{label} upper")):
        if not math.isfinite(item):
            raise ArchitectureParetoStudyError(f"{name} must be finite")
    if lower > value or value > upper:
        raise ArchitectureParetoStudyError(f"{label} interval must contain its value")


@dataclass(frozen=True, slots=True)
class ArchitectureFrontierPoint:
    """A complete deployable state bound to measured artifact evidence."""

    state_id: str
    artifact_digest: str
    family: str
    hardware_profile: str
    quality: float
    quality_low: float
    quality_high: float
    deployment_cost: float
    deployment_cost_low: float
    deployment_cost_high: float
    repetitions: int
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.state_id.startswith("state_"):
            raise ArchitectureParetoStudyError("frontier points require deployable state IDs")
        _digest(self.artifact_digest, "frontier artifact digest")
        _text(self.family, "frontier family")
        _text(self.hardware_profile, "frontier hardware profile")
        _interval(self.quality, self.quality_low, self.quality_high, "quality")
        _interval(
            self.deployment_cost,
            self.deployment_cost_low,
            self.deployment_cost_high,
            "deployment cost",
        )
        if self.quality_low < 0 or self.deployment_cost_low < 0:
            raise ArchitectureParetoStudyError("frontier metrics cannot be negative")
        if self.repetitions <= 0:
            raise ArchitectureParetoStudyError("frontier repetitions must be positive")
        keys = tuple(key for key, _ in self.provenance)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ArchitectureParetoStudyError(
                "frontier provenance must be non-empty and canonical"
            )

    @property
    def point_id(self) -> str:
        payload = {
            "schema_version": ARCHITECTURE_PARETO_STUDY_SCHEMA_VERSION,
            "state_id": self.state_id,
            "artifact_digest": self.artifact_digest,
            "family": self.family,
            "hardware_profile": self.hardware_profile,
        }
        return "architecture_frontier_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "state_id": self.state_id,
            "artifact_digest": self.artifact_digest,
            "family": self.family,
            "hardware_profile": self.hardware_profile,
            "quality": self.quality,
            "quality_low": self.quality_low,
            "quality_high": self.quality_high,
            "deployment_cost": self.deployment_cost,
            "deployment_cost_low": self.deployment_cost_low,
            "deployment_cost_high": self.deployment_cost_high,
            "repetitions": self.repetitions,
            "provenance": [{"key": key, "value": value} for key, value in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class ArchitectureStudyMetric:
    name: str
    value: float
    lower: float
    upper: float
    repetitions: int
    source: str

    def __post_init__(self) -> None:
        _text(self.name, "study metric")
        _interval(self.value, self.lower, self.upper, self.name)
        if self.repetitions <= 0:
            raise ArchitectureParetoStudyError("study metric repetitions must be positive")
        _text(self.source, "study metric source")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "repetitions": self.repetitions,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureStudyKey:
    family: str
    hardware_profile: str
    method: str
    budget: int
    seed: int

    def __post_init__(self) -> None:
        for label, value in (
            ("family", self.family),
            ("hardware profile", self.hardware_profile),
            ("method", self.method),
        ):
            _text(value, label)
        if self.budget <= 0 or self.seed < 0:
            raise ArchitectureParetoStudyError("study budget must be positive and seed unsigned")

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family,
            "hardware_profile": self.hardware_profile,
            "method": self.method,
            "budget": self.budget,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureStudyCell:
    key: ArchitectureStudyKey
    outcome: ArchitectureStudyOutcome
    frontier_points: tuple[ArchitectureFrontierPoint, ...]
    metrics: tuple[ArchitectureStudyMetric, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        point_ids = tuple(item.point_id for item in self.frontier_points)
        if point_ids != tuple(sorted(set(point_ids))):
            raise ArchitectureParetoStudyError("cell frontier points must be unique and canonical")
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise ArchitectureParetoStudyError("cell metrics must be unique and canonical")
        if self.outcome in {
            ArchitectureStudyOutcome.UNSUPPORTED,
            ArchitectureStudyOutcome.FAILED,
            ArchitectureStudyOutcome.UNKNOWN,
        }:
            if not self.reason or self.frontier_points or self.metrics:
                raise ArchitectureParetoStudyError(
                    "non-measured cells require a reason and cannot publish partial evidence"
                )
        elif not self.reason and self.outcome is ArchitectureStudyOutcome.NEGATIVE_RESULT:
            raise ArchitectureParetoStudyError("negative study cells require a retained reason")
        elif self.outcome is ArchitectureStudyOutcome.MEASURED and (
            not self.frontier_points or not self.metrics
        ):
            raise ArchitectureParetoStudyError(
                "measured cells require frontier and metric evidence"
            )

    @property
    def cell_id(self) -> str:
        return (
            "architecture_study_"
            + hashlib.sha256(_canonical(self.key.to_record()).encode()).hexdigest()
        )

    def metric(self, name: str) -> ArchitectureStudyMetric | None:
        return next((item for item in self.metrics if item.name == name), None)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ARCHITECTURE_PARETO_STUDY_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "key": self.key.to_record(),
            "outcome": self.outcome.value,
            "frontier_points": [item.to_record() for item in self.frontier_points],
            "metrics": [item.to_record() for item in self.metrics],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureStudyComparison:
    family: str
    hardware_profile: str
    budget: int
    method: str
    baseline_method: str
    metric: str
    delta: float | None
    lower: float | None
    upper: float | None
    bootstrap_repetitions: int
    claim: ArchitectureStudyClaim
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("comparison family", self.family),
            ("comparison hardware profile", self.hardware_profile),
            ("comparison method", self.method),
            ("comparison baseline", self.baseline_method),
            ("comparison metric", self.metric),
            ("comparison reason", self.reason),
        ):
            _text(value, label)
        if self.bootstrap_repetitions < 100:
            raise ArchitectureParetoStudyError(
                "comparisons require at least 100 bootstrap repetitions"
            )
        if self.delta is None or self.lower is None or self.upper is None:
            if self.claim not in {
                ArchitectureStudyClaim.UNSUPPORTED,
                ArchitectureStudyClaim.INCONCLUSIVE,
            }:
                raise ArchitectureParetoStudyError("claimed comparisons require interval evidence")
        else:
            _interval(self.delta, self.lower, self.upper, "comparison delta")

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family,
            "hardware_profile": self.hardware_profile,
            "budget": self.budget,
            "method": self.method,
            "baseline_method": self.baseline_method,
            "metric": self.metric,
            "delta": self.delta,
            "lower": self.lower,
            "upper": self.upper,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "claim": self.claim.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureParetoStudyConfig:
    families: tuple[str, ...]
    hardware_profiles: tuple[str, ...]
    methods: tuple[str, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    bootstrap_repetitions: int = 1_000
    bootstrap_confidence: float = 0.95
    baseline_method: str = "greedy"

    def __post_init__(self) -> None:
        for label, values, minimum in (
            ("families", self.families, 2),
            ("hardware profiles", self.hardware_profiles, 2),
            ("methods", self.methods, 2),
        ):
            if len(values) < minimum or values != tuple(sorted(set(values))):
                raise ArchitectureParetoStudyError(f"study requires canonical {label}")
        if len(self.budgets) < 2 or self.budgets != tuple(sorted(set(self.budgets))):
            raise ArchitectureParetoStudyError("study requires two sorted unique budgets")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise ArchitectureParetoStudyError("study requires three sorted unique seeds")
        if self.baseline_method not in self.methods:
            raise ArchitectureParetoStudyError("baseline method must be in study methods")
        if self.bootstrap_repetitions < 100 or not 0 < self.bootstrap_confidence < 1:
            raise ArchitectureParetoStudyError("bootstrap configuration is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ARCHITECTURE_PARETO_STUDY_SCHEMA_VERSION,
            "families": list(self.families),
            "hardware_profiles": list(self.hardware_profiles),
            "methods": list(self.methods),
            "budgets": list(self.budgets),
            "seeds": list(self.seeds),
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_confidence": self.bootstrap_confidence,
            "baseline_method": self.baseline_method,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureParetoStudy:
    config: ArchitectureParetoStudyConfig
    cells: tuple[ArchitectureStudyCell, ...]
    comparisons: tuple[ArchitectureStudyComparison, ...]
    decision: ArchitectureStudyClaim
    reason: str

    def __post_init__(self) -> None:
        expected = {
            (family, hardware, method, budget, seed)
            for family, hardware, method, budget, seed in product(
                self.config.families,
                self.config.hardware_profiles,
                self.config.methods,
                self.config.budgets,
                self.config.seeds,
            )
        }
        actual = {
            (
                cell.key.family,
                cell.key.hardware_profile,
                cell.key.method,
                cell.key.budget,
                cell.key.seed,
            )
            for cell in self.cells
        }
        if actual != expected or len(actual) != len(self.cells):
            raise ArchitectureParetoStudyError("study must retain the complete comparison matrix")
        _text(self.reason, "study decision reason")

    @property
    def study_id(self) -> str:
        return (
            "architecture_pareto_study_"
            + hashlib.sha256(_canonical(self.to_record(include_id=False)).encode()).hexdigest()
        )

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": ARCHITECTURE_PARETO_STUDY_SCHEMA_VERSION,
            "config": self.config.to_record(),
            "cells": [item.to_record() for item in self.cells],
            "comparisons": [item.to_record() for item in self.comparisons],
            "decision": self.decision.value,
            "reason": self.reason,
        }
        if include_id:
            record["study_id"] = self.study_id
        return record


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_frontier(points: tuple[ArchitectureFrontierPoint, ...]) -> tuple[str, ...]:
    frontier = tuple(
        point
        for point in points
        if not any(
            other.point_id != point.point_id
            and other.quality_low >= point.quality_high
            and other.deployment_cost_high <= point.deployment_cost_low
            and (
                other.quality_low > point.quality_high
                or other.deployment_cost_high < point.deployment_cost_low
            )
            for other in points
        )
    )
    return tuple(sorted(point.point_id for point in frontier))


def compute_hypervolume(points: tuple[ArchitectureFrontierPoint, ...]) -> float:
    """Compute a bounded two-objective quality/cost area for measured points."""

    if not points:
        return 0.0
    reference_cost = max(point.deployment_cost_high for point in points)
    ordered = sorted(points, key=lambda point: (point.deployment_cost_high, -point.quality_low))
    area = 0.0
    previous_cost = ordered[0].deployment_cost_high
    best_quality = 0.0
    for point in ordered:
        cost = point.deployment_cost_high
        area += max(0.0, cost - previous_cost) * best_quality
        best_quality = max(best_quality, point.quality_low)
        previous_cost = cost
    area += max(0.0, reference_cost - previous_cost) * best_quality
    return area


def _bootstrap_interval(
    deltas: tuple[float, ...], seed: int, repetitions: int, confidence: float
) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(repetitions)
    ]
    samples.sort()
    lower = samples[max(0, math.floor(((1 - confidence) / 2) * repetitions))]
    upper_index = min(repetitions - 1, math.ceil((1 - (1 - confidence) / 2) * repetitions) - 1)
    return lower, samples[upper_index]


def _comparison(
    config: ArchitectureParetoStudyConfig,
    cells: tuple[ArchitectureStudyCell, ...],
    family: str,
    hardware: str,
    budget: int,
    method: str,
) -> ArchitectureStudyComparison:
    by_key = {
        (cell.key.method, cell.key.seed): cell
        for cell in cells
        if cell.key.family == family
        and cell.key.hardware_profile == hardware
        and cell.key.budget == budget
    }
    deltas: list[float] = []
    for seed in config.seeds:
        current = by_key.get((method, seed))
        baseline = by_key.get((config.baseline_method, seed))
        if (
            current is None
            or baseline is None
            or current.outcome
            not in {ArchitectureStudyOutcome.MEASURED, ArchitectureStudyOutcome.NEGATIVE_RESULT}
            or baseline.outcome
            not in {ArchitectureStudyOutcome.MEASURED, ArchitectureStudyOutcome.NEGATIVE_RESULT}
        ):
            return ArchitectureStudyComparison(
                family,
                hardware,
                budget,
                method,
                config.baseline_method,
                "hypervolume",
                None,
                None,
                None,
                config.bootstrap_repetitions,
                ArchitectureStudyClaim.UNSUPPORTED,
                "paired measured cells are incomplete",
            )
        current_metric = current.metric("hypervolume")
        baseline_metric = baseline.metric("hypervolume")
        if current_metric is None or baseline_metric is None:
            return ArchitectureStudyComparison(
                family,
                hardware,
                budget,
                method,
                config.baseline_method,
                "hypervolume",
                None,
                None,
                None,
                config.bootstrap_repetitions,
                ArchitectureStudyClaim.INCONCLUSIVE,
                "paired cells lack hypervolume metrics",
            )
        deltas.append(current_metric.value - baseline_metric.value)
    delta = sum(deltas) / len(deltas)
    lower, upper = _bootstrap_interval(
        tuple(deltas),
        seed=budget + len(family) + len(hardware),
        repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
    )
    claim = (
        ArchitectureStudyClaim.POSITIVE
        if lower > 0
        else ArchitectureStudyClaim.NEGATIVE_RESULT
        if upper < 0
        else ArchitectureStudyClaim.INCONCLUSIVE
    )
    return ArchitectureStudyComparison(
        family,
        hardware,
        budget,
        method,
        config.baseline_method,
        "hypervolume",
        delta,
        lower,
        upper,
        config.bootstrap_repetitions,
        claim,
        "paired seed bootstrap over complete frontier metrics",
    )


def build_architecture_pareto_study(
    config: ArchitectureParetoStudyConfig,
    cells: tuple[ArchitectureStudyCell, ...],
) -> ArchitectureParetoStudy:
    comparisons = tuple(
        _comparison(config, cells, family, hardware, budget, method)
        for family, hardware, budget, method in product(
            config.families,
            config.hardware_profiles,
            config.budgets,
            (method for method in config.methods if method != config.baseline_method),
        )
    )
    claims = {comparison.claim for comparison in comparisons}
    if ArchitectureStudyClaim.POSITIVE in claims:
        decision = ArchitectureStudyClaim.POSITIVE
        reason = "at least one paired comparison has a positive bootstrap interval"
    elif claims and claims <= {ArchitectureStudyClaim.NEGATIVE_RESULT}:
        decision = ArchitectureStudyClaim.NEGATIVE_RESULT
        reason = "all complete paired comparisons are non-improving"
    elif ArchitectureStudyClaim.UNSUPPORTED in claims and len(claims) == 1:
        decision = ArchitectureStudyClaim.UNSUPPORTED
        reason = "physical paired evidence is unavailable"
    else:
        decision = ArchitectureStudyClaim.INCONCLUSIVE
        reason = "paired frontier evidence does not support a directional claim"
    return ArchitectureParetoStudy(config, cells, comparisons, decision, reason)


def build_default_architecture_pareto_study() -> ArchitectureParetoStudy:
    config = ArchitectureParetoStudyConfig(
        ("llama", "qwen"),
        ("cpu", "cuda"),
        ("beam", "competitor", "evolutionary", "greedy", "one_axis", "surrogate"),
        (8, 32),
        (11, 23, 47),
    )
    cells = tuple(
        ArchitectureStudyCell(
            ArchitectureStudyKey(family, hardware, method, budget, seed),
            ArchitectureStudyOutcome.UNSUPPORTED,
            (),
            (),
            "no licensed multi-family physical artifact and deployment bundle is available",
        )
        for family, hardware, method, budget, seed in product(
            config.families,
            config.hardware_profiles,
            config.methods,
            config.budgets,
            config.seeds,
        )
    )
    return build_architecture_pareto_study(config, cells)


DEFAULT_ARCHITECTURE_PARETO_STUDY = build_default_architecture_pareto_study()
