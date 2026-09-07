"""Evidence records and conservative recommendations for consumer repair studies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

CONSUMER_REPAIR_STUDY_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_UNITS = {
    "artifact_bytes": "bytes",
    "deployment_cost": "cost_units",
    "energy_joules": "joules",
    "failure_rate": "fraction",
    "heldout_quality": "score",
    "overfit_rate": "fraction",
    "quality_gain_per_joule": "score/joule",
    "quality_gain_per_second": "score/second",
    "quality_gain_per_token": "score/token",
    "repair_seconds": "seconds",
    "rollback_rate": "fraction",
    "tokens_per_second": "tokens/second",
}


class ConsumerRepairStudyError(ValueError):
    """Raised when consumer repair evidence is incomplete or incomparable."""


class ConsumerRepairArm(StrEnum):
    NO_REPAIR = "no_repair"
    FIXED_REPAIR = "fixed_repair"
    ORACLE_BUDGET = "oracle_budget"
    LEARNED_JOINT = "learned_joint"


class ConsumerRepairOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ConsumerRepairClaim(StrEnum):
    SUPPORTED = "supported"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


class RepairRecommendation(StrEnum):
    BENEFICIAL = "beneficial"
    HARMFUL = "harmful"
    UNNECESSARY = "unnecessary"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsumerRepairStudyError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise ConsumerRepairStudyError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConsumerRepairStudyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConsumerRepairStudyError(f"{label} must be finite")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise ConsumerRepairStudyError(f"{label} cannot be negative")
    return result


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConsumerRepairStudyError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ConsumerRepairStudyConfig:
    """Preregistered matrix and decision thresholds."""

    model_families: tuple[str, ...] = ("llama", "qwen")
    model_sizes: tuple[str, ...] = ("medium", "small")
    damage_levels: tuple[str, ...] = ("high", "low", "medium")
    repair_budgets: tuple[str, ...] = ("long", "medium", "short")
    hardware_profiles: tuple[str, ...] = ("cpu_offload", "low_vram_12gb")
    seeds: tuple[int, ...] = (0, 1, 2)
    minimum_quality_improvement: float = 0.01
    confidence_level: float = 0.95
    require_complete_matrix: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            ("model families", self.model_families, 2),
            ("model sizes", self.model_sizes, 2),
            ("damage levels", self.damage_levels, 3),
            ("repair budgets", self.repair_budgets, 3),
            ("hardware profiles", self.hardware_profiles, 2),
        )
        for label, values, minimum in dimensions:
            if len(values) < minimum or values != tuple(sorted(set(values))) or any(
                not value.strip() for value in values
            ):
                raise ConsumerRepairStudyError(f"{label} must be canonical and sufficiently broad")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise ConsumerRepairStudyError("consumer repair studies require three sorted seeds")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise ConsumerRepairStudyError("consumer repair seeds must be unsigned")
        if _finite(self.minimum_quality_improvement, "minimum quality improvement") < 0:
            raise ConsumerRepairStudyError("minimum quality improvement cannot be negative")
        confidence = _finite(self.confidence_level, "confidence level")
        if not 0.5 < confidence < 1:
            raise ConsumerRepairStudyError("confidence level must be between 0.5 and 1")

    def expected_keys(self) -> tuple[ConsumerRepairKey, ...]:
        return tuple(
            ConsumerRepairKey(family, size, hardware, damage, budget, arm, seed)
            for family in self.model_families
            for size in self.model_sizes
            for hardware in self.hardware_profiles
            for damage in self.damage_levels
            for budget in self.repair_budgets
            for arm in ConsumerRepairArm
            for seed in self.seeds
        )

    def to_record(self) -> dict[str, object]:
        return {
            "model_families": list(self.model_families),
            "model_sizes": list(self.model_sizes),
            "damage_levels": list(self.damage_levels),
            "repair_budgets": list(self.repair_budgets),
            "hardware_profiles": list(self.hardware_profiles),
            "seeds": list(self.seeds),
            "minimum_quality_improvement": self.minimum_quality_improvement,
            "confidence_level": self.confidence_level,
            "require_complete_matrix": self.require_complete_matrix,
            "hardware_limit": "12 GiB low-VRAM or CPU offload where feasible",
        }


@dataclass(frozen=True, slots=True)
class ConsumerRepairKey:
    model_family: str
    model_size: str
    hardware_profile: str
    damage_level: str
    repair_budget: str
    arm: ConsumerRepairArm
    seed: int

    def __post_init__(self) -> None:
        for label, value in (
            ("model family", self.model_family),
            ("model size", self.model_size),
            ("hardware profile", self.hardware_profile),
            ("damage level", self.damage_level),
            ("repair budget", self.repair_budget),
        ):
            _text(value, label)
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ConsumerRepairStudyError("consumer repair seed must be unsigned")

    @property
    def matched_region(self) -> tuple[str, ...]:
        return (
            self.model_family,
            self.model_size,
            self.hardware_profile,
            self.damage_level,
            self.repair_budget,
            str(self.seed),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "model_size": self.model_size,
            "hardware_profile": self.hardware_profile,
            "damage_level": self.damage_level,
            "repair_budget": self.repair_budget,
            "arm": self.arm.value,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ConsumerRepairMetric:
    name: str
    value: float

    def __post_init__(self) -> None:
        if self.name not in _METRIC_UNITS:
            raise ConsumerRepairStudyError(f"unknown consumer repair metric: {self.name}")
        _finite(self.value, f"metric {self.name}")
        if self.name.endswith("rate") and not 0 <= self.value <= 1:
            raise ConsumerRepairStudyError(f"metric {self.name} must be within [0, 1]")
        if self.name in {"artifact_bytes", "deployment_cost", "energy_joules", "repair_seconds"}:
            _nonnegative(self.value, f"metric {self.name}")

    @property
    def unit(self) -> str:
        return _METRIC_UNITS[self.name]

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "unit": self.unit, "value": self.value}


@dataclass(frozen=True, slots=True)
class ConsumerRepairInterval:
    metric: str
    lower: float
    upper: float
    level: float

    def __post_init__(self) -> None:
        if self.metric not in _METRIC_UNITS:
            raise ConsumerRepairStudyError("interval references an unknown metric")
        if self.lower > self.upper:
            raise ConsumerRepairStudyError("consumer repair interval is reversed")
        _finite(self.lower, "interval lower")
        _finite(self.upper, "interval upper")
        confidence = _finite(self.level, "interval confidence")
        if not 0.5 < confidence < 1:
            raise ConsumerRepairStudyError("interval confidence must be between 0.5 and 1")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
        }


@dataclass(frozen=True, slots=True)
class ConsumerRepairEvidence:
    key: ConsumerRepairKey
    outcome: ConsumerRepairOutcome
    source_artifact_digest: str | None
    teacher_artifact_digest: str | None
    final_artifact_digest: str | None
    data_revision: str | None
    hardware_context_id: str | None
    metrics: tuple[ConsumerRepairMetric, ...]
    intervals: tuple[ConsumerRepairInterval, ...]
    repetitions: int | None
    provenance: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        names = tuple(metric.name for metric in self.metrics)
        if names != tuple(sorted(set(names))):
            raise ConsumerRepairStudyError("consumer repair metrics must be unique and sorted")
        interval_names = tuple(item.metric for item in self.intervals)
        if interval_names != tuple(sorted(set(interval_names))):
            raise ConsumerRepairStudyError("consumer repair intervals must be unique and sorted")
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise ConsumerRepairStudyError("consumer repair provenance must be canonical")
        measured = self.outcome in {
            ConsumerRepairOutcome.MEASURED,
            ConsumerRepairOutcome.NEGATIVE_RESULT,
        }
        if measured:
            if not all(
                value is not None
                for value in (
                    self.source_artifact_digest,
                    self.final_artifact_digest,
                    self.data_revision,
                    self.hardware_context_id,
                    self.repetitions,
                )
            ):
                raise ConsumerRepairStudyError("measured cells require complete provenance")
            _digest(self.source_artifact_digest, "source artifact digest")
            _digest(self.final_artifact_digest, "final artifact digest")
            _text(self.data_revision, "data revision")
            _text(self.hardware_context_id, "hardware context ID")
            _positive(self.repetitions, "measurement repetitions")
            if not self.provenance:
                raise ConsumerRepairStudyError("measured cells require provenance")
            if set(names) != set(_METRIC_UNITS):
                raise ConsumerRepairStudyError("measured cells require the complete metric set")
            if set(interval_names) != {"deployment_cost", "heldout_quality"}:
                raise ConsumerRepairStudyError(
                    "measured cells require held-out quality and deployment intervals"
                )
            if self.outcome is ConsumerRepairOutcome.MEASURED and self.reason is not None:
                raise ConsumerRepairStudyError("measured cells cannot carry a reason")
            if self.outcome is ConsumerRepairOutcome.NEGATIVE_RESULT and not self.reason:
                raise ConsumerRepairStudyError("negative cells require a reason")
        else:
            if any(
                value is not None
                for value in (
                    self.source_artifact_digest,
                    self.teacher_artifact_digest,
                    self.final_artifact_digest,
                    self.data_revision,
                    self.hardware_context_id,
                    self.repetitions,
                )
            ) or self.metrics or self.intervals:
                raise ConsumerRepairStudyError("terminal cells cannot retain partial measurements")
            if not self.reason:
                raise ConsumerRepairStudyError("terminal cells require a reason")

    @property
    def metric_map(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CONSUMER_REPAIR_STUDY_SCHEMA_VERSION,
            "key": self.key.to_record(),
            "outcome": self.outcome.value,
            "source_artifact_digest": self.source_artifact_digest,
            "teacher_artifact_digest": self.teacher_artifact_digest,
            "final_artifact_digest": self.final_artifact_digest,
            "data_revision": self.data_revision,
            "hardware_context_id": self.hardware_context_id,
            "metrics": [item.to_record() for item in self.metrics],
            "intervals": [item.to_record() for item in self.intervals],
            "repetitions": self.repetitions,
            "provenance": list(self.provenance),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ConsumerRepairCell:
    key: ConsumerRepairKey
    evidence: ConsumerRepairEvidence

    def __post_init__(self) -> None:
        if self.key != self.evidence.key:
            raise ConsumerRepairStudyError("cell key and evidence key differ")

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.evidence.to_record()).encode()).hexdigest()
        return f"consumer_repair_{digest}"

    def to_record(self) -> dict[str, object]:
        return {"cell_id": self.cell_id, "evidence": self.evidence.to_record()}


@dataclass(frozen=True, slots=True)
class ConsumerRepairRecommendationRecord:
    region: tuple[str, ...]
    recommendation: RepairRecommendation
    selected_arm: ConsumerRepairArm | None
    reason: str

    def __post_init__(self) -> None:
        if self.region != tuple(self.region) or not self.region:
            raise ConsumerRepairStudyError("recommendation regions must be non-empty")
        _text(self.reason, "recommendation reason")

    def to_record(self) -> dict[str, object]:
        return {
            "region": list(self.region),
            "recommendation": self.recommendation.value,
            "selected_arm": None if self.selected_arm is None else self.selected_arm.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ConsumerRepairStudy:
    cells: tuple[ConsumerRepairCell, ...]
    config: ConsumerRepairStudyConfig
    study_id: str
    schema_version: int = CONSUMER_REPAIR_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONSUMER_REPAIR_STUDY_SCHEMA_VERSION:
            raise ConsumerRepairStudyError("unsupported consumer repair study schema")
        if not self.cells:
            raise ConsumerRepairStudyError("consumer repair studies require cells")
        keys = tuple(cell.key.to_record().__repr__() for cell in self.cells)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ConsumerRepairStudyError("consumer repair cells must be unique and canonical")
        if self.config.require_complete_matrix:
            actual = {cell.key for cell in self.cells}
            expected = set(self.config.expected_keys())
            if actual != expected:
                raise ConsumerRepairStudyError("consumer repair matrix is incomplete")
        expected_id = self._expected_study_id()
        if self.study_id != expected_id:
            raise ConsumerRepairStudyError("consumer repair study identity does not reconcile")

    def _expected_study_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "config": self.config.to_record(),
            "cells": [item.to_record() for item in self.cells],
        }
        return "consumer_repair_study_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def _cells_for_region(self, region: tuple[str, ...]) -> tuple[ConsumerRepairCell, ...]:
        return tuple(
            cell
            for cell in self.cells
            if cell.key.matched_region == region
        )

    def recommendation(self, region: tuple[str, ...]) -> ConsumerRepairRecommendationRecord:
        cells = self._cells_for_region(region)
        baseline = next(
            (cell for cell in cells if cell.key.arm is ConsumerRepairArm.NO_REPAIR), None
        )
        if baseline is None or baseline.evidence.outcome not in {
            ConsumerRepairOutcome.MEASURED,
            ConsumerRepairOutcome.NEGATIVE_RESULT,
        }:
            return ConsumerRepairRecommendationRecord(
                region, RepairRecommendation.UNKNOWN, None, "no measured no-repair baseline"
            )
        baseline_quality = baseline.evidence.metric_map.get("heldout_quality")
        if baseline_quality is None:
            return ConsumerRepairRecommendationRecord(
                region, RepairRecommendation.UNKNOWN, None, "baseline lacks held-out quality"
            )
        repairs = tuple(
            cell
            for cell in cells
            if cell.key.arm is not ConsumerRepairArm.NO_REPAIR
            and cell.evidence.outcome is ConsumerRepairOutcome.MEASURED
        )
        if not repairs:
            if any(
                cell.evidence.outcome
                in {ConsumerRepairOutcome.UNSUPPORTED, ConsumerRepairOutcome.FAILED}
                for cell in cells
                if cell.key.arm is not ConsumerRepairArm.NO_REPAIR
            ):
                return ConsumerRepairRecommendationRecord(
                    region,
                    RepairRecommendation.INFEASIBLE,
                    None,
                    "all repair arms are unsupported or failed",
                )
            return ConsumerRepairRecommendationRecord(
                region, RepairRecommendation.UNKNOWN, None, "repair cells are incomplete"
            )
        best = max(
            repairs,
            key=lambda cell: (
                cell.evidence.metric_map["heldout_quality"],
                -cell.evidence.metric_map["deployment_cost"],
                cell.key.arm.value,
            ),
        )
        quality = best.evidence.metric_map["heldout_quality"]
        improvement = quality - baseline_quality
        if improvement < 0:
            recommendation = RepairRecommendation.HARMFUL
            reason = "best measured repair is below the matched no-repair quality"
        elif improvement < self.config.minimum_quality_improvement:
            recommendation = RepairRecommendation.UNNECESSARY
            reason = "measured repair gain does not clear the preregistered threshold"
        else:
            recommendation = RepairRecommendation.BENEFICIAL
            reason = "measured held-out quality clears the preregistered threshold"
        return ConsumerRepairRecommendationRecord(region, recommendation, best.key.arm, reason)

    def recommendations(self) -> tuple[ConsumerRepairRecommendationRecord, ...]:
        regions = tuple(sorted({cell.key.matched_region for cell in self.cells}))
        return tuple(self.recommendation(region) for region in regions)

    def summary(self) -> dict[str, object]:
        outcomes = [cell.evidence.outcome for cell in self.cells]
        recommendations = self.recommendations()
        measured = sum(
            outcome
            in {ConsumerRepairOutcome.MEASURED, ConsumerRepairOutcome.NEGATIVE_RESULT}
            for outcome in outcomes
        )
        beneficial = sum(
            item.recommendation is RepairRecommendation.BENEFICIAL for item in recommendations
        )
        if not measured:
            claim = ConsumerRepairClaim.UNSUPPORTED
        elif beneficial:
            claim = ConsumerRepairClaim.SUPPORTED
        elif all(
            item.recommendation is RepairRecommendation.HARMFUL for item in recommendations
            if item.recommendation is not RepairRecommendation.UNKNOWN
        ):
            claim = ConsumerRepairClaim.NEGATIVE_RESULT
        else:
            claim = ConsumerRepairClaim.INCONCLUSIVE
        return {
            "study_id": self.study_id,
            "claim": claim.value,
            "cell_count": len(self.cells),
            "measured_or_negative_cells": measured,
            "outcomes": {
                outcome.value: outcomes.count(outcome) for outcome in ConsumerRepairOutcome
            },
            "recommendations": [item.to_record() for item in recommendations],
        }

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "config": self.config.to_record(),
            "cells": [item.to_record() for item in self.cells],
            "summary": self.summary(),
        }


def build_consumer_repair_study(
    cells: Iterable[ConsumerRepairCell],
    config: ConsumerRepairStudyConfig,
) -> ConsumerRepairStudy:
    """Build a deterministic study, enforcing full coverage when configured."""

    ordered = tuple(sorted(cells, key=lambda item: item.key.to_record().__repr__()))
    payload = {
        "schema_version": CONSUMER_REPAIR_STUDY_SCHEMA_VERSION,
        "config": config.to_record(),
        "cells": [item.to_record() for item in ordered],
    }
    study_id = "consumer_repair_study_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return ConsumerRepairStudy(ordered, config, study_id)


def build_default_consumer_repair_study() -> ConsumerRepairStudy:
    """Return the complete preregistered matrix without inventing measurements."""

    config = ConsumerRepairStudyConfig()
    cells = tuple(
        ConsumerRepairCell(
            key,
            ConsumerRepairEvidence(
                key,
                ConsumerRepairOutcome.UNSUPPORTED,
                None,
                None,
                None,
                None,
                None,
                (),
                (),
                None,
                (),
                "no licensed representative artifact was measured in this contract revision",
            ),
        )
        for key in config.expected_keys()
    )
    return build_consumer_repair_study(cells, config)


DEFAULT_CONSUMER_REPAIR_STUDY = build_default_consumer_repair_study()
