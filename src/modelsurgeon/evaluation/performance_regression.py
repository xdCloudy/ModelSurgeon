"""Hardware-bound runtime, memory, VRAM, and I/O regression budgets."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.runtime_telemetry import HardwareNormalizationContext

PERFORMANCE_BUDGET_SCHEMA_VERSION = 1
PERFORMANCE_REPORT_SCHEMA_VERSION = 1


class PerformanceRegressionError(ValueError):
    """Raised when measurements cannot support a valid regression decision."""


class RegressionLane(StrEnum):
    PR_CPU = "pr_cpu"
    LARGE_CPU = "large_cpu"
    GPU = "gpu"


class RegressionDirection(StrEnum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


@dataclass(frozen=True, slots=True)
class PerformanceBudget:
    case_id: str
    stage: str
    metric: str
    unit: str
    direction: RegressionDirection
    baseline: float
    relative_tolerance: float
    absolute_tolerance: float

    def __post_init__(self) -> None:
        if any(not value for value in (self.case_id, self.stage, self.metric, self.unit)):
            raise PerformanceRegressionError("performance budgets require canonical names")
        values = (self.baseline, self.relative_tolerance, self.absolute_tolerance)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise PerformanceRegressionError(
                "performance baselines and tolerances must be finite and non-negative"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.case_id, self.stage, self.metric

    @property
    def allowance(self) -> float:
        return max(self.absolute_tolerance, self.baseline * self.relative_tolerance)

    @property
    def threshold(self) -> float:
        if self.direction is RegressionDirection.MAXIMUM:
            return self.baseline + self.allowance
        return max(0.0, self.baseline - self.allowance)

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "metric": self.metric,
            "unit": self.unit,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class PerformanceBudgetManifest:
    profile: str
    lane: RegressionLane
    fixture_id: str
    min_repetitions: int
    reference_hardware: dict[str, object]
    budgets: tuple[PerformanceBudget, ...]
    schema_version: int = PERFORMANCE_BUDGET_SCHEMA_VERSION
    _reference_hardware_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_BUDGET_SCHEMA_VERSION:
            raise PerformanceRegressionError("unsupported performance budget schema")
        if not self.profile or not self.fixture_id or not self.reference_hardware:
            raise PerformanceRegressionError(
                "performance profile, fixture, and reference hardware are required"
            )
        if self.min_repetitions <= 0:
            raise PerformanceRegressionError("performance repetitions must be positive")
        keys = tuple(item.key for item in self.budgets)
        if not keys or keys != tuple(sorted(set(keys))):
            raise PerformanceRegressionError(
                "performance budgets must use unique canonical keys"
            )
        try:
            snapshot = canonical_identity_json(self.reference_hardware)
        except (TypeError, ValueError) as error:
            raise PerformanceRegressionError(
                "reference hardware must be canonical JSON"
            ) from error
        object.__setattr__(self, "_reference_hardware_json", snapshot)

    @property
    def reference_hardware_record(self) -> dict[str, object]:
        """Return the immutable canonical hardware snapshot bound to this manifest."""

        return cast(dict[str, object], json.loads(self._reference_hardware_json))

    @property
    def manifest_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self.to_record(include_id=False)).encode("utf-8")
        ).hexdigest()
        return f"performance_budget_{digest}"

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "lane": self.lane.value,
            "fixture_id": self.fixture_id,
            "min_repetitions": self.min_repetitions,
            "reference_hardware": self.reference_hardware_record,
            "budgets": [item.to_record() for item in self.budgets],
        }
        if include_id:
            record["manifest_id"] = self.manifest_id
        return record


@dataclass(frozen=True, slots=True)
class PerformanceMeasurement:
    case_id: str
    stage: str
    metric: str
    unit: str
    value: float
    repetition: int
    fixture_id: str
    hardware: HardwareNormalizationContext

    def __post_init__(self) -> None:
        if any(not value for value in (self.case_id, self.stage, self.metric, self.unit)):
            raise PerformanceRegressionError("performance measurements require names")
        if not math.isfinite(self.value) or self.value < 0:
            raise PerformanceRegressionError(
                "performance measurements must be finite and non-negative"
            )
        if self.repetition < 0 or not self.fixture_id:
            raise PerformanceRegressionError(
                "performance measurements require repetition and fixture identity"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.case_id, self.stage, self.metric

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "metric": self.metric,
            "unit": self.unit,
            "value": self.value,
            "repetition": self.repetition,
            "fixture_id": self.fixture_id,
            "hardware_context_id": self.hardware.context_id,
        }


@dataclass(frozen=True, slots=True)
class PerformanceBudgetResult:
    budget: PerformanceBudget
    observed_median: float
    observed_values: tuple[float, ...]
    passed: bool

    def to_record(self) -> dict[str, object]:
        return {
            "budget": self.budget.to_record(),
            "observed_median": self.observed_median,
            "observed_values": list(self.observed_values),
            "passed": self.passed,
            "alert": None
            if self.passed
            else (
                f"{self.budget.case_id}/{self.budget.stage}/{self.budget.metric} "
                f"median {self.observed_median:g} violates "
                f"{self.budget.direction.value} threshold {self.budget.threshold:g}"
            ),
        }


@dataclass(frozen=True, slots=True)
class PerformanceRegressionReport:
    manifest: PerformanceBudgetManifest
    hardware: HardwareNormalizationContext
    results: tuple[PerformanceBudgetResult, ...]
    schema_version: int = PERFORMANCE_REPORT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    @property
    def report_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self.to_record(include_id=False)).encode("utf-8")
        ).hexdigest()
        return f"performance_report_{digest}"

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "performance_regression_report",
            "schema_version": self.schema_version,
            "profile": self.manifest.profile,
            "lane": self.manifest.lane.value,
            "fixture_id": self.manifest.fixture_id,
            "budget_manifest_id": self.manifest.manifest_id,
            "reference_hardware": self.manifest.reference_hardware_record,
            "hardware_context_id": self.hardware.context_id,
            "hardware": self.hardware.to_record(),
            "results": [item.to_record() for item in self.results],
            "passed": self.passed,
        }
        if include_id:
            record["report_id"] = self.report_id
        return record


def evaluate_performance_regression(
    manifest: PerformanceBudgetManifest,
    measurements: tuple[PerformanceMeasurement, ...],
) -> PerformanceRegressionReport:
    """Compare repeated same-hardware measurements with immutable alert thresholds."""

    if not measurements:
        raise PerformanceRegressionError("performance regression requires measurements")
    hardware_ids = {item.hardware.context_id for item in measurements}
    if len(hardware_ids) != 1:
        raise PerformanceRegressionError(
            "performance measurements mix incomparable hardware contexts"
        )
    fixtures = {item.fixture_id for item in measurements}
    if fixtures != {manifest.fixture_id}:
        raise PerformanceRegressionError(
            "performance measurements do not match the budget fixture"
        )
    budgets = {item.key: item for item in manifest.budgets}
    unknown = {item.key for item in measurements} - set(budgets)
    if unknown:
        raise PerformanceRegressionError(
            f"measurements have no declared budget: {sorted(unknown)}"
        )

    results: list[PerformanceBudgetResult] = []
    for key, budget in sorted(budgets.items()):
        selected = tuple(item for item in measurements if item.key == key)
        repetitions = tuple(item.repetition for item in selected)
        if len(selected) < manifest.min_repetitions:
            raise PerformanceRegressionError(
                f"budget {key} has fewer than {manifest.min_repetitions} repetitions"
            )
        if len(repetitions) != len(set(repetitions)):
            raise PerformanceRegressionError(f"budget {key} repeats a repetition index")
        if any(item.unit != budget.unit for item in selected):
            raise PerformanceRegressionError(f"budget {key} has inconsistent units")
        values = tuple(sorted(item.value for item in selected))
        observed = float(statistics.median(values))
        passed = (
            observed <= budget.threshold
            if budget.direction is RegressionDirection.MAXIMUM
            else observed >= budget.threshold
        )
        results.append(PerformanceBudgetResult(budget, observed, values, passed))
    return PerformanceRegressionReport(manifest, measurements[0].hardware, tuple(results))


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PerformanceRegressionError(f"{path} must be an object")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PerformanceRegressionError(f"{path} must be a non-empty string")
    return value


def _number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PerformanceRegressionError(f"{path} must be numeric")
    return float(value)


def load_performance_budget_manifest(
    payload: str,
    profile: str,
) -> PerformanceBudgetManifest:
    """Load one strict profile from the versioned multi-profile budget file."""

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PerformanceRegressionError("performance budget file is invalid JSON") from error
    root = _mapping(raw, "budget file")
    if set(root) != {"schema_version", "profiles"}:
        raise PerformanceRegressionError("performance budget file fields are invalid")
    if root["schema_version"] != PERFORMANCE_BUDGET_SCHEMA_VERSION:
        raise PerformanceRegressionError("unsupported performance budget file schema")
    profiles = _mapping(root["profiles"], "profiles")
    if profile not in profiles:
        raise PerformanceRegressionError(f"unknown performance profile {profile!r}")
    selected = _mapping(profiles[profile], f"profiles.{profile}")
    expected = {
        "lane",
        "fixture_id",
        "min_repetitions",
        "reference_hardware",
        "budgets",
    }
    if set(selected) != expected:
        raise PerformanceRegressionError("performance profile fields are invalid")
    raw_budgets = selected["budgets"]
    if not isinstance(raw_budgets, list):
        raise PerformanceRegressionError("performance budgets must be an array")
    budgets: list[PerformanceBudget] = []
    for index, value in enumerate(raw_budgets):
        item = _mapping(value, f"budgets[{index}]")
        if set(item) != {
            "case_id",
            "stage",
            "metric",
            "unit",
            "direction",
            "baseline",
            "relative_tolerance",
            "absolute_tolerance",
        }:
            raise PerformanceRegressionError(f"budget {index} fields are invalid")
        try:
            direction = RegressionDirection(_string(item["direction"], "direction"))
        except ValueError as error:
            raise PerformanceRegressionError("budget direction is invalid") from error
        budgets.append(
            PerformanceBudget(
                _string(item["case_id"], "case_id"),
                _string(item["stage"], "stage"),
                _string(item["metric"], "metric"),
                _string(item["unit"], "unit"),
                direction,
                _number(item["baseline"], "baseline"),
                _number(item["relative_tolerance"], "relative_tolerance"),
                _number(item["absolute_tolerance"], "absolute_tolerance"),
            )
        )
    repetitions = selected["min_repetitions"]
    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise PerformanceRegressionError("min_repetitions must be an integer")
    try:
        lane = RegressionLane(_string(selected["lane"], "lane"))
    except ValueError as error:
        raise PerformanceRegressionError("performance lane is invalid") from error
    return PerformanceBudgetManifest(
        profile,
        lane,
        _string(selected["fixture_id"], "fixture_id"),
        repetitions,
        _mapping(selected["reference_hardware"], "reference_hardware"),
        tuple(sorted(budgets, key=lambda item: item.key)),
    )
