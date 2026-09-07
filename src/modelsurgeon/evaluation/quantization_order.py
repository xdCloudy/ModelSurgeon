"""Matched quantization-order study records and conservative attribution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

QUANTIZATION_ORDER_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_UNITS = {
    "quality": "score",
    "artifact_size_bytes": "bytes",
    "decode_tokens_per_second": "tokens/s",
    "peak_ram_bytes": "bytes",
    "optimization_seconds": "s",
}


class QuantizationOrderError(ValueError):
    """Raised when matched order evidence is incomplete or non-comparable."""


class QuantizationOrderArm(StrEnum):
    QUANTIZATION_ONLY = "quantization_only"
    SURGERY_THEN_QUANTIZE = "surgery_then_quantize"
    QUANTIZE_THEN_SURGERY = "quantize_then_surgery"
    HIGH_PRECISION_SURGERY = "high_precision_surgery"


class QuantizationOrderOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class QuantizationOrderMetricState(StrEnum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuantizationOrderError(f"{label} is required")
    return value


def _digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise QuantizationOrderError(f"{label} must be a lowercase SHA-256")


def _value(value: float | None) -> float:
    if value is None:
        raise QuantizationOrderError("measured order metric unexpectedly has no value")
    return value


@dataclass(frozen=True, slots=True)
class QuantizationOrderStudyKey:
    source_digest: str
    model_family: str
    compression_level: str
    codec: str
    corpus_id: str
    evaluator_id: str
    runtime_id: str
    seed: int
    structure_id: str

    def __post_init__(self) -> None:
        _digest(self.source_digest, "source digest")
        for label, value in (
            ("model family", self.model_family),
            ("compression level", self.compression_level),
            ("codec", self.codec),
            ("corpus ID", self.corpus_id),
            ("evaluator ID", self.evaluator_id),
            ("runtime ID", self.runtime_id),
            ("structure ID", self.structure_id),
        ):
            _text(value, label)
        if self.seed < 0:
            raise QuantizationOrderError("seed must be unsigned")

    def to_record(self) -> dict[str, object]:
        return {
            "source_digest": self.source_digest,
            "model_family": self.model_family,
            "compression_level": self.compression_level,
            "codec": self.codec,
            "corpus_id": self.corpus_id,
            "evaluator_id": self.evaluator_id,
            "runtime_id": self.runtime_id,
            "seed": self.seed,
            "structure_id": self.structure_id,
        }


@dataclass(frozen=True, slots=True)
class QuantizationOrderMetric:
    name: str
    state: QuantizationOrderMetricState
    value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.name not in _METRIC_UNITS:
            raise QuantizationOrderError(f"unknown order-study metric: {self.name}")
        if self.state is QuantizationOrderMetricState.MEASURED:
            if self.value is None or not math.isfinite(self.value) or self.reason is not None:
                raise QuantizationOrderError("measured order metrics require a finite value only")
        elif self.value is not None or not self.reason:
            raise QuantizationOrderError("unavailable order metrics require a reason only")

    @property
    def unit(self) -> str:
        return _METRIC_UNITS[self.name]

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "state": self.state.value,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizationOrderCell:
    key: QuantizationOrderStudyKey
    arm: QuantizationOrderArm
    outcome: QuantizationOrderOutcome
    artifact_digest: str | None
    metrics: tuple[QuantizationOrderMetric, ...]
    tool_revision: str
    outcome_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.tool_revision, "quantization tool revision")
        names = tuple(metric.name for metric in self.metrics)
        if names != tuple(sorted(set(names))):
            raise QuantizationOrderError("order-study metrics must be unique and sorted")
        if self.outcome is QuantizationOrderOutcome.MEASURED:
            if self.artifact_digest is None:
                raise QuantizationOrderError("measured order cells require an artifact digest")
            _digest(self.artifact_digest, "artifact digest")
            if set(names) != set(_METRIC_UNITS) or any(
                metric.state is not QuantizationOrderMetricState.MEASURED for metric in self.metrics
            ):
                raise QuantizationOrderError("measured order cells require every measured metric")
            if self.outcome_reason is not None:
                raise QuantizationOrderError("measured order cells cannot carry a reason")
        elif not self.outcome_reason:
            raise QuantizationOrderError("terminal order cells require a reason")

    @property
    def cell_id(self) -> str:
        identity = {
            "schema_version": QUANTIZATION_ORDER_SCHEMA_VERSION,
            "key": self.key.to_record(),
            "arm": self.arm.value,
        }
        return "quant_order_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": QUANTIZATION_ORDER_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "key": self.key.to_record(),
            "arm": self.arm.value,
            "outcome": self.outcome.value,
            "artifact_digest": self.artifact_digest,
            "metrics": [metric.to_record() for metric in self.metrics],
            "tool_revision": self.tool_revision,
            "outcome_reason": self.outcome_reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizationOrderDelta:
    metric: str
    baseline: float
    comparison: float
    delta: float

    def __post_init__(self) -> None:
        if self.metric not in _METRIC_UNITS or not all(
            math.isfinite(value) for value in (self.baseline, self.comparison, self.delta)
        ):
            raise QuantizationOrderError("order delta is not finite or uses an unknown metric")
        if self.delta != self.comparison - self.baseline:
            raise QuantizationOrderError("order delta does not reconcile")

    @property
    def unit(self) -> str:
        return _METRIC_UNITS[self.metric]

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "baseline": self.baseline,
            "comparison": self.comparison,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class QuantizationOrderComparison:
    key: QuantizationOrderStudyKey
    outcome: QuantizationOrderOutcome
    quantization_effect: tuple[QuantizationOrderDelta, ...]
    surgery_effect: tuple[QuantizationOrderDelta, ...]
    interaction_effect: tuple[QuantizationOrderDelta, ...]
    preferred_arm: QuantizationOrderArm | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is QuantizationOrderOutcome.MEASURED:
            if self.reason is not None or self.preferred_arm is None:
                raise QuantizationOrderError(
                    "measured comparisons require a preferred arm and no reason"
                )
        elif not self.reason:
            raise QuantizationOrderError("non-measured comparisons require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "key": self.key.to_record(),
            "outcome": self.outcome.value,
            "quantization_effect": [item.to_record() for item in self.quantization_effect],
            "surgery_effect": [item.to_record() for item in self.surgery_effect],
            "interaction_effect": [item.to_record() for item in self.interaction_effect],
            "preferred_arm": None if self.preferred_arm is None else self.preferred_arm.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizationOrderStudy:
    cells: tuple[QuantizationOrderCell, ...]
    study_id: str
    schema_version: int = QUANTIZATION_ORDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QUANTIZATION_ORDER_SCHEMA_VERSION:
            raise QuantizationOrderError("unsupported quantization-order schema version")
        ids = tuple(cell.cell_id for cell in self.cells)
        if not self.cells or len(ids) != len(set(ids)):
            raise QuantizationOrderError("study cells must be non-empty and uniquely identified")
        expected = (
            "quant_order_study_"
            + hashlib.sha256(
                _canonical([cell.to_record() for cell in self.cells]).encode()
            ).hexdigest()
        )
        if self.study_id != expected:
            raise QuantizationOrderError("study identity does not match its cells")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "cells": [cell.to_record() for cell in self.cells],
        }

    def compare(self, key: QuantizationOrderStudyKey) -> QuantizationOrderComparison:
        arms = {cell.arm: cell for cell in self.cells if cell.key == key}
        if set(arms) != set(QuantizationOrderArm):
            return QuantizationOrderComparison(
                key,
                QuantizationOrderOutcome.INCONCLUSIVE,
                (),
                (),
                (),
                None,
                "matched four-arm cell is incomplete",
            )
        if any(cell.outcome is not QuantizationOrderOutcome.MEASURED for cell in arms.values()):
            reason = "; ".join(
                f"{cell.arm.value}: {cell.outcome.value} ({cell.outcome_reason or 'no reason'})"
                for cell in arms.values()
                if cell.outcome is not QuantizationOrderOutcome.MEASURED
            )
            return QuantizationOrderComparison(
                key, QuantizationOrderOutcome.UNSUPPORTED, (), (), (), None, reason
            )
        values = {
            cell.arm: {metric.name: metric.value for metric in cell.metrics}
            for cell in arms.values()
        }
        metrics = tuple(sorted(_METRIC_UNITS))

        def deltas(
            base: QuantizationOrderArm, comparison: QuantizationOrderArm
        ) -> tuple[QuantizationOrderDelta, ...]:
            return tuple(
                QuantizationOrderDelta(
                    metric,
                    _value(values[base][metric]),
                    _value(values[comparison][metric]),
                    _value(values[comparison][metric]) - _value(values[base][metric]),
                )
                for metric in metrics
            )

        quantization = deltas(
            QuantizationOrderArm.HIGH_PRECISION_SURGERY, QuantizationOrderArm.SURGERY_THEN_QUANTIZE
        )
        surgery = deltas(
            QuantizationOrderArm.QUANTIZATION_ONLY, QuantizationOrderArm.QUANTIZE_THEN_SURGERY
        )
        interaction = deltas(
            QuantizationOrderArm.SURGERY_THEN_QUANTIZE, QuantizationOrderArm.QUANTIZE_THEN_SURGERY
        )
        quality = {arm: _value(values[arm]["quality"]) for arm in arms}
        preferred = max(quality, key=lambda arm: quality[arm])
        return QuantizationOrderComparison(
            key, QuantizationOrderOutcome.MEASURED, quantization, surgery, interaction, preferred
        )


def build_quantization_order_study(
    cells: tuple[QuantizationOrderCell, ...],
) -> QuantizationOrderStudy:
    """Build a content-addressed matched-arm study."""

    if not cells:
        raise QuantizationOrderError("quantization-order study requires cells")
    study_id = (
        "quant_order_study_"
        + hashlib.sha256(_canonical([cell.to_record() for cell in cells]).encode()).hexdigest()
    )
    return QuantizationOrderStudy(cells, study_id)


def build_default_quantization_order_protocol() -> dict[str, object]:
    """Return the preregistered minimum matrix without claiming measurements."""

    return {
        "schema_version": QUANTIZATION_ORDER_SCHEMA_VERSION,
        "families": ["llama", "qwen"],
        "compression_levels": ["10_percent", "20_percent"],
        "codecs": ["Q8_0", "Q6_K", "Q4_K_M"],
        "seeds": [0, 1, 2],
        "arms": [arm.value for arm in QuantizationOrderArm],
        "metrics": [{"name": name, "unit": unit} for name, unit in _METRIC_UNITS.items()],
        "decision_rule": (
            "Prefer an arm only when all four matched cells are measured and quality is highest; "
            "otherwise retain unsupported or inconclusive."
        ),
    }


def render_quantization_order_protocol() -> str:
    """Render the preregistered order-study protocol as JSON."""

    return _canonical(build_default_quantization_order_protocol()) + "\n"
