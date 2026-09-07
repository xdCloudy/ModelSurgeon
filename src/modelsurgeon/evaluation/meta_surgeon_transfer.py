"""Versioned evidence contract for the dense four-family meta-surgeon study."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

META_SURGEON_TRANSFER_SCHEMA_VERSION: Final[int] = 1
META_SURGEON_TRANSFER_PROTOCOL_REVISION: Final[str] = "meta-surgeon-transfer-v1"
SUPPORTED_TRANSFER_FAMILIES: Final[tuple[str, ...]] = ("gemma", "llama", "mistral", "qwen")
REQUIRED_TRANSFER_METRICS: Final[frozenset[str]] = frozenset(
    {
        "ranking",
        "calibration",
        "regression",
        "cumulative_frontier",
        "evaluation_savings",
    }
)


class MetaSurgeonTransferError(ValueError):
    """Raised when a transfer matrix or result violates the study contract."""


class TransferAxis(StrEnum):
    CHECKPOINT = "checkpoint"
    SIZE = "size"
    FAMILY = "family"
    CORPUS = "corpus"


class TransferControl(StrEnum):
    COLD_START = "cold_start"
    WITHIN_TARGET = "within_target"
    META = "meta"


class TransferFeatureView(StrEnum):
    RAW = "raw"
    ARCHITECTURE_NORMALIZED = "architecture_normalized"


class TransferOutcomeStatus(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetaSurgeonTransferError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetaSurgeonTransferError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MetaSurgeonTransferError(f"{label} must be finite")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TransferStudyConfig:
    """Equal budgets and minimum evidence for every matrix cell."""

    families: tuple[str, ...] = SUPPORTED_TRANSFER_FAMILIES
    minimum_sizes: int = 2
    minimum_source_examples: int = 512
    seeds: tuple[int, ...] = (0, 1, 2)
    bootstrap_repetitions: int = 1_000
    confidence: float = 0.95
    max_source_family_combinations: int = 7

    def __post_init__(self) -> None:
        if not self.families or len(self.families) != len(set(self.families)):
            raise MetaSurgeonTransferError("transfer families must be unique and non-empty")
        if any(not family.strip() for family in self.families):
            raise MetaSurgeonTransferError("transfer family names cannot be blank")
        if self.minimum_sizes < 1 or self.minimum_source_examples < 1:
            raise MetaSurgeonTransferError("transfer minimums must be positive")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise MetaSurgeonTransferError("transfer studies require three unique seeds")
        if any(isinstance(seed, bool) or not 0 <= seed < 1 << 64 for seed in self.seeds):
            raise MetaSurgeonTransferError("transfer seeds must be unsigned 64-bit integers")
        if self.bootstrap_repetitions < 100:
            raise MetaSurgeonTransferError(
                "transfer intervals require at least 100 bootstrap repetitions"
            )
        if not 0.0 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise MetaSurgeonTransferError("transfer confidence must be within (0, 1)")
        if self.max_source_family_combinations < 1:
            raise MetaSurgeonTransferError("transfer source-family bound must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "families": list(self.families),
            "minimum_sizes": self.minimum_sizes,
            "minimum_source_examples": self.minimum_source_examples,
            "seeds": list(self.seeds),
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
            "max_source_family_combinations": self.max_source_family_combinations,
        }


@dataclass(frozen=True, slots=True)
class TransferStudyModel:
    model_id: str
    checkpoint_id: str
    model_revision: str
    family: str
    size: str
    corpus_revision: str
    hardware_profile: str
    source_example_count: int | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("model ID", self.model_id),
            ("checkpoint ID", self.checkpoint_id),
            ("model revision", self.model_revision),
            ("model family", self.family),
            ("model size", self.size),
            ("corpus revision", self.corpus_revision),
            ("hardware profile", self.hardware_profile),
        ):
            _text(value, label)
        if self.source_example_count is not None and (
            isinstance(self.source_example_count, bool) or self.source_example_count < 0
        ):
            raise MetaSurgeonTransferError("source example count must be non-negative when present")

    def to_record(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "model_revision": self.model_revision,
            "family": self.family,
            "size": self.size,
            "corpus_revision": self.corpus_revision,
            "hardware_profile": self.hardware_profile,
            "source_example_count": self.source_example_count,
        }


@dataclass(frozen=True, slots=True)
class TransferStudyMetric:
    name: str
    value: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    bootstrap_repetitions: int = 0
    unit: str = "unitless"
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "transfer metric name")
        _text(self.unit, "transfer metric unit")
        if self.value is None:
            _text(self.reason or "", "unavailable transfer metric reason")
            if self.confidence_low is not None or self.confidence_high is not None:
                raise MetaSurgeonTransferError("unavailable metrics cannot carry intervals")
            return
        _finite(self.value, "transfer metric value")
        if self.confidence_low is None or self.confidence_high is None:
            raise MetaSurgeonTransferError("measured transfer metrics require confidence bounds")
        low = _finite(self.confidence_low, "transfer confidence low")
        high = _finite(self.confidence_high, "transfer confidence high")
        value = float(self.value)
        if not low <= value <= high:
            raise MetaSurgeonTransferError("transfer confidence bounds must contain the value")
        if self.bootstrap_repetitions < 100:
            raise MetaSurgeonTransferError(
                "measured transfer metrics require 100 bootstrap repetitions"
            )
        if self.reason is not None:
            raise MetaSurgeonTransferError("measured transfer metrics cannot carry a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "unit": self.unit,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TransferStudyCell:
    cell_id: str
    axis: TransferAxis
    control: TransferControl
    feature_view: TransferFeatureView
    target_model_id: str
    source_model_ids: tuple[str, ...]
    target_family: str
    target_size: str
    status: TransferOutcomeStatus = TransferOutcomeStatus.UNKNOWN
    metrics: tuple[TransferStudyMetric, ...] = ()
    evaluations: int = 0
    failures: tuple[str, ...] = ()
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("cell ID", self.cell_id),
            ("target model ID", self.target_model_id),
            ("target family", self.target_family),
            ("target size", self.target_size),
        ):
            _text(value, label)
        if len(self.source_model_ids) != len(set(self.source_model_ids)):
            raise MetaSurgeonTransferError("source model IDs must be unique")
        if self.target_model_id in self.source_model_ids:
            raise MetaSurgeonTransferError("target model cannot enter source fitting")
        if self.evaluations < 0:
            raise MetaSurgeonTransferError("transfer evaluations cannot be negative")
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise MetaSurgeonTransferError("transfer metric names must be unique")
        if self.status in {
            TransferOutcomeStatus.MEASURED,
            TransferOutcomeStatus.NEGATIVE_RESULT,
        }:
            if set(names) != REQUIRED_TRANSFER_METRICS or any(
                metric.value is None for metric in self.metrics
            ):
                raise MetaSurgeonTransferError("measured transfer cells require all five metrics")
        elif self.status is not TransferOutcomeStatus.UNKNOWN and not self.failures:
            raise MetaSurgeonTransferError("unsupported or failed transfer cells require evidence")

    @property
    def is_complete(self) -> bool:
        return self.status is not TransferOutcomeStatus.UNKNOWN

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": META_SURGEON_TRANSFER_SCHEMA_VERSION,
            "protocol_revision": META_SURGEON_TRANSFER_PROTOCOL_REVISION,
            "cell_id": self.cell_id,
            "axis": self.axis.value,
            "control": self.control.value,
            "feature_view": self.feature_view.value,
            "target_model_id": self.target_model_id,
            "source_model_ids": list(self.source_model_ids),
            "target_family": self.target_family,
            "target_size": self.target_size,
            "status": self.status.value,
            "metrics": [metric.to_record() for metric in self.metrics],
            "evaluations": self.evaluations,
            "failures": list(self.failures),
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True, slots=True)
class MetaSurgeonTransferReport:
    config: TransferStudyConfig
    models: tuple[TransferStudyModel, ...]
    cells: tuple[TransferStudyCell, ...]
    schema_version: int = META_SURGEON_TRANSFER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != META_SURGEON_TRANSFER_SCHEMA_VERSION:
            raise MetaSurgeonTransferError("unsupported meta-surgeon transfer schema")
        model_ids = tuple(model.model_id for model in self.models)
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise MetaSurgeonTransferError("transfer models must be non-empty and unique")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if not cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise MetaSurgeonTransferError("transfer cells must be non-empty and unique")
        known_models = set(model_ids)
        if any(
            cell.target_model_id not in known_models
            or any(source not in known_models for source in cell.source_model_ids)
            for cell in self.cells
        ):
            raise MetaSurgeonTransferError("transfer cell references an unknown model")
        if set(self.config.families) - {model.family for model in self.models}:
            raise MetaSurgeonTransferError("every configured family needs a target model")
        covered_families = {cell.target_family for cell in self.cells}
        if set(self.config.families) - covered_families:
            raise MetaSurgeonTransferError("every configured family needs transfer coverage")

    @property
    def complete(self) -> bool:
        return all(cell.is_complete for cell in self.cells)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": META_SURGEON_TRANSFER_PROTOCOL_REVISION,
            "config": self.config.to_record(),
            "models": [model.to_record() for model in self.models],
            "cells": [cell.to_record() for cell in self.cells],
            "complete": self.complete,
        }


def build_meta_surgeon_transfer_matrix(
    models: Sequence[TransferStudyModel], config: TransferStudyConfig | None = None
) -> MetaSurgeonTransferReport:
    """Plan a dense, source-only four-family transfer matrix."""

    resolved = config or TransferStudyConfig()
    if not models:
        raise MetaSurgeonTransferError("transfer matrix requires models")
    model_ids = tuple(model.model_id for model in models)
    if len(model_ids) != len(set(model_ids)):
        raise MetaSurgeonTransferError("transfer model IDs must be unique")
    by_id = {model.model_id: model for model in models}
    missing = set(resolved.families) - {model.family for model in models}
    if missing:
        raise MetaSurgeonTransferError(
            "transfer matrix is missing configured families: " + ", ".join(sorted(missing))
        )
    undercovered = sorted(
        family
        for family in resolved.families
        if len({model.size for model in models if model.family == family})
        < resolved.minimum_sizes
    )
    if undercovered:
        raise MetaSurgeonTransferError(
            "transfer matrix needs more declared sizes for: " + ", ".join(undercovered)
        )
    family_models = {
        family: tuple(
            sorted(
                (model for model in models if model.family == family),
                key=lambda item: item.model_id,
            )
        )
        for family in resolved.families
    }
    cells: list[TransferStudyCell] = []

    def add_cell(
        axis: TransferAxis,
        control: TransferControl,
        feature_view: TransferFeatureView,
        target: TransferStudyModel,
        source_ids: tuple[str, ...],
    ) -> None:
        identity = {
            "axis": axis.value,
            "control": control.value,
            "feature_view": feature_view.value,
            "target": target.model_id,
            "source": source_ids,
        }
        cells.append(
            TransferStudyCell(
                _digest(identity),
                axis,
                control,
                feature_view,
                target.model_id,
                source_ids,
                target.family,
                target.size,
                provenance={
                    "target_model_revision": target.model_revision,
                    "target_corpus_revision": target.corpus_revision,
                    "target_hardware_profile": target.hardware_profile,
                    "source_example_counts": {
                        model_id: by_id[model_id].source_example_count for model_id in source_ids
                    },
                    "minimum_source_examples": resolved.minimum_source_examples,
                    "seeds": list(resolved.seeds),
                },
            )
        )

    for target in sorted(models, key=lambda item: item.model_id):
        for feature_view in TransferFeatureView:
            add_cell(TransferAxis.CHECKPOINT, TransferControl.COLD_START, feature_view, target, ())
            within = tuple(
                model.model_id
                for model in family_models[target.family]
                if model.model_id != target.model_id
            )
            if within:
                axis = (
                    TransferAxis.CORPUS
                    if any(by_id[item].corpus_revision != target.corpus_revision for item in within)
                    else TransferAxis.SIZE
                    if any(by_id[item].size != target.size for item in within)
                    else TransferAxis.CHECKPOINT
                )
                add_cell(axis, TransferControl.WITHIN_TARGET, feature_view, target, within)
            other_families = tuple(
                family for family in resolved.families if family != target.family
            )
            for count in range(1, len(other_families) + 1):
                for source_families in itertools.combinations(other_families, count):
                    if count > resolved.max_source_family_combinations:
                        continue
                    source_ids = tuple(
                        sorted(
                            model.model_id
                            for family in source_families
                            for model in family_models[family]
                        )
                    )
                    add_cell(
                        TransferAxis.FAMILY,
                        TransferControl.META,
                        feature_view,
                        target,
                        source_ids,
                    )
    cells.sort(key=lambda item: item.cell_id)
    return MetaSurgeonTransferReport(
        resolved,
        tuple(sorted(models, key=lambda item: item.model_id)),
        tuple(cells),
    )


def record_meta_surgeon_transfer_result(
    report: MetaSurgeonTransferReport,
    cell_id: str,
    *,
    status: TransferOutcomeStatus,
    metrics: Sequence[TransferStudyMetric],
    evaluations: int,
    failures: Sequence[str] = (),
    provenance: Mapping[str, object] | None = None,
) -> MetaSurgeonTransferReport:
    """Attach one measured, negative, unsupported, or failed cell outcome."""

    cell = next((item for item in report.cells if item.cell_id == cell_id), None)
    if cell is None:
        raise MetaSurgeonTransferError("transfer result references an unknown cell")
    if cell.is_complete:
        raise MetaSurgeonTransferError("transfer cell result already recorded")
    names = {metric.name for metric in metrics}
    if status in {TransferOutcomeStatus.MEASURED, TransferOutcomeStatus.NEGATIVE_RESULT}:
        if names != REQUIRED_TRANSFER_METRICS or any(metric.value is None for metric in metrics):
            raise MetaSurgeonTransferError("complete transfer results require all five metrics")
    elif not failures:
        raise MetaSurgeonTransferError("non-measured transfer results require failure evidence")
    updated = replace(
        cell,
        status=status,
        metrics=tuple(sorted(metrics, key=lambda item: item.name)),
        evaluations=evaluations,
        failures=tuple(failures),
        provenance={} if provenance is None else dict(provenance),
    )
    return replace(
        report,
        cells=tuple(updated if item.cell_id == cell_id else item for item in report.cells),
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()
