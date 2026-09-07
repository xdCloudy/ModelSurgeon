"""Deterministic leave-one-out, zero-target, and few-shot transfer manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

TRANSFER_SUITE_SCHEMA_VERSION: Final[int] = 1
TRANSFER_SUITE_PROTOCOL_REVISION: Final[str] = "transfer-suite-v1"


class TransferSuiteError(ValueError):
    """Raised when transfer folds or evidence violate the held-out contract."""


class TransferFoldKind(StrEnum):
    LEAVE_ONE_CHECKPOINT = "leave_one_checkpoint"
    LEAVE_ONE_SIZE = "leave_one_size"
    LEAVE_ONE_FAMILY = "leave_one_family"
    ZERO_TARGET = "zero_target"
    FEW_SHOT = "few_shot"


class TransferResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransferSuiteError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TransferSample:
    sample_id: str
    checkpoint_id: str
    model_revision: str
    family: str
    size: str
    lineage_group_id: str
    target_available: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("sample ID", self.sample_id),
            ("checkpoint ID", self.checkpoint_id),
            ("model revision", self.model_revision),
            ("family", self.family),
            ("size", self.size),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)

    @property
    def checkpoint_key(self) -> str:
        return f"{self.checkpoint_id}@{self.model_revision}"

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "checkpoint_id": self.checkpoint_id,
            "model_revision": self.model_revision,
            "family": self.family,
            "size": self.size,
            "lineage_group_id": self.lineage_group_id,
            "target_available": self.target_available,
        }


@dataclass(frozen=True, slots=True)
class TransferSuiteConfig:
    few_shot_examples: int = 2
    minimum_folds: int = 3
    seed: int = 0
    bootstrap_repetitions: int = 200
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.few_shot_examples < 1 or self.minimum_folds < 3 or self.bootstrap_repetitions <= 0:
            raise TransferSuiteError("transfer suite bounds are invalid")
        if not 0.0 < self.confidence < 1.0:
            raise TransferSuiteError("transfer suite confidence must be within (0, 1)")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise TransferSuiteError("transfer suite seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "few_shot_examples": self.few_shot_examples,
            "minimum_folds": self.minimum_folds,
            "seed": self.seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class TransferFold:
    fold_id: str
    kind: TransferFoldKind
    heldout_key: str
    source_sample_ids: tuple[str, ...]
    target_adaptation_ids: tuple[str, ...]
    target_test_ids: tuple[str, ...]
    source_lineage_groups: tuple[str, ...]
    target_lineage_groups: tuple[str, ...]
    protocol_revision: str = TRANSFER_SUITE_PROTOCOL_REVISION

    def __post_init__(self) -> None:
        if not self.fold_id or not self.heldout_key:
            raise TransferSuiteError("transfer folds require identity and held-out key")
        if not self.target_test_ids:
            raise TransferSuiteError("transfer folds require target test samples")
        if set(self.source_sample_ids) & set(self.target_adaptation_ids + self.target_test_ids):
            raise TransferSuiteError("transfer fold source and target samples overlap")
        if set(self.source_lineage_groups) & set(self.target_lineage_groups):
            raise TransferSuiteError("transfer fold source and target ancestry groups overlap")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TRANSFER_SUITE_SCHEMA_VERSION,
            "protocol_revision": self.protocol_revision,
            "fold_id": self.fold_id,
            "kind": self.kind.value,
            "heldout_key": self.heldout_key,
            "source_sample_ids": list(self.source_sample_ids),
            "target_adaptation_ids": list(self.target_adaptation_ids),
            "target_test_ids": list(self.target_test_ids),
            "source_lineage_groups": list(self.source_lineage_groups),
            "target_lineage_groups": list(self.target_lineage_groups),
        }


@dataclass(frozen=True, slots=True)
class TransferMetric:
    name: str
    value: float | None
    interval_low: float | None
    interval_high: float | None
    status: str = "measured"

    def __post_init__(self) -> None:
        _text(self.name, "transfer metric name")
        _text(self.status, "transfer metric status")
        values = (self.value, self.interval_low, self.interval_high)
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            )
            for value in values
        ):
            raise TransferSuiteError("transfer metric values must be finite")
        if (
            self.interval_low is not None
            and self.interval_high is not None
            and self.interval_low > self.interval_high
        ):
            raise TransferSuiteError("transfer metric interval is reversed")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TransferFoldResult:
    fold: TransferFold
    status: TransferResultStatus
    metrics: tuple[TransferMetric, ...]
    evaluations: int
    failures: tuple[str, ...] = ()
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.evaluations < 0:
            raise TransferSuiteError("transfer evaluation count cannot be negative")
        required = {"ranking", "calibration", "numeric_error", "frontier_quality"}
        names = {metric.name for metric in self.metrics}
        if names != required:
            raise TransferSuiteError(
                "every transfer fold must report a complete "
                "ranking/calibration/error/frontier metric set"
            )
        if self.status is TransferResultStatus.SUCCEEDED and self.failures:
            raise TransferSuiteError("successful transfer folds cannot carry failures")
        if self.status is not TransferResultStatus.SUCCEEDED and not self.failures:
            raise TransferSuiteError("failed or unsupported folds require failure evidence")

    def to_record(self) -> dict[str, object]:
        return {
            "fold": self.fold.to_record(),
            "status": self.status.value,
            "metrics": [item.to_record() for item in self.metrics],
            "evaluations": self.evaluations,
            "failures": list(self.failures),
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True, slots=True)
class TransferSuiteReport:
    config: TransferSuiteConfig
    folds: tuple[TransferFold, ...]
    results: tuple[TransferFoldResult, ...] = ()
    schema_version: int = TRANSFER_SUITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.folds) < self.config.minimum_folds:
            raise TransferSuiteError("transfer suite does not contain the minimum fold count")
        if {fold.kind for fold in self.folds} != set(TransferFoldKind):
            raise TransferSuiteError("transfer suite requires all five declared protocols")
        fold_ids = {item.fold_id for item in self.folds}
        result_ids = tuple(item.fold.fold_id for item in self.results)
        if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= fold_ids:
            raise TransferSuiteError("transfer results must reference each fold at most once")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": TRANSFER_SUITE_PROTOCOL_REVISION,
            "config": self.config.to_record(),
            "folds": [item.to_record() for item in self.folds],
            "results": [item.to_record() for item in self.results],
        }


def build_transfer_suite(
    samples: Sequence[TransferSample], config: TransferSuiteConfig | None = None
) -> TransferSuiteReport:
    """Build deterministic source/target folds without target fitting leakage."""

    resolved = config or TransferSuiteConfig()
    if not samples:
        raise TransferSuiteError("transfer suite requires samples")
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise TransferSuiteError("transfer sample IDs must be unique")
    folds: list[TransferFold] = []

    def add_fold(
        kind: TransferFoldKind,
        heldout_key: str,
        target: tuple[TransferSample, ...],
        adaptation: int = 0,
    ) -> None:
        source = tuple(
            sample for sample in samples if sample not in target and sample.target_available
        )
        target_groups = {sample.lineage_group_id for sample in target}
        source_groups = {sample.lineage_group_id for sample in source}
        if not source or not target or source_groups & target_groups:
            return
        ordered_target = tuple(
            sorted(target, key=lambda item: _stable_rank(resolved.seed, item.sample_id))
        )
        adaptation_ids = tuple(item.sample_id for item in ordered_target[:adaptation])
        test_ids = tuple(item.sample_id for item in ordered_target[adaptation:])
        if not test_ids:
            return
        source_ids = tuple(sorted(item.sample_id for item in source))
        identity = {
            "kind": kind.value,
            "heldout_key": heldout_key,
            "source": source_ids,
            "adaptation": adaptation_ids,
            "test": test_ids,
        }
        folds.append(
            TransferFold(
                _digest(identity),
                kind,
                heldout_key,
                source_ids,
                adaptation_ids,
                test_ids,
                tuple(sorted(source_groups)),
                tuple(sorted(target_groups)),
            )
        )

    for kind, keys in (
        (
            TransferFoldKind.LEAVE_ONE_CHECKPOINT,
            sorted({sample.checkpoint_key for sample in samples}),
        ),
        (TransferFoldKind.LEAVE_ONE_SIZE, sorted({sample.size for sample in samples})),
        (TransferFoldKind.LEAVE_ONE_FAMILY, sorted({sample.family for sample in samples})),
    ):
        for key in keys:
            if kind is TransferFoldKind.LEAVE_ONE_CHECKPOINT:
                target = tuple(sample for sample in samples if sample.checkpoint_key == key)
            elif kind is TransferFoldKind.LEAVE_ONE_SIZE:
                target = tuple(sample for sample in samples if sample.size == key)
            else:
                target = tuple(sample for sample in samples if sample.family == key)
            add_fold(kind, key, target)
    for checkpoint_key in sorted({sample.checkpoint_key for sample in samples}):
        target = tuple(sample for sample in samples if sample.checkpoint_key == checkpoint_key)
        add_fold(TransferFoldKind.ZERO_TARGET, checkpoint_key, target)
        add_fold(TransferFoldKind.FEW_SHOT, checkpoint_key, target, resolved.few_shot_examples)
    folds.sort(key=lambda item: (item.kind.value, item.heldout_key, item.fold_id))
    return TransferSuiteReport(resolved, tuple(folds))


def record_transfer_fold_result(
    report: TransferSuiteReport,
    fold_id: str,
    *,
    status: TransferResultStatus,
    metrics: Mapping[str, tuple[float | None, float | None, float | None, str]],
    evaluations: int,
    failures: Sequence[str] = (),
    provenance: Mapping[str, object] | None = None,
) -> TransferSuiteReport:
    """Attach one result while requiring the complete per-fold metric surface."""

    fold = next((item for item in report.folds if item.fold_id == fold_id), None)
    if fold is None:
        raise TransferSuiteError("transfer result references an unknown fold")
    result = TransferFoldResult(
        fold,
        status,
        tuple(
            TransferMetric(name, values[0], values[1], values[2], values[3])
            for name, values in sorted(metrics.items())
        ),
        evaluations,
        tuple(failures),
        {} if provenance is None else dict(provenance),
    )
    existing = {item.fold.fold_id: item for item in report.results}
    if fold_id in existing:
        raise TransferSuiteError("transfer fold result already recorded")
    existing[fold_id] = result
    ordered = tuple(existing[item.fold_id] for item in report.folds if item.fold_id in existing)
    return TransferSuiteReport(report.config, report.folds, ordered)


def _stable_rank(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest(), "big")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
