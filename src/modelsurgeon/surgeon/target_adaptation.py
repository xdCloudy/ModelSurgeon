"""Bounded zero-shot and few-shot target adaptation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

from .compatibility import CompatibilityDecision

TARGET_ADAPTATION_SCHEMA_VERSION: Final[int] = 1
TARGET_ADAPTATION_PROTOCOL_REVISION: Final[str] = "target-adaptation-v1"
TARGET_EXAMPLE_BUDGETS: Final[tuple[int, ...]] = (0, 8, 16, 32, 64, 128)


class TargetAdaptationError(ValueError):
    """Raised when target adaptation would violate its leakage or budget contract."""


class TargetAdaptationMode(StrEnum):
    ZERO_SHOT = "zero_shot"
    SHIFT_CORRECTION = "shift_correction"
    FROZEN_HEAD = "frozen_head"
    LINEAR_ADAPTER = "linear_adapter"
    FULL_BOUNDED = "full_bounded"


class AdaptationOutcomeStatus(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AdaptationTransition(StrEnum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    INTERRUPTED = "interrupted"


class TargetExamplePartition(StrEnum):
    SUPPORT_POOL = "support_pool"
    TEST = "test"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetAdaptationError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetAdaptationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TargetAdaptationError(f"{label} must be finite")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TargetAdaptationConfig:
    budgets: tuple[int, ...] = TARGET_EXAMPLE_BUDGETS
    selection_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_epochs: int = 200
    early_stopping_patience: int = 10
    max_training_seconds: float = 300.0
    max_evaluation_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.budgets != tuple(sorted(set(self.budgets))) or not self.budgets:
            raise TargetAdaptationError("adaptation budgets must be sorted and unique")
        if self.budgets[0] != 0 or any(budget < 0 for budget in self.budgets):
            raise TargetAdaptationError("adaptation budgets must start at zero")
        if len(self.selection_seeds) < 5 or len(set(self.selection_seeds)) != len(
            self.selection_seeds
        ):
            raise TargetAdaptationError("adaptation requires five unique selection seeds")
        if any(isinstance(seed, bool) or not 0 <= seed < 1 << 64 for seed in self.selection_seeds):
            raise TargetAdaptationError("adaptation seeds must be unsigned 64-bit integers")
        if self.max_epochs <= 0 or self.early_stopping_patience <= 0:
            raise TargetAdaptationError("adaptation training bounds must be positive")
        if self.max_training_seconds <= 0 or self.max_evaluation_seconds <= 0:
            raise TargetAdaptationError("adaptation time budgets must be positive")
        _finite(self.max_training_seconds, "training time budget")
        _finite(self.max_evaluation_seconds, "evaluation time budget")

    def to_record(self) -> dict[str, object]:
        return {
            "budgets": list(self.budgets),
            "selection_seeds": list(self.selection_seeds),
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "max_training_seconds": self.max_training_seconds,
            "max_evaluation_seconds": self.max_evaluation_seconds,
        }


@dataclass(frozen=True, slots=True)
class TargetMutationExample:
    example_id: str
    target_model_id: str
    partition: TargetExamplePartition
    feature_digest: str
    label: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("target example ID", self.example_id),
            ("target model ID", self.target_model_id),
            ("feature digest", self.feature_digest),
        ):
            _text(value, label)
        if self.partition is TargetExamplePartition.SUPPORT_POOL:
            if self.label is None:
                raise TargetAdaptationError("support examples require target labels")
            _finite(self.label, "support target label")
        elif self.label is not None:
            raise TargetAdaptationError("target test examples cannot carry test outcomes")

    def to_record(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "target_model_id": self.target_model_id,
            "partition": self.partition.value,
            "feature_digest": self.feature_digest,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class TargetAdaptationRequest:
    request_id: str
    source_bundle_digest: str
    target_model_id: str
    compatibility: CompatibilityDecision
    mode: TargetAdaptationMode
    target_example_budget: int
    selection_seed: int
    support_example_ids: tuple[str, ...]
    test_example_ids: tuple[str, ...]
    target_statistics_digest: str | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, "adaptation request ID")
        _text(self.source_bundle_digest, "source bundle digest")
        _text(self.target_model_id, "target model ID")
        if self.compatibility.target_model_id != self.target_model_id:
            raise TargetAdaptationError("compatibility decision does not match target model")
        if not self.compatibility.allowed:
            raise TargetAdaptationError("adaptation requires an allowed compatibility decision")
        if self.target_example_budget < 0:
            raise TargetAdaptationError("target example budget cannot be negative")
        if not 0 <= self.selection_seed < 1 << 64:
            raise TargetAdaptationError("selection seed must be unsigned 64-bit")
        if len(self.support_example_ids) != len(set(self.support_example_ids)):
            raise TargetAdaptationError("support example IDs must be unique")
        if len(self.test_example_ids) != len(set(self.test_example_ids)):
            raise TargetAdaptationError("test example IDs must be unique")
        if set(self.support_example_ids) & set(self.test_example_ids):
            raise TargetAdaptationError("support and test examples must be disjoint")
        if self.target_example_budget == 0:
            if self.mode is not TargetAdaptationMode.ZERO_SHOT:
                raise TargetAdaptationError("zero target budget only supports zero-shot mode")
            if self.support_example_ids or self.target_statistics_digest is not None:
                raise TargetAdaptationError("zero-shot adaptation cannot consume target evidence")
        elif self.mode is TargetAdaptationMode.ZERO_SHOT:
            raise TargetAdaptationError("zero-shot adaptation requires a zero target budget")
        elif len(self.support_example_ids) != self.target_example_budget:
            raise TargetAdaptationError("support set does not match target example budget")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TARGET_ADAPTATION_SCHEMA_VERSION,
            "protocol_revision": TARGET_ADAPTATION_PROTOCOL_REVISION,
            "request_id": self.request_id,
            "source_bundle_digest": self.source_bundle_digest,
            "target_model_id": self.target_model_id,
            "compatibility": self.compatibility.to_record(),
            "mode": self.mode.value,
            "target_example_budget": self.target_example_budget,
            "selection_seed": self.selection_seed,
            "support_example_ids": list(self.support_example_ids),
            "test_example_ids": list(self.test_example_ids),
            "target_statistics_digest": self.target_statistics_digest,
        }


@dataclass(frozen=True, slots=True)
class AdaptationLearningPoint:
    target_example_budget: int
    frontier_quality: float
    evaluation_savings: float
    training_seconds: float
    evaluation_seconds: float

    def __post_init__(self) -> None:
        if self.target_example_budget < 0:
            raise TargetAdaptationError("learning-curve budget cannot be negative")
        for label, value in (
            ("frontier quality", self.frontier_quality),
            ("evaluation savings", self.evaluation_savings),
            ("training seconds", self.training_seconds),
            ("evaluation seconds", self.evaluation_seconds),
        ):
            _finite(value, label)
        if self.training_seconds < 0 or self.evaluation_seconds < 0:
            raise TargetAdaptationError("learning-curve costs cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "target_example_budget": self.target_example_budget,
            "frontier_quality": self.frontier_quality,
            "evaluation_savings": self.evaluation_savings,
            "training_seconds": self.training_seconds,
            "evaluation_seconds": self.evaluation_seconds,
        }


@dataclass(frozen=True, slots=True)
class TargetAdaptationResult:
    request: TargetAdaptationRequest
    status: AdaptationOutcomeStatus
    transition: AdaptationTransition
    child_bundle_digest: str | None = None
    learning_curve: tuple[AdaptationLearningPoint, ...] = ()
    failures: tuple[str, ...] = ()
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        budgets = tuple(point.target_example_budget for point in self.learning_curve)
        if budgets != tuple(sorted(set(budgets))):
            raise TargetAdaptationError("learning-curve budgets must be sorted and unique")
        if self.status is AdaptationOutcomeStatus.MEASURED:
            if self.child_bundle_digest is None or (
                self.transition is not AdaptationTransition.COMMITTED
            ):
                raise TargetAdaptationError("measured adaptation requires a committed child bundle")
            if not self.learning_curve:
                raise TargetAdaptationError("measured adaptation requires a learning curve")
            if self.failures:
                raise TargetAdaptationError("measured adaptation cannot carry failures")
        elif self.status is not AdaptationOutcomeStatus.UNKNOWN and not self.failures:
            raise TargetAdaptationError("non-measured adaptation requires failure evidence")
        if self.transition is AdaptationTransition.ROLLED_BACK and self.child_bundle_digest is None:
            raise TargetAdaptationError("rollback must retain the child bundle digest")

    def to_record(self) -> dict[str, object]:
        return {
            "request": self.request.to_record(),
            "status": self.status.value,
            "transition": self.transition.value,
            "child_bundle_digest": self.child_bundle_digest,
            "learning_curve": [point.to_record() for point in self.learning_curve],
            "failures": list(self.failures),
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True, slots=True)
class TargetAdaptationStudy:
    config: TargetAdaptationConfig
    requests: tuple[TargetAdaptationRequest, ...]
    results: tuple[TargetAdaptationResult, ...] = ()
    schema_version: int = TARGET_ADAPTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_ADAPTATION_SCHEMA_VERSION:
            raise TargetAdaptationError("unsupported target adaptation schema")
        request_ids = tuple(request.request_id for request in self.requests)
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise TargetAdaptationError("adaptation requests must be non-empty and unique")
        result_ids = tuple(result.request.request_id for result in self.results)
        if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= set(request_ids):
            raise TargetAdaptationError(
                "adaptation results must reference each request at most once"
            )

    @property
    def complete(self) -> bool:
        return len(self.results) == len(self.requests)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": TARGET_ADAPTATION_PROTOCOL_REVISION,
            "config": self.config.to_record(),
            "requests": [request.to_record() for request in self.requests],
            "results": [result.to_record() for result in self.results],
            "complete": self.complete,
        }


def select_target_support(
    examples: Sequence[TargetMutationExample], budget: int, seed: int
) -> tuple[str, ...]:
    """Select only labelled support-pool examples using a stable seed."""

    if budget < 0:
        raise TargetAdaptationError("support budget cannot be negative")
    pool = tuple(
        example for example in examples if example.partition is TargetExamplePartition.SUPPORT_POOL
    )
    if budget > len(pool):
        raise TargetAdaptationError("target support pool is smaller than requested budget")
    ordered = sorted(pool, key=lambda item: _stable_rank(seed, item.example_id))
    return tuple(item.example_id for item in ordered[:budget])


def plan_target_adaptation(
    source_bundle_digest: str,
    target_model_id: str,
    compatibility: CompatibilityDecision,
    examples: Sequence[TargetMutationExample],
    *,
    config: TargetAdaptationConfig | None = None,
) -> TargetAdaptationStudy:
    """Create zero/few-shot requests without exposing target test outcomes."""

    resolved = config or TargetAdaptationConfig()
    _text(source_bundle_digest, "source bundle digest")
    _text(target_model_id, "target model ID")
    if compatibility.target_model_id != target_model_id or not compatibility.allowed:
        raise TargetAdaptationError(
            "target adaptation requires an allowed target compatibility decision"
        )
    ids = tuple(example.example_id for example in examples)
    if len(ids) != len(set(ids)):
        raise TargetAdaptationError("target adaptation example IDs must be unique")
    if any(example.target_model_id != target_model_id for example in examples):
        raise TargetAdaptationError("target adaptation examples must belong to the target model")
    test_ids = tuple(
        sorted(
            example.example_id
            for example in examples
            if example.partition is TargetExamplePartition.TEST
        )
    )
    requests: list[TargetAdaptationRequest] = []
    modes = (
        TargetAdaptationMode.ZERO_SHOT,
        TargetAdaptationMode.SHIFT_CORRECTION,
        TargetAdaptationMode.FROZEN_HEAD,
        TargetAdaptationMode.LINEAR_ADAPTER,
        TargetAdaptationMode.FULL_BOUNDED,
    )
    for seed in resolved.selection_seeds:
        for budget in resolved.budgets:
            selected = select_target_support(examples, budget, seed)
            selected_modes = (TargetAdaptationMode.ZERO_SHOT,) if budget == 0 else modes[1:]
            for mode in selected_modes:
                identity = {
                    "source_bundle_digest": source_bundle_digest,
                    "target_model_id": target_model_id,
                    "compatibility": compatibility.decision_id,
                    "mode": mode.value,
                    "budget": budget,
                    "seed": seed,
                    "support": selected,
                    "test": test_ids,
                }
                requests.append(
                    TargetAdaptationRequest(
                        _digest(identity),
                        source_bundle_digest,
                        target_model_id,
                        compatibility,
                        mode,
                        budget,
                        seed,
                        selected,
                        test_ids,
                    )
                )
    requests.sort(key=lambda item: item.request_id)
    return TargetAdaptationStudy(resolved, tuple(requests))


def record_target_adaptation_result(
    study: TargetAdaptationStudy,
    request_id: str,
    *,
    status: AdaptationOutcomeStatus,
    transition: AdaptationTransition,
    child_bundle_digest: str | None = None,
    learning_curve: Sequence[AdaptationLearningPoint] = (),
    failures: Sequence[str] = (),
    provenance: Mapping[str, object] | None = None,
) -> TargetAdaptationStudy:
    """Attach one resumable outcome while preserving parent/child lineage."""

    request = next((item for item in study.requests if item.request_id == request_id), None)
    if request is None:
        raise TargetAdaptationError("adaptation result references an unknown request")
    if any(item.request.request_id == request_id for item in study.results):
        raise TargetAdaptationError("adaptation result already recorded")
    result = TargetAdaptationResult(
        request,
        status,
        transition,
        child_bundle_digest,
        tuple(learning_curve),
        tuple(failures),
        {} if provenance is None else dict(provenance),
    )
    return replace(study, results=(*study.results, result))


def _stable_rank(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest(), "big")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()
