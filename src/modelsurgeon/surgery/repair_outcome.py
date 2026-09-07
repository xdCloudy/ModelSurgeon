"""Unified bounded repair outcomes, budgets, and immutable artifact lineage."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

REPAIR_OUTCOME_SCHEMA_VERSION: Final[int] = 1
REPAIR_OUTCOME_PROTOCOL_REVISION: Final[str] = "repair-outcome-v1"


class RepairOutcomeError(ValueError):
    """Raised when repair mechanics or effectiveness evidence cannot reconcile."""


class RepairMethod(StrEnum):
    NO_REPAIR = "no_repair"
    LORA = "lora"
    SELECTED_FINETUNE = "selected_finetune"
    FULL_FINETUNE = "full_finetune"
    LOGIT_DISTILLATION = "logit_distillation"
    FEATURE_DISTILLATION = "feature_distillation"


class RepairOutcomeStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ArtifactOutputMode(StrEnum):
    NONE = "none"
    SEPARATE_LORA = "separate_lora"
    MERGED_WEIGHTS = "merged_weights"
    FULL_CHECKPOINT = "full_checkpoint"


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepairOutcomeError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairOutcomeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RepairOutcomeError(f"{label} must be finite")
    return result


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RepairOutcomeError(f"{label} must be a lowercase SHA-256")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class RepairBudget:
    max_steps: int
    max_tokens: int
    max_training_seconds: float
    max_gpu_seconds: float
    max_energy_joules: float
    max_disk_bytes: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or value <= 0
            for value in (self.max_steps, self.max_tokens, self.max_disk_bytes)
        ):
            raise RepairOutcomeError("repair step, token, and disk budgets must be positive")
        for label, value in (
            ("training seconds", self.max_training_seconds),
            ("GPU seconds", self.max_gpu_seconds),
            ("energy joules", self.max_energy_joules),
        ):
            if _finite(value, label) <= 0:
                raise RepairOutcomeError(f"repair {label} budget must be positive")

    def enforce(self, cost: RepairCost) -> None:
        if (
            cost.steps > self.max_steps
            or cost.tokens > self.max_tokens
            or cost.training_seconds > self.max_training_seconds
            or cost.gpu_seconds > self.max_gpu_seconds
            or cost.energy_joules > self.max_energy_joules
            or cost.artifact_bytes_written > self.max_disk_bytes
        ):
            raise RepairOutcomeError("repair cost exceeds its declared budget")

    def to_record(self) -> dict[str, object]:
        return {
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "max_training_seconds": self.max_training_seconds,
            "max_gpu_seconds": self.max_gpu_seconds,
            "max_energy_joules": self.max_energy_joules,
            "max_disk_bytes": self.max_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class RepairCost:
    steps: int
    tokens: int
    training_seconds: float
    gpu_seconds: float
    energy_joules: float
    artifact_bytes_written: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or value < 0
            for value in (self.steps, self.tokens, self.artifact_bytes_written)
        ):
            raise RepairOutcomeError("repair cost counts cannot be negative")
        for label, value in (
            ("training seconds", self.training_seconds),
            ("GPU seconds", self.gpu_seconds),
            ("energy joules", self.energy_joules),
        ):
            if _finite(value, label) < 0:
                raise RepairOutcomeError("repair cost values cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "training_seconds": self.training_seconds,
            "gpu_seconds": self.gpu_seconds,
            "energy_joules": self.energy_joules,
            "artifact_bytes_written": self.artifact_bytes_written,
        }


@dataclass(frozen=True, slots=True)
class RepairModelIdentity:
    checkpoint_id: str
    revision: str
    tokenizer_digest: str
    data_revision: str

    def __post_init__(self) -> None:
        for label, value in (
            ("checkpoint ID", self.checkpoint_id),
            ("model revision", self.revision),
            ("tokenizer digest", self.tokenizer_digest),
            ("data revision", self.data_revision),
        ):
            _text(value, label)
        _digest(self.tokenizer_digest, "tokenizer digest")

    def to_record(self) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "revision": self.revision,
            "tokenizer_digest": self.tokenizer_digest,
            "data_revision": self.data_revision,
        }


@dataclass(frozen=True, slots=True)
class RepairRequest:
    request_id: str
    parent_candidate_id: str
    source: RepairModelIdentity
    candidate: RepairModelIdentity
    teacher: RepairModelIdentity | None
    method: RepairMethod
    output_mode: ArtifactOutputMode
    trainable_scope: tuple[str, ...]
    budget: RepairBudget
    compatibility: Mapping[str, object]
    best_state_metric: str
    best_state_direction: MetricDirection
    quantization_order: str

    def __post_init__(self) -> None:
        for label, value in (
            ("repair request ID", self.request_id),
            ("parent candidate ID", self.parent_candidate_id),
            ("best-state metric", self.best_state_metric),
            ("quantization order", self.quantization_order),
        ):
            _text(value, label)
        if self.source.checkpoint_id == self.candidate.checkpoint_id:
            raise RepairOutcomeError("repair source and candidate must be distinct")
        if not self.compatibility:
            raise RepairOutcomeError("repair requests require compatibility evidence")
        if self.trainable_scope != tuple(sorted(set(self.trainable_scope))) or any(
            not value.strip() for value in self.trainable_scope
        ):
            raise RepairOutcomeError("trainable scope must be unique and canonical")
        distillation = self.method in {
            RepairMethod.LOGIT_DISTILLATION,
            RepairMethod.FEATURE_DISTILLATION,
        }
        if distillation and self.teacher is None:
            raise RepairOutcomeError("distillation repairs require a teacher identity")
        if self.teacher is not None and (
            self.teacher.tokenizer_digest != self.candidate.tokenizer_digest
        ):
            raise RepairOutcomeError("teacher and candidate tokenizer identities do not match")
        if self.method is RepairMethod.NO_REPAIR and (
            self.output_mode is not ArtifactOutputMode.NONE
        ):
            raise RepairOutcomeError("no-repair cannot publish an output artifact")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": REPAIR_OUTCOME_SCHEMA_VERSION,
            "protocol_revision": REPAIR_OUTCOME_PROTOCOL_REVISION,
            "request_id": self.request_id,
            "parent_candidate_id": self.parent_candidate_id,
            "source": self.source.to_record(),
            "candidate": self.candidate.to_record(),
            "teacher": None if self.teacher is None else self.teacher.to_record(),
            "method": self.method.value,
            "output_mode": self.output_mode.value,
            "trainable_scope": list(self.trainable_scope),
            "budget": self.budget.to_record(),
            "compatibility": dict(self.compatibility),
            "best_state_metric": self.best_state_metric,
            "best_state_direction": self.best_state_direction.value,
            "quantization_order": self.quantization_order,
        }


@dataclass(frozen=True, slots=True)
class RepairArtifact:
    artifact_id: str
    digest: str
    size_bytes: int
    output_mode: ArtifactOutputMode
    parent_artifact_digest: str
    published: bool

    def __post_init__(self) -> None:
        _text(self.artifact_id, "repair artifact ID")
        _digest(self.digest, "repair artifact digest")
        _digest(self.parent_artifact_digest, "parent artifact digest")
        if self.size_bytes <= 0:
            raise RepairOutcomeError("repair artifact size must be positive")
        if self.output_mode is ArtifactOutputMode.NONE:
            raise RepairOutcomeError("published repair artifacts require an output mode")

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "output_mode": self.output_mode.value,
            "parent_artifact_digest": self.parent_artifact_digest,
            "published": self.published,
        }


@dataclass(frozen=True, slots=True)
class RepairMetricEvidence:
    name: str
    direction: MetricDirection
    baseline: float
    repaired: float
    minimum_improvement: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    heldout_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "repair metric name")
        for label, value in (
            ("baseline metric", self.baseline),
            ("repaired metric", self.repaired),
            ("minimum improvement", self.minimum_improvement),
        ):
            if _finite(value, label) < 0 and label == "minimum improvement":
                raise RepairOutcomeError("minimum repair improvement cannot be negative")
        if self.confidence_low is not None:
            _finite(self.confidence_low, "repair confidence low")
        if self.confidence_high is not None:
            _finite(self.confidence_high, "repair confidence high")
        if (
            self.confidence_low is not None
            and self.confidence_high is not None
            and self.confidence_low > self.confidence_high
        ):
            raise RepairOutcomeError("repair confidence interval is reversed")
        if not self.heldout_group_ids or len(self.heldout_group_ids) != len(
            set(self.heldout_group_ids)
        ):
            raise RepairOutcomeError("repair evidence requires unique held-out groups")

    @property
    def improvement(self) -> float:
        return (
            self.repaired - self.baseline
            if self.direction is MetricDirection.HIGHER
            else self.baseline - self.repaired
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "repaired": self.repaired,
            "improvement": self.improvement,
            "minimum_improvement": self.minimum_improvement,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "heldout_group_ids": list(self.heldout_group_ids),
        }


@dataclass(frozen=True, slots=True)
class RepairArtifactLineage:
    parent_artifact_digest: str
    source_artifact_digest: str
    candidate_artifact_digest: str
    teacher_artifact_digest: str | None
    data_revision: str
    tokenizer_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("parent artifact digest", self.parent_artifact_digest),
            ("source artifact digest", self.source_artifact_digest),
            ("candidate artifact digest", self.candidate_artifact_digest),
            ("tokenizer digest", self.tokenizer_digest),
        ):
            _digest(value, label)
        if self.teacher_artifact_digest is not None:
            _digest(self.teacher_artifact_digest, "teacher artifact digest")
        _text(self.data_revision, "repair data revision")

    def to_record(self) -> dict[str, object]:
        return {
            "parent_artifact_digest": self.parent_artifact_digest,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "teacher_artifact_digest": self.teacher_artifact_digest,
            "data_revision": self.data_revision,
            "tokenizer_digest": self.tokenizer_digest,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    request: RepairRequest
    status: RepairOutcomeStatus
    cost: RepairCost
    artifacts: tuple[RepairArtifact, ...]
    lineage: RepairArtifactLineage
    heldout_evidence: tuple[RepairMetricEvidence, ...]
    restored_parent_exactly: bool
    search_parent_eligible: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        self.request.budget.enforce(self.cost)
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise RepairOutcomeError("repair artifact IDs must be unique")
        published_bytes = sum(item.size_bytes for item in self.artifacts if item.published)
        if published_bytes != self.cost.artifact_bytes_written:
            raise RepairOutcomeError("repair artifact bytes do not reconcile with cost")
        if self.lineage.candidate_artifact_digest == self.lineage.parent_artifact_digest:
            raise RepairOutcomeError("repair child artifact must differ from its parent")
        if self.status is RepairOutcomeStatus.ACCEPTED:
            if self.restored_parent_exactly or not self.search_parent_eligible:
                raise RepairOutcomeError("accepted repairs must remain eligible search parents")
            if not self.artifacts or not any(item.published for item in self.artifacts):
                raise RepairOutcomeError("accepted repairs require a published artifact")
            if not self.heldout_evidence or any(
                item.improvement < item.minimum_improvement for item in self.heldout_evidence
            ):
                raise RepairOutcomeError("accepted repairs require held-out improvement evidence")
            if self.failure_reason is not None:
                raise RepairOutcomeError("accepted repairs cannot carry a failure reason")
        else:
            if not self.restored_parent_exactly or self.search_parent_eligible:
                raise RepairOutcomeError(
                    "rejected repairs must restore the exact candidate and leave no parent"
                )
            if any(item.published for item in self.artifacts):
                raise RepairOutcomeError("rejected repairs cannot publish search artifacts")
            if self.failure_reason is None:
                raise RepairOutcomeError("non-accepted repairs require a failure reason")

    @property
    def outcome_id(self) -> str:
        return "repair_" + hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": REPAIR_OUTCOME_SCHEMA_VERSION,
            "protocol_revision": REPAIR_OUTCOME_PROTOCOL_REVISION,
            "request": self.request.to_record(),
            "status": self.status.value,
            "cost": self.cost.to_record(),
            "artifacts": [item.to_record() for item in self.artifacts],
            "lineage": self.lineage.to_record(),
            "heldout_evidence": [item.to_record() for item in self.heldout_evidence],
            "restored_parent_exactly": self.restored_parent_exactly,
            "search_parent_eligible": self.search_parent_eligible,
            "failure_reason": self.failure_reason,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["outcome_id"] = self.outcome_id
        return record


@dataclass(frozen=True, slots=True)
class RepairOutcomeManifest:
    requests: tuple[RepairRequest, ...]
    results: tuple[RepairResult, ...] = ()
    schema_version: int = REPAIR_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_OUTCOME_SCHEMA_VERSION:
            raise RepairOutcomeError("unsupported repair outcome schema")
        request_ids = tuple(item.request_id for item in self.requests)
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise RepairOutcomeError("repair requests must be non-empty and unique")
        result_ids = tuple(item.request.request_id for item in self.results)
        if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= set(request_ids):
            raise RepairOutcomeError("repair results must reference each request at most once")

    @property
    def complete(self) -> bool:
        return len(self.results) == len(self.requests)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_OUTCOME_PROTOCOL_REVISION,
            "requests": [item.to_record() for item in self.requests],
            "results": [item.to_record() for item in self.results],
            "complete": self.complete,
        }


def record_repair_result(
    manifest: RepairOutcomeManifest, result: RepairResult
) -> RepairOutcomeManifest:
    """Append one result without replacing a prior parent or outcome."""

    if result.request.request_id not in {item.request_id for item in manifest.requests}:
        raise RepairOutcomeError("repair result references an unknown request")
    if any(item.request.request_id == result.request.request_id for item in manifest.results):
        raise RepairOutcomeError("repair result already recorded")
    return replace(manifest, results=(*manifest.results, result))
