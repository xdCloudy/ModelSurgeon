"""Automatic, fail-closed model capability and candidate-space planning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.evaluation.architecture_compatibility import (
    ARCHITECTURE_COMPATIBILITY_MATRIX,
    ArchitectureCompatibilityMatrix,
    ArchitectureOperation,
    ArchitectureProfile,
    ArchitectureSupportStatus,
)
from modelsurgeon.search.candidate_space import (
    ArchitectureCandidateSpace,
    AxisDomain,
    CandidateSpaceConfig,
    CandidateSpaceRules,
    HardwareRule,
)
from modelsurgeon.search.deployable_state import ArchitectureAxis

AUTO_SPACE_SCHEMA_VERSION = 1


class AutoSpaceError(ValueError):
    """Raised when an automatic candidate-space request is malformed."""


class AutoSpaceOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ExclusionStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    FAILED = "failed"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutoSpaceError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64:
        raise AutoSpaceError(f"{label} must be a 64-character digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise AutoSpaceError(f"{label} must be hexadecimal") from error
    return result


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise AutoSpaceError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class AutoHardwareProfile:
    """Bounded target hardware envelope used for dry-run placement."""

    profile_id: str
    revision: str
    ram_bytes: int
    vram_bytes: int | None
    disk_bytes: int

    def __post_init__(self) -> None:
        _text(self.profile_id, "hardware profile ID")
        _text(self.revision, "hardware profile revision")
        _positive(self.ram_bytes, "hardware RAM")
        if self.vram_bytes is not None:
            _positive(self.vram_bytes, "hardware VRAM")
        _positive(self.disk_bytes, "hardware disk")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "revision": self.revision,
            "ram_bytes": self.ram_bytes,
            "vram_bytes": self.vram_bytes,
            "disk_bytes": self.disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilityInput:
    """Immutable model discovery result consumed by the planner."""

    model_id: str
    revision: str
    base_state_id: str
    family: ModelFamily | None
    profile: ArchitectureProfile
    parameter_count: int
    storage_bytes: int
    runtime_revision: str | None

    def __post_init__(self) -> None:
        _text(self.model_id, "model ID")
        _text(self.revision, "model revision")
        if not self.base_state_id.startswith("state_"):
            raise AutoSpaceError("model capability input requires a canonical base state")
        _positive(self.parameter_count, "model parameter count")
        _positive(self.storage_bytes, "model storage size")

    def to_record(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "base_state_id": self.base_state_id,
            "family": None if self.family is None else self.family.value,
            "profile": self.profile.value,
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "runtime_revision": self.runtime_revision,
        }


@dataclass(frozen=True, slots=True)
class AutoSpaceExclusion:
    key: str
    status: ExclusionStatus
    reason: str

    def __post_init__(self) -> None:
        _text(self.key, "exclusion key")
        _text(self.reason, "exclusion reason")

    def to_record(self) -> dict[str, str]:
        return {"key": self.key, "status": self.status.value, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class AutoCandidateSpaceRequest:
    model: ModelCapabilityInput
    hardware: AutoHardwareProfile
    objective_revision: str
    tool_revision: str
    seed: int
    domains: tuple[AxisDomain, ...]
    objective_metrics: tuple[str, ...]
    required_operations: tuple[ArchitectureOperation, ...] = (ArchitectureOperation.ANALYSIS,)
    required_plugins: tuple[str, ...] = ()
    available_plugins: tuple[str, ...] = ()
    coupling_axes: tuple[tuple[ArchitectureAxis, ArchitectureAxis], ...] = ()
    max_candidates: int = 10_000
    page_size: int = 256
    max_evaluations: int = 256

    def __post_init__(self) -> None:
        _text(self.objective_revision, "objective revision")
        _text(self.tool_revision, "tool revision")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise AutoSpaceError("candidate-space seed must be an unsigned 64-bit integer")
        if not self.domains:
            raise AutoSpaceError("automatic candidate spaces require discovered axis domains")
        if not self.objective_metrics:
            raise AutoSpaceError("automatic candidate spaces require objective metrics")
        if len(set(self.required_plugins)) != len(self.required_plugins):
            raise AutoSpaceError("required plugins must be unique")
        if len(set(self.available_plugins)) != len(self.available_plugins):
            raise AutoSpaceError("available plugins must be unique")
        if not 0 < self.max_candidates <= 100_000:
            raise AutoSpaceError("candidate cap must be within 1..100000")
        if not 0 < self.page_size <= self.max_candidates:
            raise AutoSpaceError("page size must be within the candidate cap")
        _positive(self.max_evaluations, "evaluation cap")


@dataclass(frozen=True, slots=True)
class AutoCandidateSpacePlan:
    outcome: AutoSpaceOutcome
    request: AutoCandidateSpaceRequest
    space: ArchitectureCandidateSpace | None
    exclusions: tuple[AutoSpaceExclusion, ...]
    candidate_upper_bound: int
    evaluation_upper_bound: int
    artifact_upper_bound_bytes: int
    matrix_id: str

    @property
    def plan_id(self) -> str:
        payload = {
            "schema_version": AUTO_SPACE_SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "request": self.request_record(),
            "exclusions": [item.to_record() for item in self.exclusions],
            "candidate_upper_bound": self.candidate_upper_bound,
            "evaluation_upper_bound": self.evaluation_upper_bound,
            "artifact_upper_bound_bytes": self.artifact_upper_bound_bytes,
            "matrix_id": self.matrix_id,
            "space_id": None if self.space is None else self.space.space_id,
        }
        return f"auto_space_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def request_record(self) -> dict[str, object]:
        return {
            "model": self.request.model.to_record(),
            "hardware": self.request.hardware.to_record(),
            "objective_revision": self.request.objective_revision,
            "tool_revision": self.request.tool_revision,
            "seed": self.request.seed,
            "domains": [item.to_record() for item in self.request.domains],
            "objective_metrics": list(self.request.objective_metrics),
            "required_operations": [item.value for item in self.request.required_operations],
            "required_plugins": list(self.request.required_plugins),
            "available_plugins": list(self.request.available_plugins),
            "coupling_axes": [
                [left.value, right.value] for left, right in self.request.coupling_axes
            ],
            "max_candidates": self.request.max_candidates,
            "page_size": self.request.page_size,
            "max_evaluations": self.request.max_evaluations,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "automatic_candidate_space_plan",
            "schema_version": AUTO_SPACE_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "outcome": self.outcome.value,
            "request": self.request_record(),
            "space_id": None if self.space is None else self.space.space_id,
            "exclusions": [item.to_record() for item in self.exclusions],
            "candidate_upper_bound": self.candidate_upper_bound,
            "evaluation_upper_bound": self.evaluation_upper_bound,
            "artifact_upper_bound_bytes": self.artifact_upper_bound_bytes,
            "compatibility_matrix_id": self.matrix_id,
        }


_KNOWN_OBJECTIVES = frozenset(
    {"quality", "perplexity", "parameter_count", "latency", "memory", "disk_size"}
)


def build_auto_candidate_space(
    request: AutoCandidateSpaceRequest,
    *,
    compatibility_matrix: ArchitectureCompatibilityMatrix = ARCHITECTURE_COMPATIBILITY_MATRIX,
) -> AutoCandidateSpacePlan:
    """Build a bounded space or retain explicit refusal/exclusion evidence."""

    exclusions: list[AutoSpaceExclusion] = []
    if request.model.family is None:
        exclusions.append(
            AutoSpaceExclusion(
                "architecture",
                ExclusionStatus.UNKNOWN,
                "architecture family is not known from pinned discovery evidence",
            )
        )
    if request.model.runtime_revision is None:
        exclusions.append(
            AutoSpaceExclusion(
                "runtime",
                ExclusionStatus.UNKNOWN,
                "runtime revision is missing; capability results are not comparable",
            )
        )
    missing_plugins = sorted(set(request.required_plugins) - set(request.available_plugins))
    exclusions.extend(
        AutoSpaceExclusion(
            f"plugin:{plugin}",
            ExclusionStatus.UNSUPPORTED,
            "required plugin capability is not registered",
        )
        for plugin in missing_plugins
    )
    unknown_objectives = sorted(set(request.objective_metrics) - _KNOWN_OBJECTIVES)
    exclusions.extend(
        AutoSpaceExclusion(
            f"objective:{metric}",
            ExclusionStatus.UNSUPPORTED,
            "objective metric has no built-in candidate-space adapter",
        )
        for metric in unknown_objectives
    )
    domain_axes = {domain.axis for domain in request.domains}
    for left, right in request.coupling_axes:
        if left not in domain_axes or right not in domain_axes:
            exclusions.append(
                AutoSpaceExclusion(
                    f"coupling:{left.value}:{right.value}",
                    ExclusionStatus.FAILED,
                    "coupling references an incompletely discovered axis",
                )
            )
    if request.model.storage_bytes > request.hardware.disk_bytes:
        exclusions.append(
            AutoSpaceExclusion(
                "hardware:disk",
                ExclusionStatus.FAILED,
                "source artifact exceeds the target disk envelope",
            )
        )
    if request.model.family is not None:
        for operation in request.required_operations:
            cell = next(
                cell
                for cell in compatibility_matrix.cells
                if cell.family is request.model.family
                and cell.profile is request.model.profile
                and cell.operation is operation
            )
            if cell.status is ArchitectureSupportStatus.UNSUPPORTED:
                exclusions.append(
                    AutoSpaceExclusion(
                        f"operation:{operation.value}",
                        ExclusionStatus.UNSUPPORTED,
                        cell.reason,
                    )
                )
            elif cell.status is ArchitectureSupportStatus.UNKNOWN:
                exclusions.append(
                    AutoSpaceExclusion(
                        f"operation:{operation.value}",
                        ExclusionStatus.UNKNOWN,
                        cell.reason,
                    )
                )
    candidate_upper_bound = math.prod(len(domain.choices) for domain in request.domains)
    candidate_upper_bound = min(candidate_upper_bound, request.max_candidates)
    artifact_upper_bound = min(request.hardware.disk_bytes, request.model.storage_bytes * 2)
    outcome = AutoSpaceOutcome.SUPPORTED
    if any(item.status is ExclusionStatus.FAILED for item in exclusions):
        outcome = AutoSpaceOutcome.FAILED
    elif any(item.status is ExclusionStatus.UNSUPPORTED for item in exclusions):
        outcome = AutoSpaceOutcome.UNSUPPORTED
    elif any(item.status is ExclusionStatus.UNKNOWN for item in exclusions):
        outcome = AutoSpaceOutcome.UNKNOWN
    if outcome is not AutoSpaceOutcome.SUPPORTED:
        return AutoCandidateSpacePlan(
            outcome,
            request,
            None,
            tuple(sorted(exclusions, key=lambda item: item.key)),
            candidate_upper_bound,
            min(candidate_upper_bound, request.max_evaluations),
            artifact_upper_bound,
            compatibility_matrix.matrix_id,
        )
    rules = CandidateSpaceRules(
        max_storage_bytes=request.hardware.disk_bytes,
        divisible_axes=request.coupling_axes,
    )
    space = ArchitectureCandidateSpace(
        request.model.base_state_id,
        request.domains,
        (HardwareRule(request.hardware.profile_id, max_storage_bytes=request.hardware.disk_bytes),),
        rules,
        CandidateSpaceConfig(request.seed, request.max_candidates, request.page_size),
    )
    return AutoCandidateSpacePlan(
        outcome,
        request,
        space,
        (),
        candidate_upper_bound,
        min(candidate_upper_bound, request.max_evaluations),
        artifact_upper_bound,
        compatibility_matrix.matrix_id,
    )


__all__ = [
    "AUTO_SPACE_SCHEMA_VERSION",
    "AutoCandidateSpacePlan",
    "AutoCandidateSpaceRequest",
    "AutoHardwareProfile",
    "AutoSpaceError",
    "AutoSpaceExclusion",
    "AutoSpaceOutcome",
    "ExclusionStatus",
    "ModelCapabilityInput",
    "build_auto_candidate_space",
]
