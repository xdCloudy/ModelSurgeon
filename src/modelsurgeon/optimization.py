"""Deterministic, declarative optimization planning contracts.

The planner deliberately stops at a read-only plan boundary.  A plan records
the resolved inputs, resource envelope, approvals, and uncertainties needed by
an executor without allowing a dry-run to mutate a model or an artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.config import Settings
from modelsurgeon.experiments.identity import canonical_identity_json

OPTIMIZE_PLAN_SCHEMA_VERSION = 1


class OptimizePlanError(ValueError):
    """Raised when an optimize request cannot be planned safely."""


class OptimizeOutcome(StrEnum):
    """Explicit plan outcome, including non-success states."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class OptimizePreset(StrEnum):
    """Bounded resource/quality presets exposed by the public workflow."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Versioned hardware envelope used for planning, never for probing."""

    profile_id: str
    device: str
    cpu_threads: int
    ram_gb: float
    vram_gb: float | None
    disk_gb: float

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.device.strip():
            raise OptimizePlanError("hardware profile identifiers must be non-empty")
        if self.cpu_threads <= 0 or self.ram_gb <= 0 or self.disk_gb <= 0:
            raise OptimizePlanError("hardware profile resources must be positive")
        if self.vram_gb is not None and self.vram_gb <= 0:
            raise OptimizePlanError("hardware profile VRAM must be positive when present")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "disk_gb": self.disk_gb,
        }


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """Quality target and bounded evaluation budget."""

    profile_id: str
    min_quality_retention: float
    max_perplexity_delta: float | None
    evaluation_repetitions: int
    confidence_level: float

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise OptimizePlanError("quality profile identifier must be non-empty")
        if not 0.0 <= self.min_quality_retention <= 1.0:
            raise OptimizePlanError("quality retention must be between zero and one")
        if self.max_perplexity_delta is not None and self.max_perplexity_delta < 0:
            raise OptimizePlanError("perplexity delta cannot be negative")
        if self.evaluation_repetitions <= 0:
            raise OptimizePlanError("evaluation repetitions must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise OptimizePlanError("confidence level must be between zero and one")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "min_quality_retention": self.min_quality_retention,
            "max_perplexity_delta": self.max_perplexity_delta,
            "evaluation_repetitions": self.evaluation_repetitions,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class OptimizeBudget:
    """Hard, auditable resource limits for a planned optimization."""

    evaluations: int
    repair_steps: int
    max_artifact_bytes: int
    max_ram_bytes: int
    max_vram_bytes: int | None
    max_disk_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.evaluations,
            self.repair_steps,
            self.max_artifact_bytes,
            self.max_ram_bytes,
            self.max_disk_bytes,
        )
        if any(value <= 0 for value in values):
            raise OptimizePlanError("all optimization budgets must be positive")
        if self.max_vram_bytes is not None and self.max_vram_bytes <= 0:
            raise OptimizePlanError("maximum VRAM budget must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "evaluations": self.evaluations,
            "repair_steps": self.repair_steps,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_ram_bytes": self.max_ram_bytes,
            "max_vram_bytes": self.max_vram_bytes,
            "max_disk_bytes": self.max_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class OptimizeCostEstimate:
    """Conservative cost estimate retained with the plan."""

    download_bytes: int | None
    evaluation_count: int
    repair_steps: int
    artifact_bytes: int
    cpu_seconds: int
    peak_ram_gb: float
    peak_vram_gb: float | None

    def __post_init__(self) -> None:
        for value in (
            self.evaluation_count,
            self.repair_steps,
            self.artifact_bytes,
            self.cpu_seconds,
        ):
            if value < 0:
                raise OptimizePlanError("cost estimates cannot be negative")
        if self.download_bytes is not None and self.download_bytes < 0:
            raise OptimizePlanError("download estimate cannot be negative")
        if self.peak_ram_gb <= 0 or (self.peak_vram_gb is not None and self.peak_vram_gb <= 0):
            raise OptimizePlanError("peak resource estimates must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "download_bytes": self.download_bytes,
            "evaluation_count": self.evaluation_count,
            "repair_steps": self.repair_steps,
            "artifact_bytes": self.artifact_bytes,
            "cpu_seconds": self.cpu_seconds,
            "peak_ram_gb": self.peak_ram_gb,
            "peak_vram_gb": self.peak_vram_gb,
        }


@dataclass(frozen=True, slots=True)
class OptimizeApproval:
    """An explicit human approval point in the eventual execution flow."""

    code: str
    required: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.reason.strip():
            raise OptimizePlanError("approval points require a code and reason")

    def to_record(self) -> dict[str, object]:
        return {"code": self.code, "required": self.required, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class OptimizeInput:
    """One planned input and its provenance status."""

    name: str
    value: object
    required: bool
    known: bool
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip():
            raise OptimizePlanError("planned inputs require names and sources")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "required": self.required,
            "known": self.known,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OptimizePlan:
    """Stable, JSON-serializable optimization plan."""

    outcome: OptimizeOutcome
    preset: OptimizePreset
    hardware_profile: HardwareProfile
    quality_profile: QualityProfile
    resolved_config: Mapping[str, object]
    inputs: tuple[OptimizeInput, ...]
    budget: OptimizeBudget
    cost: OptimizeCostEstimate
    commands: tuple[str, ...]
    approvals: tuple[OptimizeApproval, ...]
    uncertainties: tuple[str, ...]
    limitations: tuple[str, ...]
    dry_run: bool
    source_overwrite: bool
    rollback_policy: str = "no-overwrite; executor must use atomic write and retain prior digest"

    def __post_init__(self) -> None:
        if not self.commands:
            raise OptimizePlanError("an optimization plan must contain a command")
        if any(not value.strip() for value in self.uncertainties + self.limitations):
            raise OptimizePlanError("plan uncertainties and limitations must be non-empty")
        if self.source_overwrite:
            raise OptimizePlanError("optimization plans cannot permit source overwrite")

    @property
    def config_digest(self) -> str:
        return hashlib.sha256(
            canonical_identity_json(self.resolved_config).encode("utf-8")
        ).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZE_PLAN_SCHEMA_VERSION,
            "preset": self.preset.value,
            "hardware_profile": self.hardware_profile.to_record(),
            "quality_profile": self.quality_profile.to_record(),
            "resolved_config": dict(self.resolved_config),
            "inputs": [item.to_record() for item in self.inputs],
            "budget": self.budget.to_record(),
            "cost": self.cost.to_record(),
            "commands": list(self.commands),
            "approvals": [item.to_record() for item in self.approvals],
            "dry_run": self.dry_run,
            "source_overwrite": self.source_overwrite,
            "rollback_policy": self.rollback_policy,
        }

    @property
    def plan_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode("utf-8")
        ).hexdigest()
        return f"optimize_plan_{digest}"

    @property
    def resume_token(self) -> str:
        return hashlib.sha256(f"resume:{self.plan_id}".encode()).hexdigest()

    @property
    def executable(self) -> bool:
        return self.outcome is OptimizeOutcome.SUPPORTED and not self.uncertainties

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "optimize_plan",
            "schema_version": OPTIMIZE_PLAN_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "config_digest": self.config_digest,
            "resume_token": self.resume_token,
            "outcome": self.outcome.value,
            "preset": self.preset.value,
            "hardware_profile": self.hardware_profile.to_record(),
            "quality_profile": self.quality_profile.to_record(),
            "resolved_config": dict(self.resolved_config),
            "inputs": [item.to_record() for item in self.inputs],
            "budget": self.budget.to_record(),
            "cost_estimate": self.cost.to_record(),
            "exact_commands": list(self.commands),
            "approval_points": [item.to_record() for item in self.approvals],
            "uncertainties": list(self.uncertainties),
            "limitations": list(self.limitations),
            "dry_run": self.dry_run,
            "executable": self.executable,
            "source_overwrite": self.source_overwrite,
            "rollback_policy": self.rollback_policy,
            "artifact_lineage": {
                "source_model": self.resolved_config.get("model", {}),
                "resolved_config_digest": self.config_digest,
                "plan_id": self.plan_id,
            },
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_record(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


_HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "cpu-small": HardwareProfile("cpu-small", "cpu", 2, 8.0, None, 20.0),
    "cpu-large": HardwareProfile("cpu-large", "cpu", 8, 32.0, None, 100.0),
    "gpu-12gb": HardwareProfile("gpu-12gb", "cuda:0", 8, 32.0, 12.0, 100.0),
    "gpu-24gb": HardwareProfile("gpu-24gb", "cuda:0", 16, 64.0, 24.0, 200.0),
}
_HARDWARE_ALIASES = {"cpu": "cpu-small", "local-cpu": "cpu-small", "gpu": "gpu-12gb"}
_QUALITY_PROFILES: dict[str, QualityProfile] = {
    "fast": QualityProfile("fast", 0.95, 0.10, 1, 0.90),
    "balanced": QualityProfile("balanced", 0.98, 0.05, 3, 0.95),
    "quality": QualityProfile("quality", 0.995, 0.02, 5, 0.99),
}
_QUALITY_ALIASES = {"high": "quality", "maximum": "quality"}


def available_hardware_profiles() -> tuple[str, ...]:
    """Return stable hardware profile identifiers for CLI/API discovery."""

    return tuple(sorted(_HARDWARE_PROFILES))


def available_quality_profiles() -> tuple[str, ...]:
    """Return stable quality profile identifiers for CLI/API discovery."""

    return tuple(sorted(_QUALITY_PROFILES))


def _profile(mapping: Mapping[str, object], name: str, label: str) -> object:
    normalized = name.strip().lower()
    aliases = mapping.get(normalized)
    if aliases is None:
        choices = ", ".join(sorted(mapping))
        raise OptimizePlanError(f"unknown {label} {name!r}; choose one of: {choices}")
    return aliases


def _resolve_hardware(name: str) -> HardwareProfile:
    normalized = name.strip().lower()
    normalized = _HARDWARE_ALIASES.get(normalized, normalized)
    profile = _HARDWARE_PROFILES.get(normalized)
    if profile is None:
        choices = ", ".join(available_hardware_profiles())
        raise OptimizePlanError(f"unknown hardware profile {name!r}; choose one of: {choices}")
    return profile


def _resolve_quality(name: str) -> QualityProfile:
    normalized = name.strip().lower()
    normalized = _QUALITY_ALIASES.get(normalized, normalized)
    profile = _QUALITY_PROFILES.get(normalized)
    if profile is None:
        choices = ", ".join(available_quality_profiles())
        raise OptimizePlanError(f"unknown quality profile {name!r}; choose one of: {choices}")
    return profile


def _preset_budget(
    preset: OptimizePreset, quality: QualityProfile, hardware: HardwareProfile
) -> OptimizeBudget:
    factors = {OptimizePreset.FAST: 1, OptimizePreset.BALANCED: 2, OptimizePreset.QUALITY: 4}
    factor = factors[preset]
    return OptimizeBudget(
        evaluations=quality.evaluation_repetitions * factor,
        repair_steps=16 * factor,
        max_artifact_bytes=int(hardware.disk_gb * 1024**3 * 0.25),
        max_ram_bytes=int(hardware.ram_gb * 1024**3 * 0.75),
        max_vram_bytes=(
            None if hardware.vram_gb is None else int(hardware.vram_gb * 1024**3 * 0.80)
        ),
        max_disk_bytes=int(hardware.disk_gb * 1024**3 * 0.80),
    )


def _estimate_cost(
    *,
    preset: OptimizePreset,
    quality: QualityProfile,
    hardware: HardwareProfile,
    budget: OptimizeBudget,
    model_known: bool,
) -> OptimizeCostEstimate:
    factor = {OptimizePreset.FAST: 1, OptimizePreset.BALANCED: 2, OptimizePreset.QUALITY: 4}[preset]
    peak_vram = None if hardware.vram_gb is None else min(hardware.vram_gb * 0.72, hardware.vram_gb)
    return OptimizeCostEstimate(
        download_bytes=None if not model_known else 0,
        evaluation_count=budget.evaluations,
        repair_steps=budget.repair_steps,
        artifact_bytes=min(budget.max_artifact_bytes, 256 * 1024**2 * factor),
        cpu_seconds=quality.evaluation_repetitions * factor * max(30, 120 // hardware.cpu_threads),
        peak_ram_gb=min(hardware.ram_gb * 0.70, 16.0 + factor),
        peak_vram_gb=peak_vram,
    )


def build_optimize_plan(
    settings: Settings,
    *,
    preset: str = "balanced",
    hardware_profile: str = "cpu-small",
    quality_profile: str | None = None,
    dry_run: bool = True,
) -> OptimizePlan:
    """Resolve settings and produce a deterministic, read-only optimization plan."""

    try:
        resolved_preset = OptimizePreset(preset.strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in OptimizePreset)
        raise OptimizePlanError(f"unknown preset {preset!r}; choose one of: {choices}") from exc
    hardware = _resolve_hardware(hardware_profile)
    selected_quality = quality_profile or resolved_preset.value
    quality = _resolve_quality(selected_quality)
    if resolved_preset is OptimizePreset.FAST and quality.profile_id == "quality":
        raise OptimizePlanError(
            "fast preset cannot use the quality profile; choose balanced or quality"
        )

    resolved_config = settings.canonical_dict()
    constraints = settings.constraints
    uncertainties: list[str] = []
    limitations: list[str] = [
        "planning is read-only; execution uses a separate first-party runtime boundary",
        "costs are conservative estimates until a retained campaign measurement exists",
    ]
    failures: list[str] = []
    unsupported: list[str] = []

    model_path = settings.model.path
    model_revision = settings.model.revision
    if not model_path:
        uncertainties.append(
            "model.path is unresolved; supply an immutable local path or model identifier"
        )
    if not model_revision:
        uncertainties.append("model.revision is unresolved; pin a revision before execution")
    if settings.model.format is ModelFormat.GGUF:
        runtime = settings.runtime
        if settings.model.family is None:
            unsupported.append(
                "native GGUF optimization requires an explicit model.family because "
                "general.architecture aliases can be ambiguous"
            )
        missing_runtime = tuple(
            name
            for name, value in (
                ("runtime.llama_cli", runtime.llama_cli),
                ("runtime.llama_perplexity", runtime.llama_perplexity),
                ("runtime.llama_bench", runtime.llama_bench),
                ("runtime.expected_revision", runtime.expected_revision),
                ("calibration.dataset_revision", settings.calibration.dataset_revision),
            )
            if value is None
        )
        if missing_runtime:
            unsupported.append(
                "native GGUF optimization requires explicit runtime and calibration identity: "
                + ", ".join(missing_runtime)
            )
        if tuple(settings.search.scopes) != ("mlp_channel",):
            unsupported.append(
                "native GGUF optimization currently supports only the model-wide mlp_channel scope"
            )
        if settings.repair.method != "none":
            unsupported.append(
                "native GGUF optimization does not support "
                f"repair.method={settings.repair.method!r}"
            )
        if settings.quantization.method != "none":
            unsupported.append(
                "native GGUF optimization does not support an additional quantization stage"
            )
    if constraints.max_ram_bytes is not None and constraints.max_ram_bytes < 4 * 1024**3:
        failures.append(
            "max_ram_bytes is below the 4 GiB minimum planning envelope; choose a larger limit"
        )
    if (
        constraints.max_vram_bytes is not None
        and hardware.vram_gb is not None
        and constraints.max_vram_bytes < int(hardware.vram_gb * 1024**3 * 0.50)
    ):
        failures.append(
            f"hardware profile {hardware.profile_id} needs more VRAM than the configured limit; "
            "choose cpu-small or raise constraints.max_vram_bytes"
        )
    if quality.min_quality_retention < constraints.min_quality_retention_ratio:
        failures.append(
            "quality profile is weaker than the configured minimum retention; "
            "choose a stricter profile"
        )
    if settings.safety.trust_remote_code:
        limitations.append("remote model code is enabled and requires explicit source review")

    budget = _preset_budget(resolved_preset, quality, hardware)
    cost = _estimate_cost(
        preset=resolved_preset,
        quality=quality,
        hardware=hardware,
        budget=budget,
        model_known=model_path is not None,
    )
    if constraints.max_ram_bytes is not None and budget.max_ram_bytes > constraints.max_ram_bytes:
        failures.append(
            "planned RAM budget exceeds constraints.max_ram_bytes; choose a smaller profile"
        )
    if (
        constraints.max_disk_bytes is not None
        and budget.max_disk_bytes > constraints.max_disk_bytes
    ):
        failures.append(
            "planned disk budget exceeds constraints.max_disk_bytes; choose a smaller profile"
        )

    if failures:
        outcome = OptimizeOutcome.FAILED
        uncertainties.extend(failures)
    elif unsupported:
        outcome = OptimizeOutcome.UNSUPPORTED
        uncertainties.extend(unsupported)
    elif uncertainties:
        outcome = OptimizeOutcome.UNKNOWN
    else:
        outcome = OptimizeOutcome.SUPPORTED

    budget_record = budget.to_record()
    commands = (
        "modelsurgeon",
        "optimize",
        "--dry-run" if dry_run else "--execute",
        "--preset",
        resolved_preset.value,
        "--hardware-profile",
        hardware.profile_id,
        "--quality-profile",
        quality.profile_id,
    )
    approvals = (
        OptimizeApproval(
            "plan_review", True, "review exact inputs, budgets, uncertainties, and plan identity"
        ),
        OptimizeApproval("source_model", True, "confirm the source model and immutable revision"),
        OptimizeApproval(
            "resource_budget", True, "confirm the bounded hardware and artifact envelope"
        ),
        OptimizeApproval(
            "artifact_write",
            not dry_run,
            "execution may write only new content-addressed artifacts with atomic commits",
        ),
    )
    inputs = (
        OptimizeInput(
            "model.path", model_path, True, model_path is not None, "Settings.model.path"
        ),
        OptimizeInput(
            "model.revision",
            model_revision,
            True,
            model_revision is not None,
            "Settings.model.revision",
        ),
        OptimizeInput("resolved_config", resolved_config, True, True, "Settings.canonical_dict"),
        OptimizeInput(
            "hardware_profile", hardware.profile_id, True, True, "optimize profile registry"
        ),
        OptimizeInput(
            "quality_profile", quality.profile_id, True, True, "optimize quality registry"
        ),
    )
    # Keep the local variable useful to readers and make it part of the checked
    # construction path without exposing a mutable budget mapping.
    if not budget_record:
        raise OptimizePlanError("optimization budget resolution produced no fields")
    return OptimizePlan(
        outcome=outcome,
        preset=resolved_preset,
        hardware_profile=hardware,
        quality_profile=quality,
        resolved_config=resolved_config,
        inputs=inputs,
        budget=budget,
        cost=cost,
        commands=commands,
        approvals=approvals,
        uncertainties=tuple(uncertainties),
        limitations=tuple(limitations),
        dry_run=dry_run,
        source_overwrite=settings.safety.allow_overwrite,
    )


def write_optimize_plan(path: Path, plan: OptimizePlan, *, allow_overwrite: bool = False) -> None:
    """Persist a plan as a new JSON artifact without mutating model sources."""

    if path.exists() and not allow_overwrite:
        raise OptimizePlanError(f"refusing to overwrite existing plan artifact: {path}")
    path.write_text(plan.canonical_json() + "\n", encoding="utf-8", newline="\n")


__all__ = [
    "OPTIMIZE_PLAN_SCHEMA_VERSION",
    "HardwareProfile",
    "OptimizeApproval",
    "OptimizeBudget",
    "OptimizeCostEstimate",
    "OptimizeInput",
    "OptimizeOutcome",
    "OptimizePlan",
    "OptimizePlanError",
    "OptimizePreset",
    "QualityProfile",
    "available_hardware_profiles",
    "available_quality_profiles",
    "build_optimize_plan",
    "write_optimize_plan",
]
