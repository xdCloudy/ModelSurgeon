"""Engine-owned model and host inspection context for chat sessions.

The chat provider consumes this record as bounded context.  It is assembled
from the same deterministic GGUF discovery, hardware-profile, and capability
matrix APIs used by direct workflows; provider text is never used as
inspection evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from modelsurgeon.adapters import ArchitectureEvidence, ModelFamily, detect_model_family
from modelsurgeon.adapters.gguf import (
    GGUFDiscoveryError,
    GGUFParseError,
    GGUFValueType,
    discover_gguf_components,
    open_gguf,
)
from modelsurgeon.evaluation.architecture_compatibility import (
    ARCHITECTURE_COMPATIBILITY_MATRIX,
)
from modelsurgeon.experiments import HardwareProfile, build_default_hardware_profile

CHAT_INSPECTION_SCHEMA_VERSION: Literal[1] = 1


class ChatInspectionOutcome(StrEnum):
    """Explicit inspection state; unknown and failed are never success."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ChatInspectionError(ValueError):
    """Raised when the selected chat model cannot be inspected safely."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


HardwareProfileFactory = Callable[[str], HardwareProfile]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ChatInspectionError("model_unreadable", f"cannot read chat model: {path}") from error
    return f"sha256:{digest.hexdigest()}"


def _model_metadata(path: Path) -> tuple[str, ModelFamily, tuple[str, ...], dict[str, object]]:
    try:
        with open_gguf(path) as mapped:
            entry = mapped.container.metadata_entry("general.architecture")
            if entry is None or entry.value_type is not GGUFValueType.STRING:
                raise ChatInspectionError(
                    "unsupported_model_metadata",
                    "GGUF chat models require string general.architecture metadata",
                )
            if not isinstance(entry.value, str) or not entry.value.strip():
                raise ChatInspectionError(
                    "unsupported_model_metadata",
                    "GGUF chat model architecture metadata is empty",
                )
            architecture = entry.value
            try:
                family = detect_model_family(ArchitectureEvidence(gguf_architecture=architecture))
            except ValueError as error:
                raise ChatInspectionError("unsupported_architecture", str(error)) from error
            return (
                architecture,
                family.family,
                family.matched_evidence,
                {
                    "record_type": "chat_model_metadata",
                    "evidence_kind": "detected_model_metadata",
                    "source": "modelsurgeon.adapters.gguf.open_gguf",
                    "outcome": ChatInspectionOutcome.SUPPORTED.value,
                    "architecture": architecture,
                    "family": family.family.value,
                    "family_evidence": list(family.matched_evidence),
                },
            )
    except ChatInspectionError:
        raise
    except GGUFParseError as error:
        raise ChatInspectionError(
            "invalid_model", "GGUF chat model container or metadata validation failed"
        ) from error


def _discovery_record(
    path: Path,
    family: ModelFamily,
) -> tuple[ChatInspectionOutcome, dict[str, object]]:
    try:
        with open_gguf(path) as mapped:
            discovery = discover_gguf_components(mapped.container, family=family)
    except GGUFDiscoveryError as error:
        # A provider-only GGUF may contain enough metadata to run chat but not
        # enough shape/tensor evidence for structural inspection.  Preserve
        # that distinction instead of turning it into a false capability claim.
        outcome = (
            ChatInspectionOutcome.UNKNOWN
            if type(error).__name__ in {"MissingGGUFMetadataError", "MissingGGUFTensorError"}
            else ChatInspectionOutcome.FAILED
        )
        return outcome, {
            "record_type": "chat_model_discovery",
            "evidence_kind": "canonical_model_discovery",
            "source": "modelsurgeon.adapters.gguf.discover_gguf_components",
            "outcome": outcome.value,
            "discovery": None,
            "reason": str(error),
        }
    return ChatInspectionOutcome.SUPPORTED, {
        "record_type": "chat_model_discovery",
        "evidence_kind": "canonical_model_discovery",
        "source": "modelsurgeon.adapters.gguf.discover_gguf_components",
        "outcome": ChatInspectionOutcome.SUPPORTED.value,
        "discovery": discovery.to_record(),
        "reason": None,
    }


def _capability_record(family: ModelFamily) -> dict[str, object]:
    cells = tuple(cell for cell in ARCHITECTURE_COMPATIBILITY_MATRIX.cells if cell.family is family)
    return {
        "record_type": "chat_capability_context",
        "evidence_kind": "engine_capability_matrix",
        "source": "modelsurgeon.evaluation.architecture_compatibility",
        "outcome": ChatInspectionOutcome.SUPPORTED.value,
        "matrix_id": ARCHITECTURE_COMPATIBILITY_MATRIX.matrix_id,
        "cells": [cell.to_record() for cell in cells],
        "live_evidence": False,
    }


def _hardware_record(
    profile_factory: HardwareProfileFactory,
    path: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    try:
        profile = profile_factory(str(path.parent))
    except Exception as error:
        return (
            {
                "record_type": "chat_hardware_context",
                "evidence_kind": "detected_hardware_inventory",
                "source": (
                    "modelsurgeon.experiments.hardware_profile."
                    "build_default_hardware_profile"
                ),
                "outcome": ChatInspectionOutcome.FAILED.value,
                "profile": None,
                "memory": {
                    "outcome": ChatInspectionOutcome.UNKNOWN.value,
                    "inventory": None,
                },
                "reason": f"hardware inventory failed: {type(error).__name__}",
            },
            (f"hardware inventory failed: {type(error).__name__}",),
        )
    inventory = profile.inventory
    memory = inventory.memory.to_record()
    memory_outcome = (
        ChatInspectionOutcome.SUPPORTED.value
        if memory["total_bytes"] is not None and memory["available_bytes"] is not None
        else ChatInspectionOutcome.UNKNOWN.value
    )
    warnings = tuple(inventory.warnings)
    return (
        {
            "record_type": "chat_hardware_context",
            "evidence_kind": "detected_hardware_inventory",
            "source": "modelsurgeon.experiments.hardware_profile.build_default_hardware_profile",
            "outcome": ChatInspectionOutcome.SUPPORTED.value,
            "profile": profile.to_record(),
            "memory": {
                "outcome": memory_outcome,
                "inventory": memory,
            },
            "reason": None,
        },
        warnings,
    )


@dataclass(frozen=True, slots=True)
class ChatInspectionContext:
    """Canonical model/host context made available to intent compilation."""

    outcome: ChatInspectionOutcome
    model: Mapping[str, object]
    runtime: Mapping[str, object]
    hardware: Mapping[str, object]
    capabilities: Mapping[str, object]
    warnings: tuple[str, ...] = ()
    schema_version: Literal[1] = CHAT_INSPECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHAT_INSPECTION_SCHEMA_VERSION:
            raise ChatInspectionError(
                "schema_version", "unsupported chat inspection schema version"
            )
        if self.warnings != tuple(sorted(set(self.warnings))):
            raise ChatInspectionError(
                "warnings", "chat inspection warnings must be sorted and unique"
            )

    @property
    def inspection_id(self) -> str:
        return _digest(self._identity_record())

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "model": dict(self.model),
            "runtime": dict(self.runtime),
            "hardware": dict(self.hardware),
            "capabilities": dict(self.capabilities),
            "warnings": list(self.warnings),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "chat_inspection_context",
            "schema_version": self.schema_version,
            "inspection_id": self.inspection_id,
            **self._identity_record(),
        }

    def with_provider(
        self,
        *,
        runtime_revision: str,
        capability_card: Mapping[str, object],
    ) -> ChatInspectionContext:
        runtime = {
            "record_type": "chat_runtime_context",
            "evidence_kind": "provider_declaration",
            "source": "modelsurgeon.conversation.provider",
            "outcome": ChatInspectionOutcome.SUPPORTED.value,
            "revision": runtime_revision,
            "capability_card": dict(capability_card),
            "live_evidence": False,
            "reason": "provider startup accepted the configured runtime revision",
        }
        return replace(self, runtime=runtime)


def inspect_local_chat_model(
    path: Path,
    *,
    model_revision: str | None = None,
    runtime_revision: str | None = None,
    capability_card: Mapping[str, object] | None = None,
    hardware_profile_factory: HardwareProfileFactory = build_default_hardware_profile,
) -> ChatInspectionContext:
    """Inspect a local GGUF with the same engine-owned APIs used by direct workflows."""

    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ChatInspectionError("model_missing", f"chat model file does not exist: {path}")
    if resolved.suffix.lower() != ".gguf":
        raise ChatInspectionError(
            "unsupported_model_format",
            "the local chat provider supports only .gguf model files",
        )
    revision = _file_digest(resolved)
    if model_revision is not None and model_revision != revision:
        raise ChatInspectionError(
            "model_revision_mismatch",
            "--model-revision does not match the local model SHA-256",
        )
    architecture, family, evidence, model_metadata = _model_metadata(resolved)
    discovery_outcome, discovery = _discovery_record(resolved, family)
    model = {
        **model_metadata,
        "path": str(resolved),
        "revision": revision,
        "architecture": architecture,
        "family": family.value,
        "family_evidence": list(evidence),
        "discovery": discovery,
    }
    hardware, hardware_warnings = _hardware_record(hardware_profile_factory, resolved)
    memory_context = cast(Mapping[str, object], hardware["memory"])
    memory_outcome = str(memory_context["outcome"])
    runtime_outcome = (
        ChatInspectionOutcome.SUPPORTED.value
        if runtime_revision is not None and capability_card is not None
        else ChatInspectionOutcome.UNKNOWN.value
    )
    runtime: dict[str, object] = {
        "record_type": "chat_runtime_context",
        "evidence_kind": "provider_declaration",
        "source": "modelsurgeon.conversation.provider",
        "outcome": runtime_outcome,
        "revision": runtime_revision,
        "capability_card": None if capability_card is None else dict(capability_card),
        "live_evidence": False,
        "reason": (
            None
            if runtime_outcome == ChatInspectionOutcome.SUPPORTED.value
            else "provider runtime capability has not been established"
        ),
    }
    capability = _capability_record(family)
    outcomes = (discovery_outcome.value, memory_outcome, runtime_outcome)
    outcome = (
        ChatInspectionOutcome.FAILED
        if ChatInspectionOutcome.FAILED.value in outcomes
        else ChatInspectionOutcome.UNKNOWN
        if ChatInspectionOutcome.UNKNOWN.value in outcomes
        else ChatInspectionOutcome.SUPPORTED
    )
    warnings = tuple(sorted(set(hardware_warnings)))
    return ChatInspectionContext(outcome, model, runtime, hardware, capability, warnings)


def no_provider_inspection_context(
    *,
    hardware_profile_factory: HardwareProfileFactory = build_default_hardware_profile,
    path: Path = Path("."),
) -> ChatInspectionContext:
    """Retain explicit unavailable model/runtime evidence in no-provider mode."""

    hardware, warnings = _hardware_record(hardware_profile_factory, path.resolve())
    return ChatInspectionContext(
        ChatInspectionOutcome.UNSUPPORTED,
        {
            "record_type": "chat_model_metadata",
            "evidence_kind": "detected_model_metadata",
            "source": "modelsurgeon.adapters.gguf.open_gguf",
            "outcome": ChatInspectionOutcome.UNSUPPORTED.value,
            "path": None,
            "revision": None,
            "architecture": None,
            "family": None,
            "reason": "model inspection is unavailable in explicit no-provider mode",
        },
        {
            "record_type": "chat_runtime_context",
            "evidence_kind": "provider_declaration",
            "source": "modelsurgeon.conversation.provider",
            "outcome": ChatInspectionOutcome.UNSUPPORTED.value,
            "revision": None,
            "capability_card": None,
            "live_evidence": False,
            "reason": "no text-model provider was selected",
        },
        hardware,
        {
            "record_type": "chat_capability_context",
            "evidence_kind": "engine_capability_matrix",
            "source": "modelsurgeon.evaluation.architecture_compatibility",
            "outcome": ChatInspectionOutcome.UNKNOWN.value,
            "matrix_id": ARCHITECTURE_COMPATIBILITY_MATRIX.matrix_id,
            "cells": [],
            "live_evidence": False,
            "reason": "model family is unavailable in no-provider mode",
        },
        tuple(sorted(set(warnings))),
    )


__all__ = [
    "CHAT_INSPECTION_SCHEMA_VERSION",
    "ChatInspectionContext",
    "ChatInspectionError",
    "ChatInspectionOutcome",
    "inspect_local_chat_model",
    "no_provider_inspection_context",
]
