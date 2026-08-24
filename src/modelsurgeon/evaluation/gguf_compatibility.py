"""Auditable GGUF family, storage-profile, and operation compatibility claims."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import cast

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.adapters.gguf.architecture import resolve_gguf_architecture
from modelsurgeon.adapters.gguf.container import GGUFValueType, open_gguf
from modelsurgeon.evaluation.llama_cpp import (
    LLAMA_CPP_VALIDATION_COMMIT,
    LlamaCppGGUFValidationReport,
)

GGUF_COMPATIBILITY_SCHEMA_VERSION = 1


class GGUFCompatibilityError(ValueError):
    """Raised when a compatibility claim is incomplete or contradictory."""


class GGUFStorageProfile(StrEnum):
    """Whole-file llama.cpp quantization recipes, not individual tensor types."""

    F16 = "F16"
    Q4_K_M = "Q4_K_M"


class GGUFCompatibilityOperation(StrEnum):
    LOAD_FORWARD = "load_forward"
    DISCOVERY = "discovery"
    MLP_REMOVAL = "mlp_removal"
    ATTENTION_REMOVAL = "attention_removal"
    LAYER_REMOVAL = "layer_removal"
    LOW_RANK_REPLACEMENT = "low_rank_replacement"


class GGUFCompatibilityStatus(StrEnum):
    RUNTIME_VERIFIED = "runtime_verified"
    STRUCTURAL_ONLY = "structural_only"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


_PROFILE_FILE_TYPES = {
    GGUFStorageProfile.F16: 1,
    GGUFStorageProfile.Q4_K_M: 15,
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GGUFCompatibilityKey:
    family: ModelFamily
    storage_profile: GGUFStorageProfile
    operation: GGUFCompatibilityOperation

    def to_record(self) -> dict[str, str]:
        return {
            "family": self.family.value,
            "storage_profile": self.storage_profile.value,
            "operation": self.operation.value,
        }


@dataclass(frozen=True, slots=True)
class GGUFLoadForwardEvidence:
    key: GGUFCompatibilityKey
    source_identifier: str
    source_revision: str
    artifact_sha256: str
    artifact_size_bytes: int
    gguf_architecture: str
    gguf_file_type: int
    llama_cpp_expected_revision: str
    llama_cpp_reported_revision: str
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool

    def __post_init__(self) -> None:
        if self.key.operation is not GGUFCompatibilityOperation.LOAD_FORWARD:
            raise GGUFCompatibilityError("llama.cpp evidence is only valid for load_forward")
        if not self.source_identifier or re.fullmatch(
            r"[0-9a-f]{40}", self.source_revision
        ) is None:
            raise GGUFCompatibilityError("fixture source identifier and revision are required")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise GGUFCompatibilityError("artifact SHA-256 must be lowercase hexadecimal")
        if self.artifact_size_bytes <= 0 or not self.gguf_architecture:
            raise GGUFCompatibilityError("artifact size and GGUF architecture are required")
        expected_file_type = _PROFILE_FILE_TYPES[self.key.storage_profile]
        if self.gguf_file_type != expected_file_type:
            raise GGUFCompatibilityError(
                f"{self.key.storage_profile.value} requires general.file_type "
                f"{expected_file_type}, found {self.gguf_file_type}"
            )
        if not self.command:
            raise GGUFCompatibilityError("validation command is required")
        expected = self.llama_cpp_expected_revision.lower()
        reported = self.llama_cpp_reported_revision.lower()
        if (
            re.fullmatch(r"[0-9a-f]{7,40}", expected) is None
            or re.fullmatch(r"[0-9a-f]{7,40}", reported) is None
            or not (expected.startswith(reported) or reported.startswith(expected))
        ):
            raise GGUFCompatibilityError("llama.cpp expected and reported revisions disagree")

    @property
    def successful(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def to_record(self) -> dict[str, object]:
        return {
            **self.key.to_record(),
            "source_identifier": self.source_identifier,
            "source_revision": self.source_revision,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "gguf_architecture": self.gguf_architecture,
            "gguf_file_type": self.gguf_file_type,
            "llama_cpp_expected_revision": self.llama_cpp_expected_revision,
            "llama_cpp_reported_revision": self.llama_cpp_reported_revision,
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "successful": self.successful,
        }


def load_forward_evidence(
    report: LlamaCppGGUFValidationReport,
    *,
    family: ModelFamily,
    storage_profile: GGUFStorageProfile,
    source_identifier: str,
    source_revision: str,
) -> GGUFLoadForwardEvidence:
    """Bind a llama.cpp report to immutable artifact and GGUF metadata evidence."""

    model_path = report.model_path.resolve()
    with open_gguf(model_path) as opened:
        container = opened.container
        architecture_entry = container.metadata_entry("general.architecture")
        file_type_entry = container.metadata_entry("general.file_type")
        if architecture_entry is None or architecture_entry.value_type is not GGUFValueType.STRING:
            raise GGUFCompatibilityError("fixture requires string general.architecture metadata")
        if file_type_entry is None or file_type_entry.value_type not in {
            GGUFValueType.UINT32,
            GGUFValueType.UINT64,
        }:
            raise GGUFCompatibilityError("fixture requires integer general.file_type metadata")
        architecture = str(architecture_entry.value)
        resolve_gguf_architecture(architecture, family=family)
        file_type = cast(int, file_type_entry.value)

    return GGUFLoadForwardEvidence(
        GGUFCompatibilityKey(
            family,
            storage_profile,
            GGUFCompatibilityOperation.LOAD_FORWARD,
        ),
        source_identifier,
        source_revision,
        _sha256(model_path),
        model_path.stat().st_size,
        architecture,
        file_type,
        report.tool.expected_revision,
        report.tool.reported_revision,
        report.command,
        report.returncode,
        report.timed_out,
    )


@dataclass(frozen=True, slots=True)
class GGUFCompatibilityCell:
    key: GGUFCompatibilityKey
    status: GGUFCompatibilityStatus
    reason: str
    evidence: GGUFLoadForwardEvidence | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise GGUFCompatibilityError("every compatibility cell requires a reason")
        if self.status is GGUFCompatibilityStatus.RUNTIME_VERIFIED:
            if self.evidence is None or not self.evidence.successful:
                raise GGUFCompatibilityError(
                    "runtime_verified requires successful pinned llama.cpp evidence"
                )
            if self.evidence.key != self.key:
                raise GGUFCompatibilityError("compatibility evidence key does not match its cell")
            if self.evidence.llama_cpp_expected_revision != LLAMA_CPP_VALIDATION_COMMIT:
                raise GGUFCompatibilityError("compatibility evidence uses an unpinned llama.cpp")
        elif self.evidence is not None:
            if self.status is not GGUFCompatibilityStatus.FAILED:
                raise GGUFCompatibilityError(
                    "only runtime_verified or failed cells may contain runtime evidence"
                )
            if self.evidence.key != self.key or self.evidence.successful:
                raise GGUFCompatibilityError("failed cell requires matching unsuccessful evidence")

    @property
    def support_claimed(self) -> bool:
        return self.status is GGUFCompatibilityStatus.RUNTIME_VERIFIED

    def to_record(self) -> dict[str, object]:
        return {
            **self.key.to_record(),
            "status": self.status.value,
            "support_claimed": self.support_claimed,
            "reason": self.reason,
            "evidence": None if self.evidence is None else self.evidence.to_record(),
        }


@dataclass(frozen=True, slots=True)
class GGUFCompatibilityMatrix:
    families: tuple[ModelFamily, ...]
    storage_profiles: tuple[GGUFStorageProfile, ...]
    operations: tuple[GGUFCompatibilityOperation, ...]
    cells: tuple[GGUFCompatibilityCell, ...]
    schema_version: int = GGUF_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        dimensions: tuple[tuple[object, ...], ...] = (
            self.families,
            self.storage_profiles,
            self.operations,
        )
        if any(not dimension or len(set(dimension)) != len(dimension) for dimension in dimensions):
            raise GGUFCompatibilityError("matrix dimensions must be non-empty and unique")
        keys = tuple(cell.key for cell in self.cells)
        if len(set(keys)) != len(keys):
            raise GGUFCompatibilityError("compatibility matrix contains duplicate cells")
        expected = {
            GGUFCompatibilityKey(family, profile, operation)
            for family, profile, operation in product(
                self.families, self.storage_profiles, self.operations
            )
        }
        actual = set(keys)
        if actual != expected:
            missing = sorted(
                (key.to_record() for key in expected - actual),
                key=lambda item: tuple(item.values()),
            )
            extra = sorted(
                (key.to_record() for key in actual - expected),
                key=lambda item: tuple(item.values()),
            )
            raise GGUFCompatibilityError(
                f"compatibility matrix coverage mismatch: missing={missing}, extra={extra}"
            )

    def failed_cells(self) -> tuple[GGUFCompatibilityCell, ...]:
        return tuple(cell for cell in self.cells if cell.status is GGUFCompatibilityStatus.FAILED)

    def require_no_failures(self) -> None:
        failures = self.failed_cells()
        if failures:
            keys = ", ".join(
                "/".join(cell.key.to_record().values()) for cell in failures
            )
            raise GGUFCompatibilityError(f"runtime compatibility failures: {keys}")

    def to_record(self) -> dict[str, object]:
        ordered = sorted(
            self.cells,
            key=lambda cell: (
                cell.key.family.value,
                cell.key.storage_profile.value,
                cell.key.operation.value,
            ),
        )
        return {
            "schema_version": self.schema_version,
            "pinned_llama_cpp_revision": LLAMA_CPP_VALIDATION_COMMIT,
            "families": [family.value for family in self.families],
            "storage_profiles": [profile.value for profile in self.storage_profiles],
            "operations": [operation.value for operation in self.operations],
            "cells": [cell.to_record() for cell in ordered],
        }


def build_complete_matrix(
    *,
    families: Iterable[ModelFamily],
    storage_profiles: Iterable[GGUFStorageProfile],
    operations: Iterable[GGUFCompatibilityOperation],
    cells: Iterable[GGUFCompatibilityCell],
) -> GGUFCompatibilityMatrix:
    """Freeze and validate a complete compatibility matrix."""

    return GGUFCompatibilityMatrix(
        tuple(families),
        tuple(storage_profiles),
        tuple(operations),
        tuple(cells),
    )
