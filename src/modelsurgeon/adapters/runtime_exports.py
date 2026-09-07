"""Bounded runtime-export capability matrix and evidence contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.experiments.identity import canonical_identity_json

RUNTIME_EXPORT_SCHEMA_VERSION = 1


class RuntimeExportError(ValueError):
    """Raised when an export capability claim is incomplete or unsafe."""


class RuntimeKind(StrEnum):
    HUGGING_FACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    MLX = "mlx"
    ONNX = "onnx"
    VLLM = "vllm"


class RuntimeExportOutcome(StrEnum):
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeExportState:
    """Architecture traits that materially affect target-runtime representability."""

    name: str
    uniform_width: bool = True
    asymmetric_width: bool = False
    low_rank: bool = False
    quantized: bool = False
    mixed_codec: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RuntimeExportError("runtime export state requires a name")
        if self.uniform_width and self.asymmetric_width:
            raise RuntimeExportError("a state cannot be both uniform and asymmetric")
        if self.mixed_codec and not self.quantized:
            raise RuntimeExportError("mixed codecs require a quantized state")

    @property
    def state_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode()
        ).hexdigest()
        return f"state_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "uniform_width": self.uniform_width,
            "asymmetric_width": self.asymmetric_width,
            "low_rank": self.low_rank,
            "quantized": self.quantized,
            "mixed_codec": self.mixed_codec,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["state_id"] = self.state_id
        return record


@dataclass(frozen=True, slots=True)
class RuntimeExportRequest:
    family: ModelFamily
    runtime: RuntimeKind
    state: RuntimeExportState
    source_artifact_digest: str
    converter_revision: str
    runtime_revision: str
    command: tuple[str, ...]
    license_id: str
    config_digest: str
    metadata_complete: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("source artifact digest", self.source_artifact_digest),
            ("converter revision", self.converter_revision),
            ("runtime revision", self.runtime_revision),
            ("license", self.license_id),
            ("config digest", self.config_digest),
        ):
            if not value.strip():
                raise RuntimeExportError(f"{label} is required")
        if not self.command or any(not value.strip() for value in self.command):
            raise RuntimeExportError("runtime export command is required")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_EXPORT_SCHEMA_VERSION,
            "family": self.family.value,
            "runtime": self.runtime.value,
            "state": self.state.to_record(),
            "source_artifact_digest": self.source_artifact_digest,
            "converter_revision": self.converter_revision,
            "runtime_revision": self.runtime_revision,
            "command": list(self.command),
            "license_id": self.license_id,
            "config_digest": self.config_digest,
            "metadata_complete": self.metadata_complete,
        }


@dataclass(frozen=True, slots=True)
class RuntimeExportCell:
    family: ModelFamily
    runtime: RuntimeKind
    state: RuntimeExportState
    outcome: RuntimeExportOutcome
    reason: str
    evidence: tuple[str, ...] = ()
    source_artifact_digest: str | None = None
    converter_revision: str | None = None
    runtime_revision: str | None = None
    command: tuple[str, ...] = ()
    license_id: str | None = None
    config_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise RuntimeExportError("runtime export cells require a reason")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise RuntimeExportError("runtime export evidence must be sorted and unique")
        if self.outcome in {RuntimeExportOutcome.VERIFIED, RuntimeExportOutcome.EXPERIMENTAL}:
            if not self.evidence or not self.source_artifact_digest:
                raise RuntimeExportError("positive export outcomes require retained evidence")
            if not self.converter_revision or not self.runtime_revision or not self.config_digest:
                raise RuntimeExportError(
                    "positive export outcomes require version and config provenance"
                )

    @property
    def cell_id(self) -> str:
        return f"{self.family.value}:{self.runtime.value}:{self.state.state_id}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_EXPORT_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "family": self.family.value,
            "runtime": self.runtime.value,
            "state": self.state.to_record(),
            "outcome": self.outcome.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "source_artifact_digest": self.source_artifact_digest,
            "converter_revision": self.converter_revision,
            "runtime_revision": self.runtime_revision,
            "command": list(self.command),
            "license_id": self.license_id,
            "config_digest": self.config_digest,
        }


def plan_runtime_export(request: RuntimeExportRequest) -> RuntimeExportCell:
    """Plan one export cell and fail closed on unproven target representation."""

    state = request.state
    evidence = tuple(
        sorted(
            (
                "tests/test_runtime_exports.py",
                f"converter:{request.converter_revision}",
                f"runtime:{request.runtime_revision}",
            )
        )
    )
    if request.runtime is RuntimeKind.HUGGING_FACE:
        if not request.metadata_complete:
            return RuntimeExportCell(
                request.family,
                request.runtime,
                state,
                RuntimeExportOutcome.UNKNOWN,
                "tokenizer, generation, or config metadata is incomplete",
            )
        return RuntimeExportCell(
            request.family,
            request.runtime,
            state,
            RuntimeExportOutcome.VERIFIED,
            "native Transformers metadata and save/reload contract is available",
            evidence,
            request.source_artifact_digest,
            request.converter_revision,
            request.runtime_revision,
            request.command,
            request.license_id,
            request.config_digest,
        )
    if state.asymmetric_width or state.low_rank or state.mixed_codec:
        return RuntimeExportCell(
            request.family,
            request.runtime,
            state,
            RuntimeExportOutcome.UNSUPPORTED,
            "target runtime has no verified representation for asymmetric, "
            "low-rank, or mixed-codec state",
        )
    if request.runtime is RuntimeKind.LLAMA_CPP and request.family in {
        ModelFamily.LLAMA,
        ModelFamily.QWEN,
    }:
        return RuntimeExportCell(
            request.family,
            request.runtime,
            state,
            RuntimeExportOutcome.EXPERIMENTAL,
            "pinned llama.cpp conversion and reload require target-runtime smoke evidence",
            evidence,
            request.source_artifact_digest,
            request.converter_revision,
            request.runtime_revision,
            request.command,
            request.license_id,
            request.config_digest,
        )
    if request.runtime is RuntimeKind.LLAMA_CPP:
        return RuntimeExportCell(
            request.family,
            request.runtime,
            state,
            RuntimeExportOutcome.UNSUPPORTED,
            "family is outside the pinned llama.cpp export capability matrix",
        )
    return RuntimeExportCell(
        request.family,
        request.runtime,
        state,
        RuntimeExportOutcome.UNKNOWN,
        "target runtime adapter requires family-specific reload evidence",
    )


@dataclass(frozen=True, slots=True)
class RuntimeExportMatrix:
    cells: tuple[RuntimeExportCell, ...]
    schema_version: int = RUNTIME_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        keys = [(cell.family, cell.runtime, cell.state.state_id) for cell in self.cells]
        if len(keys) != len(set(keys)):
            raise RuntimeExportError("runtime export matrix cells must be unique")

    @property
    def matrix_id(self) -> str:
        return hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode()
        ).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        ordered = sorted(self.cells, key=lambda cell: cell.cell_id)
        return {
            "schema_version": self.schema_version,
            "cells": [cell.to_record() for cell in ordered],
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["matrix_id"] = self.matrix_id
        return record


DEFAULT_EXPORT_STATES = (
    RuntimeExportState("dense_uniform"),
    RuntimeExportState("asymmetric_width", uniform_width=False, asymmetric_width=True),
    RuntimeExportState("low_rank_quantized", low_rank=True, quantized=True),
    RuntimeExportState("mixed_codec_quantized", quantized=True, mixed_codec=True),
)


def generate_runtime_export_matrix(
    *,
    families: Iterable[ModelFamily] = tuple(ModelFamily),
    runtimes: Iterable[RuntimeKind] = tuple(RuntimeKind),
    states: Iterable[RuntimeExportState] = DEFAULT_EXPORT_STATES,
) -> RuntimeExportMatrix:
    """Generate every family/runtime/state cell with explicit non-support."""

    cells: list[RuntimeExportCell] = []
    for family, runtime, state in product(families, runtimes, states):
        request = RuntimeExportRequest(
            family,
            runtime,
            state,
            "source-unmeasured",
            "converter-unpinned",
            "runtime-unpinned",
            ("modelsurgeon", "export"),
            "unknown-license",
            "config-unresolved",
        metadata_complete=runtime is not RuntimeKind.HUGGING_FACE,
        )
        cells.append(plan_runtime_export(request))
    return RuntimeExportMatrix(tuple(cells))


__all__ = [
    "DEFAULT_EXPORT_STATES",
    "RUNTIME_EXPORT_SCHEMA_VERSION",
    "RuntimeExportCell",
    "RuntimeExportError",
    "RuntimeExportMatrix",
    "RuntimeExportOutcome",
    "RuntimeExportRequest",
    "RuntimeExportState",
    "RuntimeKind",
    "generate_runtime_export_matrix",
    "plan_runtime_export",
]
