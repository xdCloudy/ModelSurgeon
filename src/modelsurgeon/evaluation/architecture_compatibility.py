"""Generated HF and GGUF architecture capability matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.adapters import ModelFamily, ModelFormat

ARCHITECTURE_COMPATIBILITY_SCHEMA_VERSION = 1


class ArchitectureCompatibilityError(ValueError):
    """Raised when a compatibility cell is incomplete or overclaims support."""


class ArchitectureProfile(StrEnum):
    HF_DENSE = "HF dense"
    GGUF_F16 = "GGUF F16"
    GGUF_Q4_K_M = "GGUF Q4_K_M"

    @property
    def model_format(self) -> ModelFormat:
        return ModelFormat.HUGGING_FACE if self is self.HF_DENSE else ModelFormat.GGUF


class ArchitectureOperation(StrEnum):
    ANALYSIS = "analysis"
    MLP_MASK = "MLP mask"
    ATTENTION_MASK = "attention mask"
    LAYER_MASK = "layer mask"
    PHYSICAL_MLP = "physical MLP"
    PHYSICAL_ATTENTION = "physical attention"
    PHYSICAL_LAYER = "physical layer"
    LOW_RANK = "low rank"
    CHECKPOINT_WRITE = "checkpoint write"


class ArchitectureSupportStatus(StrEnum):
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ArchitectureCompatibilityCell:
    family: ModelFamily
    profile: ArchitectureProfile
    operation: ArchitectureOperation
    status: ArchitectureSupportStatus
    reason: str
    evidence: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ArchitectureCompatibilityError("compatibility cells require a reason")
        if self.status in {
            ArchitectureSupportStatus.VERIFIED,
            ArchitectureSupportStatus.EXPERIMENTAL,
        } and not self.evidence:
            raise ArchitectureCompatibilityError(
                "verified and experimental cells require automated-test evidence"
            )
        if len(set(self.evidence)) != len(self.evidence) or len(set(self.constraints)) != len(
            self.constraints
        ):
            raise ArchitectureCompatibilityError("cell evidence and constraints must be unique")

    @property
    def supported(self) -> bool:
        return self.status is ArchitectureSupportStatus.VERIFIED

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "format": self.profile.model_format.value,
            "profile": self.profile.value,
            "operation": self.operation.value,
            "status": self.status.value,
            "supported": self.supported,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "constraints": list(self.constraints),
        }


_HF_DISCOVERY = "tests/test_huggingface_discovery.py"
_HF_REAL = {
    ArchitectureOperation.PHYSICAL_MLP: "docs/research/v0.7-hf-physical-mlp-evidence.md",
    ArchitectureOperation.PHYSICAL_ATTENTION: (
        "docs/research/v0.7-hf-physical-attention-evidence.md"
    ),
    ArchitectureOperation.PHYSICAL_LAYER: "docs/research/v0.7-hf-physical-layer-evidence.md",
    ArchitectureOperation.LOW_RANK: "docs/research/v0.7-hf-low-rank-evidence.md",
    ArchitectureOperation.CHECKPOINT_WRITE: (
        "docs/research/v0.7-atomic-safetensors-evidence.md"
    ),
}
_HF_TESTS = {
    ArchitectureOperation.MLP_MASK: "tests/test_mlp_channel_mask.py",
    ArchitectureOperation.ATTENTION_MASK: "tests/test_attention_head_mask.py",
    ArchitectureOperation.LAYER_MASK: "tests/test_layer_bypass.py",
    ArchitectureOperation.PHYSICAL_MLP: "tests/test_hf_physical_mlp.py",
    ArchitectureOperation.PHYSICAL_ATTENTION: "tests/test_hf_physical_attention.py",
    ArchitectureOperation.PHYSICAL_LAYER: "tests/test_hf_physical_layers.py",
    ArchitectureOperation.LOW_RANK: "tests/test_hf_low_rank.py",
    ArchitectureOperation.CHECKPOINT_WRITE: "tests/test_safetensors_checkpoint.py",
}


def _hf_cell(
    family: ModelFamily, operation: ArchitectureOperation
) -> ArchitectureCompatibilityCell:
    if operation is ArchitectureOperation.ANALYSIS:
        return ArchitectureCompatibilityCell(
            family,
            ArchitectureProfile.HF_DENSE,
            operation,
            ArchitectureSupportStatus.VERIFIED,
            "family-specific discovery contract has focused fixture coverage",
            (_HF_DISCOVERY,),
        )
    test = _HF_TESTS[operation]
    if family is ModelFamily.LLAMA and operation in _HF_REAL:
        return ArchitectureCompatibilityCell(
            family,
            ArchitectureProfile.HF_DENSE,
            operation,
            ArchitectureSupportStatus.VERIFIED,
            "real SmolLM2 execution and save/reload evidence is recorded",
            (test, _HF_REAL[operation]),
            ("canonical model.layers module layout",),
        )
    return ArchitectureCompatibilityCell(
        family,
        ArchitectureProfile.HF_DENSE,
        operation,
        ArchitectureSupportStatus.EXPERIMENTAL,
        "generic canonical-module implementation has focused tests but no family-specific "
        "real checkpoint proof",
        (test,),
        ("canonical model.layers module layout",),
    )


def _gguf_cell(
    family: ModelFamily,
    profile: ArchitectureProfile,
    operation: ArchitectureOperation,
) -> ArchitectureCompatibilityCell:
    compatibility_test = "tests/test_gguf_compatibility.py"
    if operation is ArchitectureOperation.ANALYSIS:
        verified = profile is ArchitectureProfile.GGUF_F16 or family is ModelFamily.LLAMA
        return ArchitectureCompatibilityCell(
            family,
            profile,
            operation,
            ArchitectureSupportStatus.VERIFIED
            if verified
            else ArchitectureSupportStatus.UNKNOWN,
            "pinned parser/discovery plus llama.cpp load evidence"
            if verified
            else "no pinned Q4_K_M family fixture/load evidence",
            (compatibility_test,) if verified else (),
        )
    if operation in {
        ArchitectureOperation.MLP_MASK,
        ArchitectureOperation.ATTENTION_MASK,
        ArchitectureOperation.LAYER_MASK,
    }:
        return ArchitectureCompatibilityCell(
            family,
            profile,
            operation,
            ArchitectureSupportStatus.UNSUPPORTED,
            "GGUF is an offline container and has no in-runtime masking session",
        )
    if operation in {
        ArchitectureOperation.PHYSICAL_MLP,
        ArchitectureOperation.PHYSICAL_ATTENTION,
    } and family in {ModelFamily.MISTRAL, ModelFamily.GEMMA}:
        return ArchitectureCompatibilityCell(
            family,
            profile,
            operation,
            ArchitectureSupportStatus.UNSUPPORTED,
            "native coupled MLP/attention planners support only Llama and dense Qwen",
        )
    if (
        family is ModelFamily.LLAMA
        and profile is ArchitectureProfile.GGUF_Q4_K_M
        and operation
        in {
            ArchitectureOperation.PHYSICAL_MLP,
            ArchitectureOperation.CHECKPOINT_WRITE,
        }
    ):
        return ArchitectureCompatibilityCell(
            family,
            profile,
            operation,
            ArchitectureSupportStatus.VERIFIED,
            "real model-wide Q4_K_M output passed pinned llama.cpp",
            (
                "tests/test_native_gguf_mlp_execute.py",
                "docs/research/v0.7-native-q4-k-m-mlp-proof.md",
            ),
            ("codec-aligned unchanged-codec edits",),
        )
    tests = {
        ArchitectureOperation.PHYSICAL_MLP: "tests/test_native_gguf_mlp_execute.py",
        ArchitectureOperation.PHYSICAL_ATTENTION: "tests/test_attention_head_execute.py",
        ArchitectureOperation.PHYSICAL_LAYER: "tests/test_native_gguf_layer_execute.py",
        ArchitectureOperation.LOW_RANK: "tests/test_native_gguf_low_rank_execute.py",
        ArchitectureOperation.CHECKPOINT_WRITE: "tests/test_gguf_writer.py",
    }
    return ArchitectureCompatibilityCell(
        family,
        profile,
        operation,
        ArchitectureSupportStatus.EXPERIMENTAL,
        "native structural tests pass but no family/profile post-operation runtime proof",
        (tests[operation],),
        ("exact native codec or copy-only storage layout",),
    )


def generate_architecture_compatibility_cells() -> tuple[ArchitectureCompatibilityCell, ...]:
    """Generate every finite family/profile/operation cell from capability rules."""

    cells: list[ArchitectureCompatibilityCell] = []
    for family, operation in product(ModelFamily, ArchitectureOperation):
        cells.append(_hf_cell(family, operation))
    for family, profile, operation in product(
        ModelFamily,
        (ArchitectureProfile.GGUF_F16, ArchitectureProfile.GGUF_Q4_K_M),
        ArchitectureOperation,
    ):
        cells.append(_gguf_cell(family, profile, operation))
    return tuple(cells)


@dataclass(frozen=True, slots=True)
class ArchitectureCompatibilityMatrix:
    cells: tuple[ArchitectureCompatibilityCell, ...]
    schema_version: int = ARCHITECTURE_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected = {
            (family, profile, operation)
            for family in ModelFamily
            for profile in ArchitectureProfile
            for operation in ArchitectureOperation
        }
        actual = {(cell.family, cell.profile, cell.operation) for cell in self.cells}
        if len(actual) != len(self.cells) or actual != expected:
            raise ArchitectureCompatibilityError(
                "architecture matrix must contain every unique family/profile/operation cell"
            )

    @property
    def matrix_id(self) -> str:
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        ordered = sorted(
            self.cells,
            key=lambda cell: (cell.family.value, cell.profile.value, cell.operation.value),
        )
        return {
            "schema_version": self.schema_version,
            "cells": [cell.to_record() for cell in ordered],
        }

    def markdown_table(self) -> str:
        """Render the compact public table deterministically from matrix cells."""

        symbols = {
            ArchitectureSupportStatus.VERIFIED: "V",
            ArchitectureSupportStatus.EXPERIMENTAL: "E",
            ArchitectureSupportStatus.UNSUPPORTED: "—",
            ArchitectureSupportStatus.UNKNOWN: "?",
        }
        columns = tuple(product(ArchitectureProfile, ModelFamily))
        lines = [
            "| Operation | "
            + " | ".join(f"{profile.value} {family.value}" for profile, family in columns)
            + " |",
            "| --- | " + " | ".join("---" for _ in columns) + " |",
        ]
        indexed = {
            (cell.profile, cell.family, cell.operation): cell for cell in self.cells
        }
        for operation in ArchitectureOperation:
            values = [
                symbols[indexed[(profile, family, operation)].status]
                for profile, family in columns
            ]
            lines.append(f"| {operation.value} | " + " | ".join(values) + " |")
        return "\n".join(lines)


ARCHITECTURE_COMPATIBILITY_MATRIX = ArchitectureCompatibilityMatrix(
    generate_architecture_compatibility_cells()
)
