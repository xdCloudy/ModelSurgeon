from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.evaluation.architecture_compatibility import (
    ARCHITECTURE_COMPATIBILITY_MATRIX,
    ArchitectureCompatibilityCell,
    ArchitectureCompatibilityError,
    ArchitectureOperation,
    ArchitectureProfile,
    ArchitectureSupportStatus,
)


def _cell(
    family: ModelFamily,
    profile: ArchitectureProfile,
    operation: ArchitectureOperation,
) -> ArchitectureCompatibilityCell:
    return next(
        cell
        for cell in ARCHITECTURE_COMPATIBILITY_MATRIX.cells
        if (cell.family, cell.profile, cell.operation) == (family, profile, operation)
    )


def test_matrix_is_complete_and_unknown_never_means_supported() -> None:
    matrix = ARCHITECTURE_COMPATIBILITY_MATRIX
    assert len(matrix.cells) == 4 * 3 * 9
    assert len(matrix.matrix_id) == 64
    assert Counter(cell.status for cell in matrix.cells) == {
        ArchitectureSupportStatus.EXPERIMENTAL: 57,
        ArchitectureSupportStatus.UNSUPPORTED: 32,
        ArchitectureSupportStatus.VERIFIED: 16,
        ArchitectureSupportStatus.UNKNOWN: 3,
    }
    assert all(
        not cell.supported
        for cell in matrix.cells
        if cell.status
        in {
            ArchitectureSupportStatus.UNKNOWN,
            ArchitectureSupportStatus.UNSUPPORTED,
            ArchitectureSupportStatus.EXPERIMENTAL,
        }
    )


def test_verified_and_experimental_cells_require_test_evidence() -> None:
    assert all(
        cell.evidence
        for cell in ARCHITECTURE_COMPATIBILITY_MATRIX.cells
        if cell.status
        in {ArchitectureSupportStatus.VERIFIED, ArchitectureSupportStatus.EXPERIMENTAL}
    )
    with pytest.raises(ArchitectureCompatibilityError, match="evidence"):
        ArchitectureCompatibilityCell(
            ModelFamily.LLAMA,
            ArchitectureProfile.HF_DENSE,
            ArchitectureOperation.ANALYSIS,
            ArchitectureSupportStatus.VERIFIED,
            "overclaim",
        )


def test_real_proofs_and_explicit_native_boundaries_are_reflected() -> None:
    q4_mlp = _cell(
        ModelFamily.LLAMA,
        ArchitectureProfile.GGUF_Q4_K_M,
        ArchitectureOperation.PHYSICAL_MLP,
    )
    assert q4_mlp.supported
    assert "v0.7-native-q4-k-m-mlp-proof.md" in q4_mlp.evidence[1]

    for family in (ModelFamily.MISTRAL, ModelFamily.GEMMA):
        assert (
            _cell(
                family,
                ArchitectureProfile.GGUF_F16,
                ArchitectureOperation.PHYSICAL_ATTENTION,
            ).status
            is ArchitectureSupportStatus.UNSUPPORTED
        )


def test_missing_q4_family_runtime_evidence_remains_unknown() -> None:
    for family in (ModelFamily.MISTRAL, ModelFamily.QWEN, ModelFamily.GEMMA):
        cell = _cell(
            family,
            ArchitectureProfile.GGUF_Q4_K_M,
            ArchitectureOperation.ANALYSIS,
        )
        assert cell.status is ArchitectureSupportStatus.UNKNOWN
        assert not cell.supported


def test_published_table_is_generated_from_the_matrix() -> None:
    documentation = (
        Path(__file__).parents[1] / "docs" / "architecture-compatibility.md"
    ).read_text(encoding="utf-8")
    generated = documentation.split("<!-- BEGIN GENERATED MATRIX -->\n", 1)[1].split(
        "\n<!-- END GENERATED MATRIX -->", 1
    )[0]
    assert generated == ARCHITECTURE_COMPATIBILITY_MATRIX.markdown_table()
