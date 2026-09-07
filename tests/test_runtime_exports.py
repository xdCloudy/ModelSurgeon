"""Bounded target-runtime export capability matrix tests."""

from __future__ import annotations

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.runtime_exports import (
    RuntimeExportError,
    RuntimeExportOutcome,
    RuntimeExportRequest,
    RuntimeExportState,
    RuntimeKind,
    generate_runtime_export_matrix,
    plan_runtime_export,
)


def _request(
    runtime: RuntimeKind,
    state: RuntimeExportState,
    *,
    family: ModelFamily = ModelFamily.LLAMA,
    metadata_complete: bool = True,
) -> RuntimeExportRequest:
    return RuntimeExportRequest(
        family,
        runtime,
        state,
        "sha256:" + "1" * 64,
        "converter-v1",
        "runtime-v1",
        ("modelsurgeon", "export", runtime.value),
        "apache-2.0",
        "sha256:" + "2" * 64,
        metadata_complete,
    )


def test_huggingface_verified_and_metadata_unknown_paths_are_explicit() -> None:
    dense = RuntimeExportState("dense")
    verified = plan_runtime_export(_request(RuntimeKind.HUGGING_FACE, dense))
    assert verified.outcome is RuntimeExportOutcome.VERIFIED
    unknown = plan_runtime_export(
        _request(RuntimeKind.HUGGING_FACE, dense, metadata_complete=False)
    )
    assert unknown.outcome is RuntimeExportOutcome.UNKNOWN


def test_unsupported_structural_states_are_not_approximated() -> None:
    state = RuntimeExportState(
        "asymmetric-low-rank", uniform_width=False, asymmetric_width=True, low_rank=True
    )
    cell = plan_runtime_export(_request(RuntimeKind.LLAMA_CPP, state))
    assert cell.outcome is RuntimeExportOutcome.UNSUPPORTED
    assert "asymmetric" in cell.reason


def test_llama_cpp_family_and_other_family_paths_are_distinct() -> None:
    llama = plan_runtime_export(_request(RuntimeKind.LLAMA_CPP, RuntimeExportState("dense")))
    qwen = plan_runtime_export(
        _request(RuntimeKind.LLAMA_CPP, RuntimeExportState("dense"), family=ModelFamily.QWEN)
    )
    mistral = plan_runtime_export(
        _request(RuntimeKind.LLAMA_CPP, RuntimeExportState("dense"), family=ModelFamily.MISTRAL)
    )
    assert llama.outcome is RuntimeExportOutcome.EXPERIMENTAL
    assert qwen.outcome is RuntimeExportOutcome.EXPERIMENTAL
    assert mistral.outcome is RuntimeExportOutcome.UNSUPPORTED


def test_matrix_is_complete_and_deterministic() -> None:
    first = generate_runtime_export_matrix()
    second = generate_runtime_export_matrix()
    assert len(first.cells) == 4 * 5 * 4
    assert first.matrix_id == second.matrix_id
    assert {cell.outcome for cell in first.cells} >= {
        RuntimeExportOutcome.UNKNOWN,
        RuntimeExportOutcome.UNSUPPORTED,
    }


def test_invalid_state_and_positive_cell_evidence_are_rejected() -> None:
    try:
        RuntimeExportState("invalid", uniform_width=True, asymmetric_width=True)
    except RuntimeExportError:
        pass
    else:
        raise AssertionError("conflicting structural state should be rejected")
