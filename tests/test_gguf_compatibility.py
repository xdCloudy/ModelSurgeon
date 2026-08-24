from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf.container import GGUFValueType
from modelsurgeon.evaluation.gguf_compatibility import (
    GGUFCompatibilityCell,
    GGUFCompatibilityError,
    GGUFCompatibilityKey,
    GGUFCompatibilityOperation,
    GGUFCompatibilityStatus,
    GGUFLoadForwardEvidence,
    GGUFStorageProfile,
    build_complete_matrix,
    load_forward_evidence,
)
from modelsurgeon.evaluation.llama_cpp import (
    LLAMA_CPP_VALIDATION_COMMIT,
    LlamaCppGGUFValidationReport,
    LlamaCppToolProvenance,
)


def _key(
    operation: GGUFCompatibilityOperation = GGUFCompatibilityOperation.LOAD_FORWARD,
) -> GGUFCompatibilityKey:
    return GGUFCompatibilityKey(ModelFamily.LLAMA, GGUFStorageProfile.F16, operation)


def _evidence(*, returncode: int | None = 0) -> GGUFLoadForwardEvidence:
    return GGUFLoadForwardEvidence(
        _key(),
        "fixture/model",
        "a" * 40,
        "b" * 64,
        100,
        "llama",
        1,
        LLAMA_CPP_VALIDATION_COMMIT,
        LLAMA_CPP_VALIDATION_COMMIT[:9],
        ("llama-cli", "-m", "fixture.gguf"),
        returncode,
        False,
    )


def test_runtime_claim_requires_successful_matching_evidence() -> None:
    with pytest.raises(GGUFCompatibilityError, match="successful"):
        GGUFCompatibilityCell(
            _key(),
            GGUFCompatibilityStatus.RUNTIME_VERIFIED,
            "claimed",
        )
    with pytest.raises(GGUFCompatibilityError, match="successful"):
        GGUFCompatibilityCell(
            _key(),
            GGUFCompatibilityStatus.RUNTIME_VERIFIED,
            "claimed",
            _evidence(returncode=1),
        )

    cell = GGUFCompatibilityCell(
        _key(),
        GGUFCompatibilityStatus.RUNTIME_VERIFIED,
        "pinned runtime passed",
        _evidence(),
    )

    assert cell.support_claimed
    assert cell.to_record()["evidence"] is not None


def test_non_runtime_statuses_cannot_be_misread_as_support() -> None:
    for status in (
        GGUFCompatibilityStatus.STRUCTURAL_ONLY,
        GGUFCompatibilityStatus.UNSUPPORTED,
        GGUFCompatibilityStatus.FAILED,
    ):
        cell = GGUFCompatibilityCell(_key(), status, "not a support claim")
        assert not cell.support_claimed


def test_storage_profile_rejects_wrong_general_file_type() -> None:
    with pytest.raises(GGUFCompatibilityError, match=r"general\.file_type 15"):
        GGUFLoadForwardEvidence(
            GGUFCompatibilityKey(
                ModelFamily.LLAMA,
                GGUFStorageProfile.Q4_K_M,
                GGUFCompatibilityOperation.LOAD_FORWARD,
            ),
            "fixture/model",
            "a" * 40,
            "b" * 64,
            100,
            "llama",
            1,
            LLAMA_CPP_VALIDATION_COMMIT,
            LLAMA_CPP_VALIDATION_COMMIT[:9],
            ("llama-cli",),
            0,
            False,
        )


def test_matrix_requires_every_family_profile_operation_cell() -> None:
    cells = (
        GGUFCompatibilityCell(
            _key(), GGUFCompatibilityStatus.RUNTIME_VERIFIED, "passed", _evidence()
        ),
        GGUFCompatibilityCell(
            _key(GGUFCompatibilityOperation.DISCOVERY),
            GGUFCompatibilityStatus.STRUCTURAL_ONLY,
            "parser and discovery unit tests only",
        ),
    )
    matrix = build_complete_matrix(
        families=(ModelFamily.LLAMA,),
        storage_profiles=(GGUFStorageProfile.F16,),
        operations=(
            GGUFCompatibilityOperation.LOAD_FORWARD,
            GGUFCompatibilityOperation.DISCOVERY,
        ),
        cells=cells,
    )
    assert len(matrix.to_record()["cells"]) == 2  # type: ignore[arg-type]

    with pytest.raises(GGUFCompatibilityError, match="coverage mismatch"):
        build_complete_matrix(
            families=(ModelFamily.LLAMA,),
            storage_profiles=(GGUFStorageProfile.F16,),
            operations=(
                GGUFCompatibilityOperation.LOAD_FORWARD,
                GGUFCompatibilityOperation.DISCOVERY,
            ),
            cells=cells[:1],
        )


def test_matrix_fails_scheduled_gate_on_runtime_failure() -> None:
    matrix = build_complete_matrix(
        families=(ModelFamily.LLAMA,),
        storage_profiles=(GGUFStorageProfile.F16,),
        operations=(GGUFCompatibilityOperation.LOAD_FORWARD,),
        cells=(
            GGUFCompatibilityCell(
                _key(),
                GGUFCompatibilityStatus.FAILED,
                "upstream runtime rejected fixture",
                _evidence(returncode=2),
            ),
        ),
    )
    with pytest.raises(GGUFCompatibilityError, match="llama/F16/load_forward"):
        matrix.require_no_failures()


def test_report_binding_reads_metadata_and_hashes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"fixture payload")
    tool = LlamaCppToolProvenance(
        Path("llama-cli"),
        LLAMA_CPP_VALIDATION_COMMIT,
        LLAMA_CPP_VALIDATION_COMMIT[:9],
        ("llama-cli", "--version"),
        "version",
    )
    report = LlamaCppGGUFValidationReport(
        model,
        tool,
        ("llama-cli", "-m", str(model)),
        0,
        False,
        "ok",
        "",
        False,
        False,
    )

    class _Entry:
        def __init__(self, value: str | int, value_type: object) -> None:
            self.value = value
            self.value_type = value_type

    class _Container:
        @staticmethod
        def metadata_entry(key: str) -> _Entry | None:
            entries = {
                "general.architecture": _Entry("llama", GGUFValueType.STRING),
                "general.file_type": _Entry(1, GGUFValueType.UINT32),
            }
            return entries.get(key)

    class _Opened:
        container = _Container()

        def __enter__(self) -> _Opened:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "modelsurgeon.evaluation.gguf_compatibility.open_gguf", lambda _path: _Opened()
    )
    evidence = load_forward_evidence(
        report,
        family=ModelFamily.LLAMA,
        storage_profile=GGUFStorageProfile.F16,
        source_identifier="fixture/model",
        source_revision="a" * 40,
    )

    assert evidence.successful
    assert evidence.artifact_sha256 == (
        "c51d9cab89aec430d3274a51bbb930e124935ca08ab6f20d46e1c44651f46497"
    )
