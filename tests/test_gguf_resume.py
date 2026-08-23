"""Tests for tensor-boundary resumable GGUF output construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import (
    GGUFDiskEstimate,
    GGUFResumeError,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    discard_resumable_gguf,
    open_gguf,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_resumably,
)

_INPUT = "a" * 64
_METADATA = (GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "test"),)


def _tensors(first: object = (b"12345678",)) -> tuple[GGUFWriteTensor, ...]:
    return (
        GGUFWriteTensor("first.weight", (2,), 0, first),  # type: ignore[arg-type]
        GGUFWriteTensor("second.weight", (2,), 0, (b"abcdefgh",)),
    )


def _disk(path: Path, tensors: tuple[GGUFWriteTensor, ...]):
    layout = plan_gguf_output(_METADATA, tensors)
    return preflight_gguf_disk(
        path, path.parent, GGUFDiskEstimate(layout.total_bytes, 0)
    )


def test_power_loss_checkpoint_never_publishes_and_resume_skips_completed_tensor(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.gguf"
    initial = _tensors()

    def interrupt(index: int) -> None:
        if index == 0:
            raise RuntimeError("simulated power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        write_gguf_resumably(
            destination,
            _METADATA,
            initial,
            _disk(destination, initial),
            input_sha256=_INPUT,
            checkpoint_hook=interrupt,
        )
    assert not destination.exists()

    def completed_must_not_be_read():
        raise AssertionError("completed tensor was read again")
        yield b""  # pragma: no cover

    resumed_tensors = _tensors(completed_must_not_be_read())
    result = write_gguf_resumably(
        destination,
        _METADATA,
        resumed_tensors,
        _disk(destination, resumed_tensors),
        input_sha256=_INPUT,
    )

    assert destination.exists()
    assert tuple(name for name, _ in result.tensor_sha256) == (
        "first.weight",
        "second.weight",
    )
    with open_gguf(destination) as output:
        assert len(output.container.tensors) == 2


def test_partial_tensor_is_truncated_to_last_committed_boundary(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"

    def broken():
        yield b"1234"
        raise RuntimeError("lost power mid-tensor")

    interrupted = _tensors(broken())
    with pytest.raises(RuntimeError, match="mid-tensor"):
        write_gguf_resumably(
            destination,
            _METADATA,
            interrupted,
            _disk(destination, interrupted),
            input_sha256=_INPUT,
        )

    fresh = _tensors()
    write_gguf_resumably(
        destination,
        _METADATA,
        fresh,
        _disk(destination, fresh),
        input_sha256=_INPUT,
    )
    assert destination.exists()


def test_changed_input_or_plan_invalidates_resume_without_publication(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"
    tensors = _tensors()

    def interrupt(index: int) -> None:
        raise RuntimeError(index)

    with pytest.raises(RuntimeError):
        write_gguf_resumably(
            destination,
            _METADATA,
            tensors,
            _disk(destination, tensors),
            input_sha256=_INPUT,
            checkpoint_hook=interrupt,
        )
    with pytest.raises(GGUFResumeError, match="changed input"):
        write_gguf_resumably(
            destination,
            _METADATA,
            _tensors(),
            _disk(destination, _tensors()),
            input_sha256="b" * 64,
        )
    changed = (
        GGUFWriteTensor("renamed.weight", (2,), 0, (b"12345678",)),
        GGUFWriteTensor("second.weight", (2,), 0, (b"abcdefgh",)),
    )
    with pytest.raises(GGUFResumeError, match="output plan"):
        write_gguf_resumably(
            destination,
            _METADATA,
            changed,
            _disk(destination, changed),
            input_sha256=_INPUT,
        )
    assert not destination.exists()


def test_explicit_discard_removes_only_resume_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"
    keep = tmp_path / "keep"
    keep.write_text("safe", encoding="utf-8")
    tensors = _tensors()

    def interrupt(index: int) -> None:
        raise RuntimeError(index)

    with pytest.raises(RuntimeError):
        write_gguf_resumably(
            destination,
            _METADATA,
            tensors,
            _disk(destination, tensors),
            input_sha256=_INPUT,
            checkpoint_hook=interrupt,
        )
    discard_resumable_gguf(destination)

    assert not list(tmp_path.glob(".*modelsurgeon.resume*"))
    assert keep.read_text(encoding="utf-8") == "safe"
