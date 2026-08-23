"""Tests for transactional streaming GGUF output construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import (
    GGUFDiskEstimate,
    GGUFValueType,
    GGUFWriteError,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    open_gguf,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder


def _metadata() -> tuple[GGUFWriteMetadata, ...]:
    return (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "test"),
        GGUFWriteMetadata(
            "test.labels",
            GGUFValueType.ARRAY,
            ("a", "b"),
            GGUFValueType.STRING,
        ),
    )


def _plan(tmp_path: Path, tensors: tuple[GGUFWriteTensor, ...]):
    layout = plan_gguf_output(_metadata(), tensors)
    disk = preflight_gguf_disk(
        tmp_path / "output.gguf",
        tmp_path,
        GGUFDiskEstimate(layout.total_bytes, 0, alignment_bytes=layout.alignment),
    )
    return layout, disk


def test_streams_aligned_tensors_and_publishes_only_after_finalize(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"

    def second_chunks():
        assert not destination.exists()
        yield b"ef"
        yield b"ghijkl"

    tensors = (
        GGUFWriteTensor("first.weight", (2,), 0, (b"ab", b"cdefgh")),
        GGUFWriteTensor("second.weight", (2,), 0, second_chunks()),
    )
    layout, disk = _plan(tmp_path, tensors)

    result = write_gguf_transactionally(destination, _metadata(), tensors, disk)

    assert destination.exists()
    assert result.file_size == layout.total_bytes
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result.tensors[0].data_offset % 32 == 0
    assert result.tensors[1].data_offset % 32 == 0
    with open_gguf(destination) as written:
        assert written.container.version == 3
        assert written.container.metadata_entry("test.labels").value == ("a", "b")  # type: ignore[union-attr]
        assert tuple(tensor.relative_offset for tensor in written.container.tensors) == (
            0,
            32,
        )


def test_short_tensor_removes_staging_file_and_never_publishes(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"
    tensors = (GGUFWriteTensor("short.weight", (2,), 0, (b"short",)),)
    _, disk = _plan(tmp_path, tensors)

    with pytest.raises(GGUFWriteError, match="supplied 5 bytes; expected 8"):
        write_gguf_transactionally(destination, _metadata(), tensors, disk)

    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"
    destination.write_bytes(b"original")
    tensors = (GGUFWriteTensor("tensor.weight", (2,), 0, (b"12345678",)),)
    layout = plan_gguf_output(_metadata(), tensors)
    disk = preflight_gguf_disk(
        destination,
        tmp_path,
        GGUFDiskEstimate(layout.total_bytes, 0),
    )

    with pytest.raises(GGUFWriteError, match="already exists"):
        write_gguf_transactionally(destination, _metadata(), tensors, disk)

    assert destination.read_bytes() == b"original"


def test_big_endian_v3_layout_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "output.gguf"
    tensors = (GGUFWriteTensor("tensor.weight", (2,), 0, (b"12345678",)),)
    layout = plan_gguf_output(_metadata(), tensors, byte_order=ByteOrder.BIG)
    disk = preflight_gguf_disk(
        destination,
        tmp_path,
        GGUFDiskEstimate(layout.total_bytes, 0),
    )

    write_gguf_transactionally(
        destination,
        _metadata(),
        tensors,
        disk,
        byte_order=ByteOrder.BIG,
    )

    with open_gguf(destination) as written:
        assert written.container.byte_order is ByteOrder.BIG


def test_layout_rejects_duplicate_names_and_mismatched_alignment_metadata() -> None:
    tensor = GGUFWriteTensor("same", (2,), 0, (b"12345678",))
    with pytest.raises(GGUFWriteError, match="tensor names"):
        plan_gguf_output(_metadata(), (tensor, tensor))
    with pytest.raises(GGUFWriteError, match="must match"):
        plan_gguf_output(
            (GGUFWriteMetadata("general.alignment", GGUFValueType.UINT32, 64),),
            (tensor,),
            alignment=32,
        )
