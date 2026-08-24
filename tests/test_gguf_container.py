"""Tests for bounded, read-only GGUF container indexing."""

from __future__ import annotations

import struct
import sys
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

from modelsurgeon.adapters.gguf import (
    ByteOrder,
    CorruptGGUFError,
    GGMLQuantizationType,
    GGUFParserLimits,
    GGUFResourceLimitError,
    GGUFValueType,
    UnsupportedGGUFTypeError,
    UnsupportedGGUFVersionError,
    open_gguf,
)

Metadata = tuple[str, GGUFValueType, object, GGUFValueType | None]
Tensor = tuple[str, tuple[int, ...], int, int]


def _string(value: str, endian: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(endian + "Q", len(encoded)) + encoded


def _scalar(value_type: GGUFValueType, value: object, endian: str) -> bytes:
    if value_type is GGUFValueType.STRING:
        return _string(str(value), endian)
    codes = {
        GGUFValueType.UINT8: "B",
        GGUFValueType.INT8: "b",
        GGUFValueType.UINT16: "H",
        GGUFValueType.INT16: "h",
        GGUFValueType.UINT32: "I",
        GGUFValueType.INT32: "i",
        GGUFValueType.FLOAT32: "f",
        GGUFValueType.BOOL: "B",
        GGUFValueType.UINT64: "Q",
        GGUFValueType.INT64: "q",
        GGUFValueType.FLOAT64: "d",
    }
    return struct.pack(endian + codes[value_type], value)


def _metadata(entry: Metadata, endian: str) -> bytes:
    key, value_type, value, element_type = entry
    result = _string(key, endian) + struct.pack(endian + "I", value_type)
    if value_type is GGUFValueType.ARRAY:
        assert element_type is not None
        items = tuple(value)  # type: ignore[arg-type]
        result += struct.pack(endian + "IQ", element_type, len(items))
        result += b"".join(_scalar(element_type, item, endian) for item in items)
        return result
    return result + _scalar(value_type, value, endian)


def _write_gguf(
    path: Path,
    *,
    version: int = 3,
    byte_order: ByteOrder = ByteOrder.LITTLE,
    metadata: tuple[Metadata, ...] = (),
    tensors: tuple[Tensor, ...] = (),
    payload_size: int | None = None,
) -> bytes:
    endian = "<" if byte_order is ByteOrder.LITTLE else ">"
    result = bytearray(b"GGUF")
    result.extend(struct.pack(endian + "IQQ", version, len(tensors), len(metadata)))
    for entry in metadata:
        result.extend(_metadata(entry, endian))
    descriptor_start = len(result)
    for name, dimensions, ggml_type_id, relative_offset in tensors:
        result.extend(_string(name, endian))
        result.extend(struct.pack(endian + "I", len(dimensions)))
        result.extend(struct.pack(endian + f"{len(dimensions)}Q", *dimensions))
        result.extend(struct.pack(endian + "IQ", ggml_type_id, relative_offset))
    alignment = next(
        (int(value) for key, _, value, _ in metadata if key == "general.alignment"), 32
    )
    result.extend(bytes((-len(result)) % alignment))
    if payload_size is None:
        payload_size = max((offset + 32 for _, _, _, offset in tensors), default=0)
    result.extend(bytes(payload_size))
    path.write_bytes(result)
    return bytes(result[descriptor_start:])


def test_indexes_typed_metadata_and_tensor_ranges_without_payload_objects(tmp_path: Path) -> None:
    path = tmp_path / "typed.gguf"
    metadata: tuple[Metadata, ...] = (
        ("general.architecture", GGUFValueType.STRING, "llama", None),
        ("general.alignment", GGUFValueType.UINT32, 32, None),
        ("general.quantization_version", GGUFValueType.UINT32, 2, None),
        ("flag", GGUFValueType.BOOL, True, None),
        ("signed", GGUFValueType.INT64, -7, None),
        ("scores", GGUFValueType.ARRAY, (1.5, -2.0), GGUFValueType.FLOAT32),
        ("names", GGUFValueType.ARRAY, ("a", "beta"), GGUFValueType.STRING),
    )
    _write_gguf(
        path,
        metadata=metadata,
        tensors=(("blk.0.weight", (32,), 8, 0),),
        payload_size=34,
    )

    with open_gguf(path) as reader:
        container = reader.container
        tensor = container.tensor("blk.0.weight")
        assert container.version == 3
        assert container.byte_order is ByteOrder.LITTLE
        assert container.alignment == 32
        assert [entry.key for entry in container.metadata] == [item[0] for item in metadata]
        assert container.metadata_entry("scores").value == pytest.approx((1.5, -2.0))  # type: ignore[union-attr]
        assert container.metadata_entry("names").value == ("a", "beta")  # type: ignore[union-attr]
        assert tensor is not None
        assert tensor.quant_type is GGMLQuantizationType.Q8_0
        assert tensor.dimensions == (32,)
        assert tensor.byte_size == 34
        assert tensor.data_offset == container.data_offset
        assert reader.raw_bytes(tensor.data_offset, 2) == b"\0\0"


@pytest.mark.parametrize(
    ("type_id", "quant_type", "byte_size"),
    [
        (2, GGMLQuantizationType.Q4_0, 18),
        (3, GGMLQuantizationType.Q4_1, 20),
        (6, GGMLQuantizationType.Q5_0, 22),
        (7, GGMLQuantizationType.Q5_1, 24),
    ],
)
def test_indexes_legacy_mixed_recipe_tensors_for_byte_preserving_copy(
    tmp_path: Path,
    type_id: int,
    quant_type: GGMLQuantizationType,
    byte_size: int,
) -> None:
    path = tmp_path / f"legacy-{type_id}.gguf"
    _write_gguf(
        path,
        metadata=(("general.quantization_version", GGUFValueType.UINT32, 2, None),),
        tensors=(("weight", (32,), type_id, 0),),
        payload_size=byte_size,
    )

    with open_gguf(path) as reader:
        tensor = reader.container.tensors[0]
        assert tensor.quant_type is quant_type
        assert tensor.byte_size == byte_size


def test_v2_and_byte_swapped_v3_are_detected(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.gguf"
    big = tmp_path / "big-v3.gguf"
    _write_gguf(v2, version=2)
    _write_gguf(
        big,
        byte_order=ByteOrder.BIG,
        metadata=(("general.alignment", GGUFValueType.UINT32, 64, None),),
    )

    with open_gguf(v2) as reader:
        assert reader.container.byte_order is ByteOrder.LITTLE
        assert reader.container.version == 2
    with open_gguf(big) as reader:
        assert reader.container.byte_order is ByteOrder.BIG
        assert reader.container.alignment == 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.__setitem__(slice(8, 16), struct.pack("<Q", 2_000_000)), "tensor count"),
        (
            lambda data: data.__setitem__(slice(16, 24), struct.pack("<Q", 2_000_000)),
            "metadata count",
        ),
    ],
)
def test_attacker_controlled_counts_fail_before_allocation(
    tmp_path: Path, mutation: Callable[[bytearray], None], match: str
) -> None:
    path = tmp_path / "count.gguf"
    _write_gguf(path)
    data = bytearray(path.read_bytes())
    mutation(data)
    path.write_bytes(data)

    with pytest.raises(GGUFResourceLimitError, match=match):
        open_gguf(path, limits=GGUFParserLimits(max_tensors=10, max_metadata_entries=10))


def test_truncated_string_and_unsupported_version_fail_safely(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(struct.pack("<4sIQQQ", b"GGUF", 3, 0, 1, 100))
    with pytest.raises(CorruptGGUFError, match="metadata key 0"):
        open_gguf(truncated)

    old = tmp_path / "v1.gguf"
    old.write_bytes(struct.pack("<4sIQQ", b"GGUF", 1, 0, 0))
    with pytest.raises(UnsupportedGGUFVersionError, match="expected v2 or v3"):
        open_gguf(old)


def test_duplicate_names_unaligned_offsets_and_overlaps_fail(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.gguf"
    _write_gguf(
        duplicate,
        tensors=(("weight", (8,), 0, 0), ("weight", (8,), 0, 32)),
        payload_size=64,
    )
    with pytest.raises(CorruptGGUFError, match="duplicate GGUF tensor name"):
        open_gguf(duplicate)

    unaligned = tmp_path / "unaligned.gguf"
    _write_gguf(unaligned, tensors=(("weight", (8,), 0, 1),), payload_size=64)
    with pytest.raises(CorruptGGUFError, match="not aligned"):
        open_gguf(unaligned)

    overlap = tmp_path / "overlap.gguf"
    _write_gguf(
        overlap,
        tensors=(("a", (16,), 0, 0), ("b", (16,), 0, 32)),
        payload_size=96,
    )
    with pytest.raises(CorruptGGUFError, match="overlap"):
        open_gguf(overlap)


def test_out_of_file_ranges_partial_blocks_and_unknown_types_fail(tmp_path: Path) -> None:
    outside = tmp_path / "outside.gguf"
    _write_gguf(outside, tensors=(("weight", (16,), 0, 0),), payload_size=4)
    with pytest.raises(CorruptGGUFError, match="exceeds file size"):
        open_gguf(outside)

    partial = tmp_path / "partial.gguf"
    _write_gguf(partial, tensors=(("weight", (31,), 8, 0),), payload_size=64)
    with pytest.raises(CorruptGGUFError, match="not divisible"):
        open_gguf(partial)

    unknown = tmp_path / "unknown.gguf"
    _write_gguf(unknown, tensors=(("weight", (32,), 999, 0),), payload_size=128)
    with pytest.raises(UnsupportedGGUFTypeError, match="type id 999"):
        open_gguf(unknown)

    missing_version = tmp_path / "missing-quantization-version.gguf"
    _write_gguf(
        missing_version,
        tensors=(("weight", (32,), 8, 0),),
        payload_size=34,
    )
    with pytest.raises(CorruptGGUFError, match=r"general\.quantization_version"):
        open_gguf(missing_version)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="40 GiB sparse-file teardown is prohibitively slow on Windows; covered in Linux CI",
)
def test_sparse_40_gib_container_has_bounded_python_memory(tmp_path: Path) -> None:
    import resource

    path = tmp_path / "sparse-40-gib.gguf"
    with path.open("w+b") as stream:
        _write_empty_sparse_header(stream, 40 * 1024**3)

    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    try:
        with open_gguf(path) as reader:
            _, peak = tracemalloc.get_traced_memory()
            assert reader.container.file_size == 40 * 1024**3
            assert reader.container.tensors == ()
        assert peak < 2 * 1024**2
        rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert rss_after_kib - rss_before_kib < 32 * 1024
    finally:
        tracemalloc.stop()


def _write_empty_sparse_header(stream: BinaryIO, size: int) -> None:
    stream.write(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
    stream.truncate(size)
