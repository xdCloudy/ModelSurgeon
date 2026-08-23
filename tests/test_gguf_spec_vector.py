"""Conformance tests for the bounded GGUF specification spike vector."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

from modelsurgeon.adapters.gguf import GGMLQuantizationType as ModelSurgeonQuantType
from modelsurgeon.adapters.gguf import open_gguf

FIXTURE = Path(__file__).parent / "fixtures" / "gguf_spec_v1.json"
VALUE_TYPES = {
    "uint32": 4,
    "bool": 7,
    "string": 8,
    "array_uint32": 9,
}


def _vector() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _write_string(stream: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def _write_value(stream: BinaryIO, value_type: str, value: object) -> None:
    if value_type == "string":
        _write_string(stream, cast(str, value))
    elif value_type == "uint32":
        stream.write(struct.pack("<I", cast(int, value)))
    elif value_type == "bool":
        stream.write(struct.pack("<?", cast(bool, value)))
    elif value_type == "array_uint32":
        items = cast(list[int], value)
        stream.write(struct.pack("<I", VALUE_TYPES["uint32"]))
        stream.write(struct.pack("<Q", len(items)))
        stream.write(struct.pack(f"<{len(items)}I", *items))
    else:  # pragma: no cover - vector schema is closed by this test
        raise AssertionError(f"unsupported fixture metadata type: {value_type}")


def _write_fixture(path: Path, vector: dict[str, object]) -> None:
    container = cast(dict[str, object], vector["container"])
    metadata = cast(list[dict[str, object]], container["metadata"])
    tensor = cast(dict[str, object], container["tensor"])
    alignment = cast(int, container["alignment"])
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<4sIQQ",
                b"GGUF",
                cast(int, container["version"]),
                1,
                len(metadata),
            )
        )
        for entry in metadata:
            value_type = cast(str, entry["type"])
            _write_string(stream, cast(str, entry["key"]))
            stream.write(struct.pack("<I", VALUE_TYPES[value_type]))
            _write_value(stream, value_type, entry["value"])
        _write_string(stream, cast(str, tensor["name"]))
        dimensions = cast(list[int], tensor["dimensions"])
        stream.write(struct.pack("<I", len(dimensions)))
        stream.write(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        stream.write(struct.pack("<I", cast(int, tensor["ggml_type_id"])))
        stream.write(struct.pack("<Q", cast(int, tensor["relative_offset"])))
        padding = (-stream.tell()) % alignment
        stream.write(bytes(padding))
        stream.write(struct.pack("<e", 0.5))
        stream.write(struct.pack("<32b", *range(-16, 16)))


@dataclass
class _Cursor:
    data: bytes
    offset: int = 0

    def unpack(self, format_string: str) -> tuple[object, ...]:
        values = struct.unpack_from(format_string, self.data, self.offset)
        self.offset += struct.calcsize(format_string)
        return cast(tuple[object, ...], values)

    def string(self) -> str:
        (length,) = self.unpack("<Q")
        start = self.offset
        self.offset += cast(int, length)
        return self.data[start : self.offset].decode("utf-8")


def _oracle_read(path: Path) -> dict[str, object]:
    """Independent stdlib reader for only the closed conformance-vector grammar."""
    cursor = _Cursor(path.read_bytes())
    magic, version, tensor_count, kv_count = cursor.unpack("<4sIQQ")
    metadata: dict[str, object] = {}
    for _ in range(cast(int, kv_count)):
        key = cursor.string()
        (value_type,) = cursor.unpack("<I")
        if value_type == VALUE_TYPES["string"]:
            metadata[key] = cursor.string()
        elif value_type == VALUE_TYPES["uint32"]:
            (metadata[key],) = cursor.unpack("<I")
        elif value_type == VALUE_TYPES["bool"]:
            (metadata[key],) = cursor.unpack("<?")
        elif value_type == VALUE_TYPES["array_uint32"]:
            (element_type,) = cursor.unpack("<I")
            assert element_type == VALUE_TYPES["uint32"]
            (length,) = cursor.unpack("<Q")
            metadata[key] = list(cursor.unpack(f"<{length}I"))
        else:  # pragma: no cover - vector schema is closed by this test
            raise AssertionError(f"unexpected metadata type {value_type}")
    tensor_name = cursor.string()
    (dimension_count,) = cursor.unpack("<I")
    dimensions = list(cursor.unpack(f"<{dimension_count}Q"))
    ggml_type, relative_offset = cursor.unpack("<IQ")
    alignment = cast(int, metadata.get("general.alignment", 32))
    data_offset = (cursor.offset + alignment - 1) // alignment * alignment
    return {
        "magic": magic,
        "version": version,
        "tensor_count": tensor_count,
        "metadata": metadata,
        "tensor": {
            "name": tensor_name,
            "dimensions": dimensions,
            "ggml_type_id": ggml_type,
            "relative_offset": relative_offset,
            "data_offset": data_offset,
        },
    }


def test_fixture_metadata_agrees_in_two_independent_readers(tmp_path: Path) -> None:
    vector = _vector()
    path = tmp_path / "spec-v3-q8_0.gguf"
    _write_fixture(path, vector)

    oracle = _oracle_read(path)
    official = GGUFReader(path)
    native = open_gguf(path)
    expected = cast(dict[str, object], vector["container"])
    expected_metadata = {
        cast(str, item["key"]): item["value"]
        for item in cast(list[dict[str, object]], expected["metadata"])
    }
    assert oracle["magic"] == b"GGUF"
    assert oracle["version"] == expected["version"]
    assert oracle["metadata"] == expected_metadata
    try:
        assert official.alignment == expected["alignment"]
        for key, value in expected_metadata.items():
            assert official.fields[key].contents() == value
            expected_native = tuple(value) if isinstance(value, list) else value
            assert native.container.metadata_entry(key).value == expected_native  # type: ignore[union-attr]
        expected_tensor = cast(dict[str, object], expected["tensor"])
        tensor = official.tensors[0]
        native_tensor = native.container.tensors[0]
        assert tensor.name == expected_tensor["name"] == native_tensor.name
        assert tensor.tensor_type.name == expected_tensor["ggml_type"]
        assert native_tensor.quant_type is ModelSurgeonQuantType.Q8_0
        assert tensor.shape.tolist() == expected_tensor["dimensions"]
        assert native_tensor.dimensions == tuple(expected_tensor["dimensions"])  # type: ignore[arg-type]
        assert tensor.n_elements == expected_tensor["element_count"]
        assert tensor.n_bytes == expected_tensor["byte_count"] == native_tensor.byte_size
        oracle_data_offset = cast(dict[str, object], oracle["tensor"])["data_offset"]
        assert official.data_offset == oracle_data_offset == native.container.data_offset
    finally:
        native.close()


def test_quant_block_layouts_match_pinned_official_reader() -> None:
    layouts = cast(dict[str, list[int]], _vector()["quant_layouts"])

    actual = {
        name: list(GGML_QUANT_SIZES[GGMLQuantizationType[name]]) for name in layouts
    }

    assert actual == layouts
