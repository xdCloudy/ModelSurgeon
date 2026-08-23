import json
from pathlib import Path

import pytest

from modelsurgeon.adapters.safetensors import (
    SafetensorsIndexError,
    inspect_safetensors,
    inspect_safetensors_file,
)


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def test_inspect_single_file_without_reading_tensor_values(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(
        path,
        {
            "model.layers.0.weight": ("F32", [2], b"\x00" * 8),
            "model.layers.0.bias": ("F16", [2], b"\x00" * 4),
        },
    )

    entries = inspect_safetensors(path)

    assert [entry.tensor_name for entry in entries] == [
        "model.layers.0.bias",
        "model.layers.0.weight",
    ]
    assert entries[0].shape == (2,)
    assert entries[0].dtype == "F16"
    assert entries[0].byte_size == 4
    assert entries[0].data_offset < path.stat().st_size
    assert json.loads(json.dumps(entries[0].to_record()))["shard"] == path.name


def test_inspect_sharded_index(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    write_safetensors(first, {"model.embed.weight": ("F16", [2, 2], b"\x00" * 8)})
    write_safetensors(second, {"model.norm.weight": ("F32", [2], b"\x00" * 8)})
    index = {
        "metadata": {"total_size": 16},
        "weight_map": {
            "model.embed.weight": first.name,
            "model.norm.weight": second.name,
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    entries = inspect_safetensors(tmp_path)

    assert [(entry.tensor_name, entry.shard) for entry in entries] == [
        ("model.embed.weight", first.name),
        ("model.norm.weight", second.name),
    ]
    assert sum(entry.byte_size for entry in entries) == 16


def test_missing_shard_fails_safely(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(SafetensorsIndexError, match="does not exist"):
        inspect_safetensors(index_path)


def test_index_and_shard_disagreement_is_rejected(tmp_path: Path) -> None:
    shard = tmp_path / "part.safetensors"
    write_safetensors(shard, {"actual": ("F32", [1], b"\x00" * 4)})
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"expected": shard.name}}), encoding="utf-8"
    )

    with pytest.raises(SafetensorsIndexError, match="disagrees"):
        inspect_safetensors(index_path)


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        (10).to_bytes(8, "little") + b"{}",
        (2).to_bytes(8, "little") + b"[]",
        (3).to_bytes(8, "little") + b"not",
    ],
)
def test_malformed_headers_are_rejected(contents: bytes, tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(contents)

    with pytest.raises(SafetensorsIndexError):
        inspect_safetensors_file(path)


def test_out_of_bounds_overlap_and_size_mismatch_are_rejected(tmp_path: Path) -> None:
    cases = [
        {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]},
        },
        {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
        },
        {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
        },
    ]
    for index, header in enumerate(cases):
        path = tmp_path / f"bad-{index}.safetensors"
        encoded = json.dumps(header).encode("utf-8")
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\x00" * 8)

        with pytest.raises(SafetensorsIndexError):
            inspect_safetensors_file(path)

