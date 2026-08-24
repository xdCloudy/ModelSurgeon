from __future__ import annotations

import json

import pytest

from modelsurgeon.adapters.safetensors import (
    inspect_safetensors,
    write_safetensors_checkpoint_atomic,
)

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _source(directory) -> bytes:
    directory.mkdir()
    shard = directory / "model.safetensors"
    safetensors_torch.save_file({"original": torch.arange(3)}, shard)
    return shard.read_bytes()


def test_sharded_checkpoint_is_verified_and_source_is_unchanged(tmp_path) -> None:
    source = tmp_path / "source"
    original = _source(source)
    destination = tmp_path / "published"
    tensors = {
        "model.a": torch.arange(4, dtype=torch.float32),
        "model.b": torch.arange(4, dtype=torch.int32),
    }

    report = write_safetensors_checkpoint_atomic(
        source,
        destination,
        tensors,
        {"architectures": ["TinyModel"], "hidden_size": 4},
        max_shard_bytes=16,
        max_tensor_bytes=16,
    )

    assert report.sharded
    assert (source / "model.safetensors").read_bytes() == original
    assert [item.tensor_name for item in report.tensors] == ["model.a", "model.b"]
    assert [item.shape for item in report.tensors] == [(4,), (4,)]
    assert len({item.sha256 for item in report.tensors}) == 2
    assert {entry.tensor_name for entry in inspect_safetensors(destination)} == set(tensors)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 32
    assert tuple(index["weight_map"]) == ("model.a", "model.b")
    assert json.loads((destination / "config.json").read_text())["hidden_size"] == 4
    for name, expected in tensors.items():
        actual = safetensors_torch.load_file(destination / index["weight_map"][name])[name]
        assert torch.equal(actual, expected)


def test_single_checkpoint_omits_shard_index(tmp_path) -> None:
    source = tmp_path / "source"
    _source(source)
    destination = tmp_path / "published"

    report = write_safetensors_checkpoint_atomic(
        source,
        destination,
        {"weight": torch.eye(2)},
        {"model_type": "tiny"},
        max_shard_bytes=64,
    )

    assert not report.sharded
    assert (destination / "model.safetensors").is_file()
    assert not (destination / "model.safetensors.index.json").exists()


def test_interrupted_shard_write_never_publishes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    original = _source(source)
    destination = tmp_path / "published"
    real_save = safetensors_torch.save_file
    calls = 0

    def fail_second(tensors, path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        real_save(tensors, path)

    monkeypatch.setattr(safetensors_torch, "save_file", fail_second)
    with pytest.raises(RuntimeError, match="staging failed"):
        write_safetensors_checkpoint_atomic(
            source,
            destination,
            {"a": torch.ones(4), "b": torch.zeros(4)},
            {},
            max_shard_bytes=16,
        )

    assert not destination.exists()
    assert (source / "model.safetensors").read_bytes() == original
    assert not tuple(tmp_path.glob(".published.*.staging"))
