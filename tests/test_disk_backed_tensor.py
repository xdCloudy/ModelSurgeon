"""Tests for checksummed resumable disk-backed tensor intermediates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import (
    DiskBackedTensor,
    DiskTensorError,
    DiskTensorSpec,
    StaleDiskTensorError,
)


def _spec(*, plan: str = "a" * 64) -> DiskTensorSpec:
    return DiskTensorSpec("model.layers.0.mlp", (10,), "float32", 4, 16, plan)


def test_chunks_commit_atomically_resume_and_reconstruct_in_bounded_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tensor"
    tensor = DiskBackedTensor.create(path, _spec())
    tensor.append(bytes(range(16)))
    tensor.append(bytes(range(16, 32)))

    resumed = DiskBackedTensor.resume(path, _spec())
    final = resumed.append(bytes(range(32, 40)))

    assert final.byte_count == 8
    assert resumed.manifest.complete is True
    assert resumed.manifest.completed_bytes == 40
    assert resumed.peak_buffer_bytes == 16
    assert b"".join(resumed.iter_chunks()) == bytes(range(40))
    assert not (path / ".manifest.json.tmp").exists()


def test_changed_plan_and_corrupt_committed_chunk_fail_resume(tmp_path: Path) -> None:
    path = tmp_path / "tensor"
    tensor = DiskBackedTensor.create(path, _spec())
    tensor.append(bytes(16))

    with pytest.raises(StaleDiskTensorError, match="different input or plan"):
        DiskBackedTensor.resume(path, _spec(plan="b" * 64))

    (path / "chunk-00000000.bin").write_bytes(bytes([1]) * 16)
    with pytest.raises(StaleDiskTensorError, match="corrupt"):
        DiskBackedTensor.resume(path, _spec())


def test_interrupted_manifest_and_orphan_chunks_are_detected_and_recoverable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tensor"
    tensor = DiskBackedTensor.create(path, _spec())
    tensor.append(bytes(16))
    (path / ".manifest.json.tmp").write_text("partial", encoding="utf-8")
    (path / "chunk-00000001.bin").write_bytes(bytes(16))

    with pytest.raises(StaleDiskTensorError, match="manifest commit"):
        DiskBackedTensor.resume(path, _spec())

    resumed = DiskBackedTensor.resume(path, _spec(), recover_stale=True)
    assert resumed.manifest.completed_bytes == 16
    assert not (path / ".manifest.json.tmp").exists()
    assert not (path / "chunk-00000001.bin").exists()


def test_manifest_is_strict_and_cleanup_removes_only_tensor_directory(tmp_path: Path) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    path = tmp_path / "tensor"
    tensor = DiskBackedTensor.create(path, _spec())
    manifest = json.loads(tensor.manifest_path.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    tensor.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiskTensorError, match="fields"):
        DiskBackedTensor.resume(path, _spec())

    tensor.cleanup()
    assert not path.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_append_requires_exact_bounded_chunk_size(tmp_path: Path) -> None:
    tensor = DiskBackedTensor.create(tmp_path / "tensor", _spec())
    with pytest.raises(DiskTensorError, match="exactly 16 bytes"):
        tensor.append(bytes(15))
