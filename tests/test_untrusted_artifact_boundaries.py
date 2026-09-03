"""Property-style coverage for untrusted GGUF bytes and checkpoint paths."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import GGUFParseError, open_gguf
from modelsurgeon.surgery.checkpoint_destination import (
    AtomicCheckpointDestination,
    CheckpointDestinationError,
)


def test_arbitrary_small_gguf_inputs_fail_only_with_parser_errors(tmp_path: Path) -> None:
    """Parser fuzz corpus must not leak mmap/struct exceptions for hostile bytes."""

    generator = random.Random(201)
    payloads = [b""]
    payloads.extend(generator.randbytes(size) for size in range(1, 257))
    for index, payload in enumerate(payloads):
        path = tmp_path / f"untrusted-{index}.gguf"
        path.write_bytes(payload)
        with pytest.raises(GGUFParseError):
            open_gguf(path)


@pytest.mark.parametrize(
    "destination_text",
    ("source.gguf", "./source.gguf", "nested/../source.gguf"),
)
def test_checkpoint_path_aliases_never_target_the_source(
    tmp_path: Path, destination_text: str
) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"source")

    with pytest.raises(CheckpointDestinationError, match="same checkpoint"):
        AtomicCheckpointDestination(source, tmp_path / destination_text)

    assert source.read_bytes() == b"source"
