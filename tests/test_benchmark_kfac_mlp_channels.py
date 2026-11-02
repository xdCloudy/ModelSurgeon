"""Tests for the bounded KFAC physical benchmark input windows."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("transformers")

from tools.benchmark_kfac_mlp_channels import _chunks


def test_chunks_starts_at_requested_offset_and_preserves_target_count() -> None:
    chunks = _chunks(list(range(20)), target_tokens=5, sequence_length=4, offset_tokens=7)

    assert [chunk.tolist() for chunk in chunks] == [[[7, 8, 9, 10]], [[10, 11, 12]]]


def test_chunks_rejects_an_offset_past_available_targets() -> None:
    with pytest.raises(ValueError, match="dataset provides 0 of 2 requested targets"):
        _chunks(list(range(5)), target_tokens=2, sequence_length=4, offset_tokens=4)
