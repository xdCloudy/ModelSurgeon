"""Tests for loading task-quality JSONL datasets without model dependencies."""

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.evaluation.task_quality import CodeExactMatchDataset, TaskQualityEvaluationError


def test_task_quality_dataset_streams_and_caps_retained_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        b'{"id":"case-1","prompt":"first","reference":"return 1"}\n'
        b'{"id":"case-2","prompt":"second","reference":"return 2"}\n'
    )
    path = tmp_path / "large.jsonl"
    path.write_bytes(payload)

    def fail_read_bytes(_: Path) -> bytes:
        raise AssertionError("dataset loader must stream instead of reading all bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    dataset = CodeExactMatchDataset.load(path, max_samples=1)

    assert [example.example_id for example in dataset.examples] == ["case-1"]
    assert dataset.digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        ('{"id":"case-1","prompt":"second","reference":"return 2"}\n', "duplicate"),
        ('{"id":"case-2","prompt":"second"}\n', "reference"),
        ("not-json\n", "line 2"),
    ],
)
def test_task_quality_dataset_validates_rows_after_sample_limit(
    tmp_path: Path, tail: str, message: str
) -> None:
    path = tmp_path / "tail.jsonl"
    path.write_text(
        '{"id":"case-1","prompt":"first","reference":"return 1"}\n' + tail,
        encoding="utf-8",
    )

    with pytest.raises(TaskQualityEvaluationError, match=message):
        CodeExactMatchDataset.load(path, max_samples=1)
