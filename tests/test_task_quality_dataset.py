"""Tests for loading task-quality JSONL datasets without model dependencies."""

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.evaluation import task_quality
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


def test_task_quality_dataset_enforces_byte_row_and_line_count_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b"{" * 20)
    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_DATASET_BYTES", 16)
    with pytest.raises(TaskQualityEvaluationError, match="dataset exceeds 16 bytes"):
        CodeExactMatchDataset.load(path)

    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_DATASET_BYTES", 1024)
    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_ROW_BYTES", 16)
    with pytest.raises(TaskQualityEvaluationError, match="line 1 exceeds 16 bytes"):
        CodeExactMatchDataset.load(path)

    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_ROW_BYTES", 1024)
    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_ROWS", 1)
    path.write_text(
        '{"id":"case-1","prompt":"first","reference":"return 1"}\n'
        '{"id":"case-2","prompt":"second","reference":"return 2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskQualityEvaluationError, match="dataset exceeds 1 rows"):
        CodeExactMatchDataset.load(path, max_samples=1)


def test_task_quality_dataset_normalizes_json_recursion_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested.jsonl"
    path.write_text('{"id":"case-1","prompt":"first","reference":"return 1"}\n')

    def fail_json_decode(_: str) -> object:
        raise RecursionError("parser recursion limit")

    monkeypatch.setattr(task_quality.json, "loads", fail_json_decode)

    with pytest.raises(TaskQualityEvaluationError, match="line 1 is not valid JSON"):
        CodeExactMatchDataset.load(path)


def test_task_quality_dataset_normalizes_json_recursion_after_sample_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deeply-nested-tail.jsonl"
    valid_row = '{"id":"case-1","prompt":"first","reference":"return 1"}\n'
    deeply_nested_row = "[" * 100_000 + "0" + "]" * 100_000 + "\n"
    path.write_text(valid_row + deeply_nested_row, encoding="utf-8")

    with pytest.raises(TaskQualityEvaluationError, match="line 2 is not valid JSON"):
        CodeExactMatchDataset.load(path, max_samples=1)
