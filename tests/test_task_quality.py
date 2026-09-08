"""Tests for the real task-quality evaluator and retention gate."""

import json
from pathlib import Path

import pytest

from modelsurgeon.evaluation.task_quality import (
    CodeExactMatchDataset,
    TaskQualityEvaluationError,
    evaluate_code_exact_match,
    evaluate_code_quality_gate,
)

torch = pytest.importorskip("torch")


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt: str, **_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1 if prompt == "first" else 2]]),
            "attention_mask": torch.tensor([[1]]),
        }

    def decode(self, values: torch.Tensor, **_: object) -> str:
        return {3: "return 1", 4: "return 2"}[int(values[0])]


class _Model(torch.nn.Module):
    def __init__(self, *, correct: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.correct = correct

    def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        prompt_id = int(input_ids[0, 0])
        token = {1: 3, 2: (4 if self.correct else 3)}[prompt_id]
        return torch.cat((input_ids, torch.tensor([[token]])), dim=1)


def _dataset(tmp_path: Path) -> CodeExactMatchDataset:
    path = tmp_path / "coding.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"id": "case-1", "prompt": "first", "reference": "return 1"},
                {"id": "case-2", "prompt": "second", "reference": "return 2"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return CodeExactMatchDataset.load(path, revision="dataset-r1", split="python")


def test_code_exact_match_is_physical_and_content_addressed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    baseline = evaluate_code_exact_match(
        _Model(correct=True), _Tokenizer(), torch, dataset, max_new_tokens=4
    )
    candidate = evaluate_code_exact_match(
        _Model(correct=False), _Tokenizer(), torch, dataset, max_new_tokens=4
    )

    assert baseline.score == 1.0
    assert candidate.score == 0.5
    assert baseline.to_record()["measurement_authority"] == "physical_model_evaluation"
    assert dataset.to_record()["digest"].startswith("sha256:")


def test_task_quality_gate_enforces_retention() -> None:
    accepted = evaluate_code_quality_gate(1.0, 0.99, min_quality_retention_ratio=0.99)
    rejected = evaluate_code_quality_gate(1.0, 0.98, min_quality_retention_ratio=0.99)

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["rejection_reasons"]


def test_task_quality_rejects_zero_baseline_and_duplicate_ids(tmp_path: Path) -> None:
    zero_baseline = evaluate_code_quality_gate(0.0, 0.0, min_quality_retention_ratio=0.99)
    assert zero_baseline["accepted"] is False
    assert zero_baseline["inconclusive"] is True
    assert "zero baseline" in zero_baseline["rejection_reasons"][0]

    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"id":"same","prompt":"a","reference":"b"}\n'
        '{"id":"same","prompt":"c","reference":"d"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskQualityEvaluationError, match="duplicate"):
        CodeExactMatchDataset.load(path)
