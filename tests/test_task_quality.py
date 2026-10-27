"""Tests for the real task-quality evaluator and retention gate."""

import json
from pathlib import Path

import pytest

from modelsurgeon.evaluation import task_quality
from modelsurgeon.evaluation.task_quality import (
    CodeExactMatchDataset,
    MultipleChoiceDataset,
    TaskQualityEvaluationError,
    evaluate_code_exact_match,
    evaluate_code_quality_gate,
    evaluate_multiple_choice_accuracy,
    evaluate_task_quality_gate,
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
        self.dropout = torch.nn.Dropout(p=0.0)
        self.correct = correct

    def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        self.dropout(torch.ones(1))
        prompt_id = int(input_ids[0, 0])
        token = {1: 3, 2: (4 if self.correct else 3)}[prompt_id]
        return torch.cat((input_ids, torch.tensor([[token]])), dim=1)


class _FailingModel(_Model):
    def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        del input_ids
        raise RuntimeError("generation failed")


class _ChoiceTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        tokens = {"Prompt": [10], "Prompt A": [10, 11], "Prompt B": [10, 12]}
        return {"input_ids": tokens.get(text, [])}


class _ChoiceModel(torch.nn.Module):
    def __init__(self, *, correct: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.dropout = torch.nn.Dropout(p=0.0)
        self.correct = correct
        self.forward_calls = 0

    def forward(self, *, input_ids: torch.Tensor) -> object:
        self.forward_calls += 1
        logits = self.dropout(torch.zeros((1, input_ids.shape[1], 16), device=input_ids.device))
        logits[0, 0, 11 if self.correct else 12] = 2.0
        logits[0, 0, 12 if self.correct else 11] = 1.0
        return type("Output", (), {"logits": logits})()


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
    baseline_model = _Model(correct=True).train()
    baseline_model.dropout.eval()
    original_modes = tuple(module.training for module in baseline_model.modules())

    baseline = evaluate_code_exact_match(
        baseline_model, _Tokenizer(), torch, dataset, max_new_tokens=4
    )
    candidate = evaluate_code_exact_match(
        _Model(correct=False), _Tokenizer(), torch, dataset, max_new_tokens=4
    )

    assert baseline.score == 1.0
    assert tuple(module.training for module in baseline_model.modules()) == original_modes
    assert candidate.score == 0.5
    assert baseline.to_record()["measurement_authority"] == "physical_model_evaluation"
    assert dataset.to_record()["digest"].startswith("sha256:")


def test_task_quality_restores_mixed_modes_when_generation_fails(tmp_path: Path) -> None:
    model = _FailingModel(correct=True).train()
    model.dropout.eval()
    original_modes = tuple(module.training for module in model.modules())

    with pytest.raises(TaskQualityEvaluationError, match="generation failed"):
        evaluate_code_exact_match(
            model,
            _Tokenizer(),
            torch,
            _dataset(tmp_path),
            max_new_tokens=4,
        )

    assert tuple(module.training for module in model.modules()) == original_modes


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
        '{"id":"same","prompt":"a","reference":"b"}\n{"id":"same","prompt":"c","reference":"d"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskQualityEvaluationError, match="duplicate"):
        CodeExactMatchDataset.load(path)


def test_code_dataset_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b"{" * 33)
    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_DATASET_BYTES", 32)

    with pytest.raises(TaskQualityEvaluationError, match="exceeds 32 bytes"):
        CodeExactMatchDataset.load(path)


def test_code_dataset_sample_limit_bounds_retained_rows(tmp_path: Path) -> None:
    path = tmp_path / "many.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"case-{index}", "prompt": "p", "reference": "r"})
            for index in range(3)
        ),
        encoding="utf-8",
    )

    dataset = CodeExactMatchDataset.load(path, max_samples=1)

    assert [example.example_id for example in dataset.examples] == ["case-0"]
    assert dataset.digest.startswith("sha256:")


def test_code_dataset_rejects_excess_rows_without_sample_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "too-many.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"case-{index}", "prompt": "p", "reference": "r"})
            for index in range(3)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(task_quality, "MAX_TASK_QUALITY_EXAMPLES", 2)

    with pytest.raises(TaskQualityEvaluationError, match="exceeds 2 examples"):
        CodeExactMatchDataset.load(path)


def test_multiple_choice_uses_teacher_forced_log_likelihood_and_retains_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "choices.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "prompt": "Prompt",
                "choices": [" A", " B"],
                "answer_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = MultipleChoiceDataset.load(path, revision="bench-r1", split="validation")
    baseline_model = _ChoiceModel(correct=True).train()
    baseline_model.dropout.eval()
    original_modes = tuple(module.training for module in baseline_model.modules())

    baseline = evaluate_multiple_choice_accuracy(baseline_model, _ChoiceTokenizer(), torch, dataset)
    damaged = evaluate_multiple_choice_accuracy(
        _ChoiceModel(correct=False), _ChoiceTokenizer(), torch, dataset
    )

    assert baseline.score == 1.0
    assert tuple(module.training for module in baseline_model.modules()) == original_modes
    assert damaged.score == 0.0
    row = baseline.to_record()
    assert row["metric"] == "multiple_choice_accuracy"
    assert row["measurement_authority"] == "physical_model_evaluation"
    assert row["predictions"][0]["answer_index"] == 0
    scores = row["predictions"][0]["choice_log_likelihoods"]
    assert scores[0] > scores[1]
    assert row["dataset"]["digest"].startswith("sha256:")


def test_multiple_choice_retokenized_prompt_boundary_uses_real_combined_tokens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "choices.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "prompt": "abc",
                "choices": [" A", " B"],
                "answer_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = MultipleChoiceDataset.load(path)

    class BoundaryTokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {
                "input_ids": {
                    "abc": [10, 11],
                    "abc A": [20, 21, 22],
                    "abc B": [20, 21, 23],
                }[text]
            }

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.sequences: list[tuple[int, ...]] = []

        def forward(self, *, input_ids: torch.Tensor) -> object:
            self.sequences.append(tuple(int(token) for token in input_ids[0].tolist()))
            logits = torch.zeros((1, input_ids.shape[1], 32), device=input_ids.device)
            return type("Output", (), {"logits": logits})()

    model = RecordingModel()
    evaluate_multiple_choice_accuracy(model, BoundaryTokenizer(), torch, dataset)

    assert model.sequences == [(1, 20, 21, 22), (1, 20, 21, 23)]


def test_multiple_choice_dataset_rejects_invalid_choices_and_gates_zero_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"id":"bad","prompt":"q","choices":["a","a"],"answer_index":0}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskQualityEvaluationError, match="duplicate choices"):
        MultipleChoiceDataset.load(path)

    zero = evaluate_task_quality_gate(
        0.0,
        0.25,
        metric="multiple_choice_accuracy",
        min_quality_retention_ratio=0.99,
    )
    assert zero["accepted"] is False
    assert zero["inconclusive"] is True


def test_multiple_choice_budget_is_checked_before_any_model_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "choices.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "prompt": "Prompt",
                "choices": [" A", " B"],
                "answer_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = MultipleChoiceDataset.load(path)
    model = _ChoiceModel(correct=True)
    monkeypatch.setattr(
        "modelsurgeon.evaluation.task_quality.MAX_MULTIPLE_CHOICE_FORWARD_EVALUATIONS", 1
    )

    with pytest.raises(TaskQualityEvaluationError, match="forward-evaluation budget"):
        evaluate_multiple_choice_accuracy(model, _ChoiceTokenizer(), torch, dataset)

    assert model.forward_calls == 0
