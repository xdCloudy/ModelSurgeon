import json
from pathlib import Path

import pytest
from tools.run_lora_post_surgery_study import _measure_task_quality, _ordered_test_rows

from modelsurgeon.evaluation.task_quality import CodeExactMatchDataset


def test_test_row_order_is_preserved_without_a_sampling_seed() -> None:
    rows = tuple((index, f"row-{index}") for index in range(20))

    assert _ordered_test_rows(rows, None) is rows


def test_test_row_shuffle_is_deterministic_and_preserves_every_row() -> None:
    rows = tuple((index, f"row-{index}") for index in range(20))

    first = _ordered_test_rows(rows, 2026)
    repeated = _ordered_test_rows(rows, 2026)

    assert first == repeated
    assert first != rows
    assert set(first) == set(rows)


def test_study_can_measure_task_quality_for_an_arm(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompt: str, **_: object) -> dict[str, object]:
            return {
                "input_ids": torch.tensor([[1 if prompt == "first" else 2]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, values: object, **_: object) -> str:
            return {3: "return 1", 4: "return 2"}[int(values[0])]  # type: ignore[index]

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def generate(self, *, input_ids: object, **_: object) -> object:
            prompt_id = int(input_ids[0, 0])  # type: ignore[index]
            token = 3 if prompt_id == 1 else 4
            return torch.cat((input_ids, torch.tensor([[token]])), dim=1)

    dataset_path = tmp_path / "task.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "add", "prompt": "first", "reference": "return 1"}) + "\n",
        encoding="utf-8",
    )
    dataset = CodeExactMatchDataset.load(dataset_path, revision="test-v1")

    record = _measure_task_quality(
        Model(), Tokenizer(), torch, dataset, max_new_tokens=4
    )

    assert record is not None
    assert record["matches"] == 1
    assert record["total"] == 1
    assert record["measurement_authority"] == "physical_model_evaluation"
    assert record["dataset"]["path"] == "task.jsonl"
