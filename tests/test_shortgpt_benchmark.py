"""Tests for the reproducible ShortGPT physical benchmark report."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.benchmark_shortgpt_depth import (
    _limitations,
    _measure_perplexity,
    _model_identifier,
    _validated_random_seeds,
    run,
)

torch = pytest.importorskip("torch")


def test_report_limitations_reflect_the_number_of_random_seeds() -> None:
    limitations = _limitations(5)

    assert "5 predeclared random seeds give descriptive, not powered, uncertainty estimates." in (
        limitations
    )
    assert all("Three predeclared random seeds" not in item for item in limitations)


def test_model_identifier_is_inferred_from_hugging_face_cache_without_machine_paths(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "revision"
    model_path.mkdir(parents=True)

    assert _model_identifier(str(model_path), model_path, None) == "Qwen/Qwen2.5-0.5B"
    assert _model_identifier(str(model_path), model_path, "Custom/Name") == "Custom/Name"
    assert _model_identifier("Org/RemoteModel", model_path / "missing", None) == "Org/RemoteModel"


def test_local_model_outside_hugging_face_cache_requires_explicit_identity(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()

    with pytest.raises(ValueError, match="--model-identifier is required"):
        _model_identifier(str(model_path), model_path, None)


def test_random_seed_plan_requires_unique_nonnegative_integer_seeds() -> None:
    assert _validated_random_seeds([11, 23, 47]) == (11, 23, 47)
    with pytest.raises(ValueError, match="unique"):
        _validated_random_seeds([11, 11, 47])
    with pytest.raises(ValueError, match="nonnegative integers"):
        _validated_random_seeds([11, -1, 47])
    with pytest.raises(ValueError, match="nonnegative integers"):
        _validated_random_seeds([11, True, 47])


def test_programmatic_runner_rejects_duplicate_random_seeds_before_setup() -> None:
    with pytest.raises(ValueError, match="unique"):
        run(SimpleNamespace(random_seeds=(11, 11, 47)))


class _TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        assert not use_cache
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_repeated_perplexity_measurement_reports_median_prefill_throughput() -> None:
    model = _TinyCausalLM()
    chunks = ((1, 2, 3, 4), (4, 5, 6))

    result = _measure_perplexity(
        model,
        torch,
        chunks,
        repeats=3,
        warmup_chunks=((7, 6, 5),),
    )

    assert result["device"] == "cpu"
    assert result["measurement_repeats"] == 3
    assert result["token_count"] == 5
    assert len(result["wall_seconds_samples"]) == 3
    assert len(result["tokens_per_second_samples"]) == 3
    assert result["wall_seconds"] > 0
    assert result["tokens_per_second"] > 0
    assert result["perplexity"] > 0
