from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from modelsurgeon.surgery.lora_repair import (  # noqa: E402
    LoRAOutputMode,
    LoRARepairConfig,
    LoRARepairError,
    lora_adapter_state_dict,
    run_bounded_lora_repair,
)


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, input_ids: object, labels: object) -> object:
        prediction = self.proj(input_ids.float())
        loss = torch.nn.functional.mse_loss(prediction, labels.float())
        return SimpleNamespace(loss=loss)


EXAMPLES = (
    {
        "input_ids": torch.tensor([[1.0, 0.0, 0.5, -0.5]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
    },
)


def _run(model: object, mode: LoRAOutputMode) -> object:
    return run_bounded_lora_repair(
        model,
        EXAMPLES,
        ("proj",),
        LoRARepairConfig(
            rank=2,
            alpha=4,
            learning_rate=0.05,
            max_steps=3,
            seed=7,
            output_mode=mode,
        ),
        source_checkpoint_id="checkpoint_source",
        candidate_checkpoint_id="checkpoint_candidate",
    )


def test_separate_lora_is_bounded_deterministic_and_preserves_base_weights() -> None:
    torch.manual_seed(2)
    original = _TinyLM()
    first = deepcopy(original)
    second = deepcopy(original)
    base_weight = first.proj.weight.detach().clone()
    result = _run(first, LoRAOutputMode.SEPARATE)
    repeated = _run(second, LoRAOutputMode.SEPARATE)

    assert result.completed_steps == 3
    assert result.trainable_parameters == 16
    assert result.seed == 7
    assert result.adapter_sha256 == repeated.adapter_sha256
    assert torch.equal(first.proj.base.weight, base_weight)
    state = lora_adapter_state_dict(first, ("proj",))
    assert set(state) == {"proj.lora_a", "proj.lora_b"}
    assert result.resource_use.wall_seconds >= 0


def test_merged_lora_updates_only_candidate_and_removes_adapter_wrapper() -> None:
    torch.manual_seed(3)
    model = _TinyLM()
    source_weight = model.proj.weight.detach().clone()
    result = _run(model, LoRAOutputMode.MERGED)

    assert result.output_mode is LoRAOutputMode.MERGED
    assert isinstance(model.proj, torch.nn.Linear)
    assert not torch.equal(model.proj.weight, source_weight)
    with pytest.raises(LoRARepairError, match="does not retain"):
        lora_adapter_state_dict(model, ("proj",))


def test_invalid_target_restores_model_before_failure() -> None:
    model = _TinyLM()
    original = model.proj
    with pytest.raises(LoRARepairError, match="does not exist"):
        run_bounded_lora_repair(
            model,
            EXAMPLES,
            ("proj", "zzz"),
            LoRARepairConfig(max_steps=1),
            source_checkpoint_id="checkpoint_source",
            candidate_checkpoint_id="checkpoint_candidate",
        )
    assert model.proj is original
