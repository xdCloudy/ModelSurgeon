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
VALIDATION_EXAMPLES = (
    {
        "input_ids": torch.tensor([[0.2, 1.0, -1.0, 0.5]]),
        "labels": torch.tensor([[1.0, -0.5, 0.0, 0.5]]),
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
    assert result.selected_step == 3
    assert result.validation_losses == ()
    assert len(result.training_examples_sha256) == 64
    assert result.trainable_parameters == 16
    assert result.max_trainable_parameters == 10_000_000
    assert result.to_record()["max_trainable_parameters"] == 10_000_000
    assert result.seed == 7
    assert result.adapter_sha256 == repeated.adapter_sha256
    assert torch.equal(first.proj.base_layer.weight, base_weight)
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


def test_heldout_selection_restores_best_adapter_checkpoint() -> None:
    torch.manual_seed(9)
    original = _TinyLM()
    longer_run = deepcopy(original)
    one_step_run = deepcopy(original)
    settings = {
        "rank": 2,
        "alpha": 4,
        "learning_rate": 0.05,
        "seed": 7,
        "output_mode": LoRAOutputMode.SEPARATE,
    }
    selected = run_bounded_lora_repair(
        longer_run,
        EXAMPLES,
        ("proj",),
        LoRARepairConfig(max_steps=8, **settings),
        source_checkpoint_id="checkpoint_source",
        candidate_checkpoint_id="checkpoint_candidate",
        validation_examples=VALIDATION_EXAMPLES,
        validation_dataset_sha256="a" * 64,
        evaluation_source="external_text",
    )
    one_step = run_bounded_lora_repair(
        one_step_run,
        EXAMPLES,
        ("proj",),
        LoRARepairConfig(max_steps=1, **settings),
        source_checkpoint_id="checkpoint_source",
        candidate_checkpoint_id="checkpoint_candidate",
        validation_examples=VALIDATION_EXAMPLES,
        validation_dataset_sha256="a" * 64,
        evaluation_source="external_text",
    )

    assert selected.completed_steps == 8
    assert selected.selected_step == 1
    assert selected.validation_losses[0] == min(selected.validation_losses)
    assert selected.adapter_sha256 == one_step.adapter_sha256
    assert selected.training_examples_sha256 != selected.validation_examples_sha256
    assert selected.validation_dataset_sha256 == "a" * 64
    assert selected.to_record()["evaluation_source"] == "external_text"


def test_lora_repair_rejects_overlapping_training_and_validation_examples() -> None:
    model = _TinyLM()
    original = model.proj
    with pytest.raises(LoRARepairError, match="must be disjoint"):
        run_bounded_lora_repair(
            model,
            EXAMPLES,
            ("proj",),
            LoRARepairConfig(max_steps=1),
            source_checkpoint_id="checkpoint_source",
            candidate_checkpoint_id="checkpoint_candidate",
            validation_examples=EXAMPLES,
        )
    assert model.proj is original


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


def test_training_failure_restores_original_module_and_grad_state() -> None:
    model = _TinyLM()
    original = model.proj
    original_training = model.training
    original_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())

    with pytest.raises(LoRARepairError, match="requires labels"):
        run_bounded_lora_repair(
            model,
            ({"input_ids": EXAMPLES[0]["input_ids"]},),
            ("proj",),
            LoRARepairConfig(max_steps=1),
            source_checkpoint_id="checkpoint_source",
            candidate_checkpoint_id="checkpoint_candidate",
        )

    assert model.proj is original
    assert model.training is original_training
    restored_requires_grad = tuple(
        parameter.requires_grad for parameter in model.parameters()
    )
    assert restored_requires_grad == original_requires_grad


def test_trainable_parameter_budget_rejects_before_wrapping_targets() -> None:
    model = _TinyLM()
    original = model.proj
    config = LoRARepairConfig(rank=2, max_trainable_parameters=15)

    with pytest.raises(LoRARepairError, match="parameter budget exceeded"):
        run_bounded_lora_repair(
            model,
            EXAMPLES,
            ("proj",),
            config,
            source_checkpoint_id="checkpoint_source",
            candidate_checkpoint_id="checkpoint_candidate",
        )

    assert model.proj is original
    assert model.proj.weight.requires_grad
