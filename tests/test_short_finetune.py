from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from modelsurgeon.surgery.short_finetune import (  # noqa: E402
    FineTuneParameterMode,
    FineTuneStatus,
    ShortFineTuneConfig,
    ShortFineTuneError,
    run_short_finetune_repair,
)


class _TinyRepairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 1, bias=False)
        self.unused = torch.nn.Parameter(torch.tensor([3.0]))

    def forward(self, input_ids: object, labels: object) -> object:
        prediction = self.proj(input_ids.float())
        return SimpleNamespace(loss=torch.nn.functional.mse_loss(prediction, labels.float()))


TRAIN = ({"input_ids": torch.tensor([[1.0, 0.0]]), "labels": torch.tensor([[1.0]])},)


def _config(**overrides: object) -> ShortFineTuneConfig:
    values = {
        "parameter_mode": FineTuneParameterMode.SELECTED,
        "parameter_names": ("proj.weight",),
        "learning_rate": 0.1,
        "max_steps": 4,
        "validation_patience": 2,
        "max_trainable_parameters": 2,
        "seed": 4,
    }
    values.update(overrides)
    return ShortFineTuneConfig(**values)


def _run(
    model: object,
    validation: tuple[dict[str, object], ...],
    config: ShortFineTuneConfig,
) -> object:
    return run_short_finetune_repair(
        model,
        TRAIN,
        validation,
        config,
        source_checkpoint_id="checkpoint_source",
        candidate_parent_checkpoint_id="checkpoint_candidate",
    )


def test_selected_finetune_accepts_validation_gain_and_preserves_other_parameters() -> None:
    torch.manual_seed(1)
    model = _TinyRepairModel()
    unused = model.unused.detach().clone()
    result = _run(model, TRAIN, _config())

    assert result.status is FineTuneStatus.ACCEPTED
    assert result.output_checkpoint_id is not None
    assert result.validation_loss_delta is not None and result.validation_loss_delta < 0
    assert result.trainable_parameters == 2
    assert torch.equal(model.unused, unused)
    assert result.weights_restored is False


def test_overfit_rejection_restores_exact_no_repair_weights() -> None:
    torch.manual_seed(2)
    model = _TinyRepairModel()
    before = deepcopy(model.state_dict())
    opposite = ({"input_ids": torch.tensor([[1.0, 0.0]]), "labels": torch.tensor([[-1.0]])},)
    result = _run(model, opposite, _config(max_validation_loss_increase=0.0))

    assert result.status is FineTuneStatus.REJECTED_OVERFIT
    assert result.output_checkpoint_id is None
    assert result.weights_restored is True
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_trainable_parameter_budget_fails_before_mutation() -> None:
    model = _TinyRepairModel()
    before = model.proj.weight.detach().clone()
    with pytest.raises(ShortFineTuneError, match="budget exceeded"):
        _run(model, TRAIN, _config(max_trainable_parameters=1))
    assert torch.equal(model.proj.weight, before)


def test_wall_budget_exhaustion_restores_exact_no_repair_weights() -> None:
    model = _TinyRepairModel()
    before = deepcopy(model.state_dict())
    result = _run(model, TRAIN, _config(max_wall_seconds=1e-12))

    assert result.status is FineTuneStatus.BUDGET_EXHAUSTED
    assert result.completed_steps == 0
    assert result.output_checkpoint_id is None
    assert result.weights_restored is True
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
