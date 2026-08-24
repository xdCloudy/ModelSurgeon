from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from modelsurgeon.surgery.distillation_repair import (  # noqa: E402
    DistillationRepairConfig,
    DistillationRepairError,
    DistillationStatus,
    TeacherLogitSource,
    TokenizerSignature,
    run_distillation_repair,
)


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(3, 3))

    def forward(
        self,
        input_ids: object,
        attention_mask: object | None = None,
        labels: object | None = None,
    ) -> object:
        del attention_mask
        logits = self.table[input_ids]
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, 3), labels[:, 1:].reshape(-1)
            )
        return SimpleNamespace(logits=logits, loss=loss)


SIGNATURE = TokenizerSignature(3, "0" * 64, eos_token_id=2)
EXAMPLES = (
    {
        "input_ids": torch.tensor([[0, 1]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[0, 2]]),
    },
)
TEACHER_LOGITS = (torch.tensor([[[0.0, 4.0, -1.0], [-1.0, 0.0, 4.0]]]),)


def _config(**overrides: object) -> DistillationRepairConfig:
    values = {
        "parameter_names": ("table",),
        "temperature": 2.0,
        "learning_rate": 0.1,
        "max_steps": 4,
        "max_trainable_parameters": 9,
        "seed": 5,
    }
    values.update(overrides)
    return DistillationRepairConfig(**values)


def _run(
    candidate: object,
    config: DistillationRepairConfig,
    **overrides: object,
) -> object:
    arguments = {
        "teacher_tokenizer": SIGNATURE,
        "candidate_tokenizer": SIGNATURE,
        "source_checkpoint_id": "checkpoint_source",
        "candidate_parent_checkpoint_id": "checkpoint_candidate",
        "baseline_logits": TEACHER_LOGITS,
    }
    arguments.update(overrides)
    return run_distillation_repair(candidate, EXAMPLES, config, **arguments)


def test_precomputed_logits_are_immutable_and_selected_repair_is_recorded() -> None:
    candidate = _TinyCausalModel()
    immutable_before = TEACHER_LOGITS[0].clone()
    result = _run(
        candidate,
        _config(distillation_weight=0.75, supervised_weight=0.25),
    )

    assert result.status is DistillationStatus.ACCEPTED
    assert result.teacher_source is TeacherLogitSource.PRECOMPUTED
    assert result.output_checkpoint_id is not None
    assert result.repaired_loss is not None and result.repaired_loss < result.baseline_loss
    assert result.config.temperature == 2.0
    assert result.config.distillation_weight == 0.75
    assert result.config.supervised_weight == 0.25
    assert result.resource_use.token_rows == 2
    assert result.resource_use.teacher_logit_bytes == TEACHER_LOGITS[0].numel() * 4
    assert torch.equal(TEACHER_LOGITS[0], immutable_before)


def test_distinct_teacher_inference_is_frozen_and_mode_is_restored() -> None:
    candidate = _TinyCausalModel()
    teacher = _TinyCausalModel()
    with torch.no_grad():
        teacher.table[:2].copy_(TEACHER_LOGITS[0][0])
    teacher.train()
    before = teacher.table.detach().clone()
    result = _run(
        candidate,
        _config(max_steps=2),
        baseline_logits=None,
        teacher_model=teacher,
    )

    assert result.teacher_source is TeacherLogitSource.TEACHER_INFERENCE
    assert teacher.training is True
    assert teacher.table.requires_grad is True
    assert torch.equal(teacher.table, before)


def test_tokenizer_mismatch_fails_before_candidate_mutation() -> None:
    candidate = _TinyCausalModel()
    before = deepcopy(candidate.state_dict())
    mismatch = TokenizerSignature(3, "1" * 64, eos_token_id=2)
    with pytest.raises(DistillationRepairError, match="incompatible"):
        _run(candidate, _config(), candidate_tokenizer=mismatch)
    assert all(torch.equal(candidate.state_dict()[name], value) for name, value in before.items())


def test_wall_budget_exhaustion_restores_candidate_exactly() -> None:
    candidate = _TinyCausalModel()
    before = deepcopy(candidate.state_dict())
    result = _run(candidate, _config(max_wall_seconds=1e-12))

    assert result.status is DistillationStatus.BUDGET_EXHAUSTED
    assert result.completed_steps == 0
    assert result.output_checkpoint_id is None
    assert all(torch.equal(candidate.state_dict()[name], value) for name, value in before.items())


def test_no_improvement_rejects_and_restores_candidate() -> None:
    candidate = _TinyCausalModel()
    before = deepcopy(candidate.state_dict())
    matching = (torch.zeros(1, 2, 3),)
    result = _run(candidate, _config(max_steps=1), baseline_logits=matching)

    assert result.status is DistillationStatus.REJECTED_NO_IMPROVEMENT
    assert result.output_checkpoint_id is None
    assert result.repaired_loss == pytest.approx(result.baseline_loss)
    assert all(torch.equal(candidate.state_dict()[name], value) for name, value in before.items())


def test_tokenizer_signature_hashes_effective_vocabulary() -> None:
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "b": 1, "<eos>": 2},
        bos_token_id=None,
        eos_token_id=2,
        pad_token_id=None,
        unk_token_id=None,
    )
    first = TokenizerSignature.from_tokenizer(tokenizer)
    second = TokenizerSignature.from_tokenizer(tokenizer)
    assert first == second
    assert first.vocabulary_size == 3
