"""Integration coverage for mixed cumulative Hugging Face surgery."""

from __future__ import annotations

from pathlib import Path

import torch

from modelsurgeon.adapters.huggingface import (
    remove_huggingface_attention_heads,
    remove_huggingface_mlp_channels,
    remove_huggingface_transformer_layers,
)
from modelsurgeon.surgery import (
    HuggingFaceEdit,
    run_huggingface_cumulative_sequence,
)


class _Config:
    intermediate_size = 4
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    num_key_value_groups = 2
    hidden_size = 8
    head_dim = 2


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(8, 4, bias=False)
        self.up_proj = torch.nn.Linear(8, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 8, bias=False)


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])

    def generate(self) -> torch.Tensor:
        return torch.ones(1)


def _edits() -> tuple[HuggingFaceEdit, ...]:
    return (
        HuggingFaceEdit(
            "mlp-width",
            "remove_mlp_channels",
            lambda model: remove_huggingface_mlp_channels(model, (0,)),
        ),
        HuggingFaceEdit(
            "attention-heads",
            "remove_attention_heads",
            lambda model: remove_huggingface_attention_heads(model, (0, 1)),
        ),
        HuggingFaceEdit(
            "transformer-layer",
            "remove_transformer_layer",
            lambda model: remove_huggingface_transformer_layers(model, (0,)),
        ),
    )


def _publisher(model: _Model, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, destination)
    return destination


def _reloader(path: Path) -> _Model:
    return torch.load(path, weights_only=False)


def test_three_mixed_edits_save_reload_generate_and_reconcile(tmp_path: Path) -> None:
    edits = _edits()
    sequences = (edits, (edits[1], edits[0], edits[2]), (edits[2], edits[1], edits[0]))
    runs = [
        run_huggingface_cumulative_sequence(
            _Model(),
            sequence,
            output_root=tmp_path / str(index),
            source_outcome_id="source-hf-outcome",
            publish=_publisher,
            reload=_reloader,
            generate=lambda model: bool(model.generate().numel()),
        )
        for index, sequence in enumerate(sequences)
    ]

    assert all(len(run.stages) == 3 for run in runs)
    assert all(run.failed_index is None and run.reconciliation.valid for run in runs)
    assert all(run.reconciliation.stage_count == 3 for run in runs)
    assert all(
        run.stages[-1].outcome.architecture.active_parameters
        < run.stages[0].outcome.architecture.source_parameters
        for run in runs
    )
    assert all(
        stage.reloadable and stage.generation_smoke and stage.artifact.is_file()
        for run in runs
        for stage in run.stages
    )
    assert len({run.sequence_id for run in runs}) == 3


def test_failed_stage_removes_child_and_keeps_last_accepted_model(tmp_path: Path) -> None:
    calls = 0

    def generate(model: _Model) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    run = run_huggingface_cumulative_sequence(
        _Model(),
        _edits()[:2],
        output_root=tmp_path,
        source_outcome_id="source-hf-outcome",
        publish=_publisher,
        reload=_reloader,
        generate=generate,
    )

    assert len(run.stages) == 1
    assert run.failed_index == 1
    assert run.failure_reason
    assert run.final_model.config.intermediate_size == 3
    assert not any(path.name.endswith("attention-heads") for path in tmp_path.iterdir())
