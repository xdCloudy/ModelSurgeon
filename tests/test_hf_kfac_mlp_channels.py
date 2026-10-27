from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.huggingface.kfac_mlp_channels import (
    HuggingFaceMLPKFACError,
    score_huggingface_gated_mlp_channels_kfac,
)

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _TinyLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.mlp = _TinyMLP(hidden_size, intermediate_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class _TinyBackbone(nn.Module):
    def __init__(self, layer_count: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(13, hidden_size)
        self.layers = nn.ModuleList(
            [_TinyLayer(hidden_size, intermediate_size) for _ in range(layer_count)]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _TinyCausalLM(nn.Module):
    def __init__(self, *, model_type: str = "llama") -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=2,
            hidden_size=4,
            intermediate_size=6,
        )
        self.model = _TinyBackbone(2, 4, 6)
        self.lm_head = nn.Linear(4, 13, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        logits = self.lm_head(self.model(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 13),
            labels[:, 1:].reshape(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_collects_cpu_factors_and_scores_shared_gated_mlp_channels() -> None:
    model = _TinyCausalLM()
    model.train()
    token_ids = torch.tensor([[1, 3, 2, 5, 8, 4, 7, 6]], dtype=torch.int64)
    down_projections = [layer.mlp.down_proj for layer in model.model.layers]

    report = score_huggingface_gated_mlp_channels_kfac(
        model,
        (token_ids[:, :4], token_ids[:, 4:]),
        max_calibration_target_tokens=6,
        max_workspace_bytes=128 * 1024 * 1024,
    )

    assert model.training
    assert report.model_type == "llama"
    assert report.layer_count == 2
    assert report.calibration_target_tokens == 6
    assert report.intermediate_size == 6
    assert len(report.per_layer_costs) == 2
    assert len(report.aggregate_costs) == 6
    assert len(report.activation_factor_sha256) == 2
    assert len(report.gradient_factor_sha256) == 2
    assert all(cost >= 0.0 for cost in report.aggregate_costs)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module._forward_hooks for module in down_projections)
    assert all(not module._forward_pre_hooks for module in down_projections)


def test_collects_bfloat16_cpu_factors_without_upcasting_model_weights() -> None:
    model = _TinyCausalLM().to(dtype=torch.bfloat16)
    token_ids = torch.tensor([[1, 3, 2, 5, 8, 4, 7, 6]], dtype=torch.int64)

    report = score_huggingface_gated_mlp_channels_kfac(
        model,
        (token_ids,),
        max_calibration_target_tokens=7,
        max_workspace_bytes=128 * 1024 * 1024,
    )

    assert report.calibration_target_tokens == 7
    assert all(parameter.dtype is torch.bfloat16 for parameter in model.parameters())
    assert all(torch.isfinite(torch.tensor(cost)) for cost in report.aggregate_costs)


def test_rejects_unsupported_model_type_before_registering_hooks() -> None:
    model = _TinyCausalLM(model_type="gpt2")
    with pytest.raises(HuggingFaceMLPKFACError, match="only Llama and Qwen2"):
        score_huggingface_gated_mlp_channels_kfac(model, ())
    assert all(not module._forward_hooks for module in model.modules())


def test_rejects_calibration_exceeding_configured_token_budget() -> None:
    model = _TinyCausalLM()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with pytest.raises(HuggingFaceMLPKFACError, match="exceed max_calibration_target_tokens"):
        score_huggingface_gated_mlp_channels_kfac(
            model,
            (token_ids,),
            max_calibration_target_tokens=2,
            max_workspace_bytes=128 * 1024 * 1024,
        )
    assert all(not module._forward_hooks for module in model.modules())


def test_preflights_total_workspace_before_allocating_collectors() -> None:
    model = _TinyCausalLM()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with pytest.raises(HuggingFaceMLPKFACError, match="planned KFAC collector workspace"):
        score_huggingface_gated_mlp_channels_kfac(
            model,
            (token_ids,),
            max_calibration_target_tokens=3,
            max_workspace_bytes=1,
        )
