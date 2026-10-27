from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from modelsurgeon.adapters.huggingface import Qwen35MoeReapError, Qwen35MoeReapObserver

try:
    torch = import_module("torch")
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        pytest.skip(
            "Qwen3.5 REAP tests require the optional PyTorch dependency", allow_module_level=True
        )
    raise


def _qwen35_module():
    module_name = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {
            "transformers",
            "transformers.models.qwen3_5_moe",
            module_name,
        }:
            pytest.skip("Qwen3.5 REAP tests require Transformers with Qwen3.5 support")
        raise


def test_missing_qwen35_model_module_skips_cleanly(monkeypatch) -> None:
    module_name = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"

    def import_missing_module(_name: str):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(f"{__name__}.import_module", import_missing_module)

    with pytest.raises(pytest.skip.Exception, match=r"Qwen3\.5 support"):
        _qwen35_module()


class TinyExperts(torch.nn.Module):
    def __init__(
        self, num_experts: int = 4, hidden_size: int = 3, intermediate_size: int = 5
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


def dense_reference(module, hidden_states, indices, weights, valid_mask):
    values = [[] for _ in range(module.num_experts)]
    for token in range(hidden_states.shape[0]):
        if not bool(valid_mask[token]):
            continue
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            projected = torch.nn.functional.linear(
                hidden_states[token], module.gate_up_proj[expert]
            )
            gate, up = projected.chunk(2, dim=-1)
            intermediate = module.act_fn(gate) * up
            output = torch.nn.functional.linear(intermediate, module.down_proj[expert])
            values[expert].append(
                torch.linalg.vector_norm(output.float()) * weights[token, slot].float()
            )
    means = [
        torch.stack(expert_values).mean().item() if expert_values else None
        for expert_values in values
    ]
    counts = [len(expert_values) for expert_values in values]
    return means, counts


def test_sparse_scores_match_dense_reference_and_exclude_padding() -> None:
    torch.manual_seed(20261026)
    module = TinyExperts()
    hidden_states = torch.randn(7, 3)
    indices = torch.tensor([[0, 1], [2, 0], [1, 3], [2, 3], [0, 2], [1, 3], [0, 1]])
    weights = torch.rand(7, 2)
    weights /= weights.sum(dim=-1, keepdim=True)
    valid_mask = torch.tensor([True, True, False, True, True, False, True])
    expected_scores, expected_counts = dense_reference(
        module, hidden_states, indices, weights, valid_mask
    )

    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    observer.update(0, module, hidden_states, indices, weights, valid_token_mask=valid_mask)
    result = observer.report()

    assert result.routed_counts_by_layer == (tuple(expected_counts),)
    for actual, expected in zip(result.scores_by_layer[0], expected_scores, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_unseen_experts_are_none_and_counts_accumulate_across_batches() -> None:
    module = TinyExperts(num_experts=3)
    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=3)
    hidden = torch.ones(2, 3)
    indices = torch.tensor([[0], [0]])
    weights = torch.ones(2, 1)

    observer.update(0, module, hidden, indices, weights, valid_token_mask=None)
    observer.update(0, module, hidden, indices, weights, valid_token_mask=None)
    result = observer.report()

    assert result.routed_counts_by_layer == ((4, 0, 0),)
    assert result.scores_by_layer[0][0] is not None
    assert result.scores_by_layer[0][1:] == (None, None)
    assert result.to_record()["schema_version"] == 1


def test_invalid_duplicate_or_out_of_range_routes_are_rejected_without_partial_update() -> None:
    module = TinyExperts()
    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    hidden = torch.ones(1, 3)
    weights = torch.ones(1, 2)

    for indices in (torch.tensor([[1, 1]]), torch.tensor([[1, 4]])):
        with pytest.raises(Qwen35MoeReapError):
            observer.update(0, module, hidden, indices, weights, valid_token_mask=None)
        assert observer.report().routed_counts_by_layer == ((0, 0, 0, 0),)


@pytest.mark.parametrize(
    "weights",
    (torch.tensor([[0.2, 0.3]]), torch.tensor([[1.1, -0.1]]), torch.tensor([[1.1]])),
)
def test_unnormalized_or_out_of_range_router_weights_are_rejected(weights) -> None:
    module = TinyExperts()
    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    hidden = torch.ones(1, 3)
    indices = torch.tensor([[0, 1]]) if weights.shape[1] == 2 else torch.tensor([[0]])

    with pytest.raises(Qwen35MoeReapError, match="normalized per token"):
        observer.update(0, module, hidden, indices, weights, valid_token_mask=None)

    assert observer.report().routed_counts_by_layer == ((0, 0, 0, 0),)


def test_forward_hooks_observe_and_close_removes_hooks() -> None:
    module = TinyExperts()
    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    indices = torch.tensor([[0], [2]])
    weights = torch.ones(2, 1)
    observer.register_hooks(
        {0: module},
        valid_token_mask_provider=lambda _layer, _args: torch.ones(2, dtype=torch.bool),
    )
    module(torch.ones(2, 3), indices, weights)
    assert observer.report().routed_counts_by_layer == ((1, 0, 1, 0),)
    observer.close()
    observer.close()
    assert len(module._forward_hooks) == 0


def test_real_transformers_qwen35_fused_expert_contract() -> None:
    qwen_module = _qwen35_module()
    torch.manual_seed(20261026)
    module = qwen_module.Qwen3_5MoeExperts(
        SimpleNamespace(
            num_experts=4,
            hidden_size=3,
            moe_intermediate_size=5,
            hidden_act="silu",
            _experts_implementation="eager",
        )
    )
    torch.nn.init.normal_(module.gate_up_proj)
    torch.nn.init.normal_(module.down_proj)
    hidden_states = torch.randn(7, 3)
    indices = torch.tensor([[0, 1], [2, 0], [1, 3], [2, 3], [0, 2], [1, 3], [0, 1]])
    weights = torch.rand(7, 2)
    weights /= weights.sum(dim=-1, keepdim=True)
    valid_mask = torch.tensor([True, True, False, True, True, False, True])
    expected_scores, expected_counts = dense_reference(
        module, hidden_states, indices, weights, valid_mask
    )

    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    observer.register_hooks(
        {0: module},
        valid_token_mask_provider=lambda _layer, _args: valid_mask,
    )
    output = module(hidden_states, indices, weights)
    result = observer.report()

    assert output.shape == hidden_states.shape
    assert result.routed_counts_by_layer == (tuple(expected_counts),)
    for actual, expected in zip(result.scores_by_layer[0], expected_scores, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6)
    observer.close()
    assert len(module._forward_hooks) == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_qwen35_router_passes_final_renormalized_top_k_weights(dtype: Any) -> None:
    qwen_module = _qwen35_module()
    torch.manual_seed(20261027)
    config = SimpleNamespace(
        num_experts=4,
        hidden_size=3,
        num_experts_per_tok=2,
        moe_intermediate_size=5,
        hidden_act="silu",
        _experts_implementation="eager",
    )
    router = qwen_module.Qwen3_5MoeTopKRouter(config)
    experts = qwen_module.Qwen3_5MoeExperts(config)
    router.to(dtype=dtype)
    experts.to(dtype=dtype)
    torch.nn.init.normal_(router.weight)
    torch.nn.init.normal_(experts.gate_up_proj)
    torch.nn.init.normal_(experts.down_proj)
    hidden_states = torch.randn(7, 3, dtype=dtype)
    _, top_k_weights, top_k_index = router(hidden_states)

    assert torch.allclose(
        top_k_weights.sum(dim=-1).float(),
        torch.ones(hidden_states.shape[0]),
        atol=float(torch.finfo(dtype).eps) * top_k_weights.shape[1],
        rtol=0.0,
    )
    expected_scores, expected_counts = dense_reference(
        experts,
        hidden_states,
        top_k_index,
        top_k_weights,
        torch.ones(hidden_states.shape[0], dtype=torch.bool),
    )

    observer = Qwen35MoeReapObserver(num_layers=1, num_experts=4)
    observer.update(
        0,
        experts,
        hidden_states,
        top_k_index,
        top_k_weights,
        valid_token_mask=None,
    )
    result = observer.report()

    assert result.routed_counts_by_layer == (tuple(expected_counts),)
    for actual, expected in zip(result.scores_by_layer[0], expected_scores, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6)
