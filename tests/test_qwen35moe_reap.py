from __future__ import annotations

import pytest

from modelsurgeon.adapters.huggingface import Qwen35MoeReapError, Qwen35MoeReapObserver

torch = pytest.importorskip("torch")


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
            values[expert].append(torch.linalg.vector_norm(output) * weights[token, slot])
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
