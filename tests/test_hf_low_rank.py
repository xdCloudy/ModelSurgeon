from __future__ import annotations

import pytest

from modelsurgeon.adapters.huggingface import replace_huggingface_linears_low_rank

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(6, 4, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


def test_exact_rank_two_replacement_records_error_parameters_and_flops() -> None:
    model = _Model()
    left = torch.tensor(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0)))
    right = torch.tensor(((1.0, 2.0, 0.0, -1.0, 0.5, 1.0), (0.0, 1.0, 2.0, 0.5, -1.0, 1.0)))
    with torch.no_grad():
        model.proj.weight.copy_(left @ right)
    values = torch.randn(3, 6)
    expected = model(values)

    report = replace_huggingface_linears_low_rank(model, (("proj", 2),))
    actual = model(values)
    item = report.replacements[0]

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert item.requested_rank == item.effective_rank == 2
    assert item.relative_frobenius_error < 1e-12
    assert item.old_parameters == 28
    assert item.new_parameters == 24
    assert item.old_flops_per_token == 48
    assert item.new_flops_per_token == 40
    assert report.parameter_delta == -4
    assert report.flop_delta_per_token == -8


def test_replacement_rejects_invalid_rank_before_mutating_model() -> None:
    model = _Model()
    original = model.proj

    with pytest.raises(ValueError, match="below both dimensions"):
        replace_huggingface_linears_low_rank(model, (("proj", 4),))

    assert model.proj is original
