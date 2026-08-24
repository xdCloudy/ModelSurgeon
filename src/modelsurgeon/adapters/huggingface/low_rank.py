"""Bounded SVD replacement of selected Hugging Face linear modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

HF_LOW_RANK_SCHEMA_VERSION: Final[int] = 1


class HuggingFaceLowRankError(ValueError):
    """Raised when a selected linear or SVD workspace is unsupported."""


@dataclass(frozen=True, slots=True)
class LowRankReplacement:
    module_path: str
    requested_rank: int
    effective_rank: int
    relative_frobenius_error: float
    old_parameters: int
    new_parameters: int
    old_flops_per_token: int
    new_flops_per_token: int

    @property
    def parameter_delta(self) -> int:
        return self.new_parameters - self.old_parameters

    @property
    def flop_delta_per_token(self) -> int:
        return self.new_flops_per_token - self.old_flops_per_token


@dataclass(frozen=True, slots=True)
class LowRankReplacementReport:
    replacements: tuple[LowRankReplacement, ...]
    schema_version: int = HF_LOW_RANK_SCHEMA_VERSION

    @property
    def parameter_delta(self) -> int:
        return sum(item.parameter_delta for item in self.replacements)

    @property
    def flop_delta_per_token(self) -> int:
        return sum(item.flop_delta_per_token for item in self.replacements)


def replace_huggingface_linears_low_rank(
    model: Any,
    ranks_by_module: tuple[tuple[str, int], ...],
    *,
    max_matrix_elements: int = 16_777_216,
    max_workspace_bytes: int = 536_870_912,
) -> LowRankReplacementReport:
    """Replace canonical selected Linear modules with exact two-factor SVD approximations."""

    if not ranks_by_module or tuple(path for path, _ in ranks_by_module) != tuple(
        sorted({path for path, _ in ranks_by_module})
    ):
        raise HuggingFaceLowRankError("low-rank module paths must be non-empty, unique, canonical")
    if max_matrix_elements <= 0 or max_workspace_bytes <= 0:
        raise HuggingFaceLowRankError("low-rank workspace limits must be positive")
    torch = __import__("torch")
    modules = dict(model.named_modules())
    prepared: list[tuple[str, Any, Any, int]] = []
    for path, rank in ranks_by_module:
        module = modules.get(path)
        weight = getattr(module, "weight", None)
        if not isinstance(module, torch.nn.Linear) or weight is None:
            raise HuggingFaceLowRankError(f"selected module {path!r} is not torch.nn.Linear")
        rows, columns = (int(weight.shape[0]), int(weight.shape[1]))
        if rank <= 0 or rank >= min(rows, columns):
            raise HuggingFaceLowRankError(f"rank for {path!r} must be below both dimensions")
        elements = rows * columns
        workspace = elements * 8 * 4 + (rows + columns) * rank * 8
        if elements > max_matrix_elements or workspace > max_workspace_bytes:
            raise HuggingFaceLowRankError(f"selected module {path!r} exceeds SVD workspace")
        parent_path, _, leaf = path.rpartition(".")
        parent = modules.get(parent_path) if parent_path else model
        if parent is None or not hasattr(parent, leaf):
            raise HuggingFaceLowRankError(f"selected module parent for {path!r} is missing")
        prepared.append((path, parent, module, rank))
    reports: list[LowRankReplacement] = []
    for path, parent, module, rank in prepared:
        weight = module.weight.detach().to(device="cpu", dtype=torch.float64)
        u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        left = u[:, :rank] * singular[:rank]
        right = vh[:rank, :]
        factorized = _factorized_linear(torch, module, rank, left, right)
        setattr(parent, path.rpartition(".")[2], factorized)
        approximation = left @ right
        denominator = float(torch.linalg.vector_norm(weight))
        error = float(torch.linalg.vector_norm(weight - approximation))
        relative = 0.0 if denominator == 0.0 else error / denominator
        old_parameters = int(module.weight.numel()) + (
            0 if module.bias is None else int(module.bias.numel())
        )
        new_parameters = sum(int(parameter.numel()) for parameter in factorized.parameters())
        rows, columns = int(module.weight.shape[0]), int(module.weight.shape[1])
        reports.append(
            LowRankReplacement(
                path,
                rank,
                rank,
                relative,
                old_parameters,
                new_parameters,
                2 * rows * columns,
                2 * rank * (rows + columns),
            )
        )
    return LowRankReplacementReport(tuple(reports))


def _factorized_linear(torch: Any, module: Any, rank: int, left: Any, right: Any) -> Any:
    class _LowRankLinear(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = int(module.in_features)
            self.out_features = int(module.out_features)
            self.rank = rank
            self.down = torch.nn.Linear(self.in_features, rank, bias=False)
            self.up = torch.nn.Linear(rank, self.out_features, bias=module.bias is not None)

        def forward(self, values: Any) -> Any:
            return self.up(self.down(values))

    result = _LowRankLinear().to(device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        result.down.weight.copy_(right.to(device=module.weight.device, dtype=module.weight.dtype))
        result.up.weight.copy_(left.to(device=module.weight.device, dtype=module.weight.dtype))
        if module.bias is not None:
            result.up.bias.copy_(module.bias.detach())
    result.train(module.training)
    return result
