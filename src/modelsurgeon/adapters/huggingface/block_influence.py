"""ShortGPT-style block influence measurement for supported decoder stacks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class HuggingFaceBlockInfluenceError(ValueError):
    """Raised when a model cannot provide comparable decoder-block activations."""


@dataclass(frozen=True, slots=True)
class BlockInfluenceScore:
    layer_index: int
    score: float
    token_count: int


def measure_huggingface_block_influence(
    model: Any,
    token_chunks: Iterable[tuple[int, ...]],
    *,
    input_device: Any,
) -> tuple[BlockInfluenceScore, ...]:
    """Measure mean ``1 - cosine(input, output)`` for every decoder block.

    The calculation is row-wise over corresponding token states. This preserves
    ShortGPT's Block Influence definition without constructing its reference
    implementation's quadratic token-by-token similarity matrix.
    """

    torch = __import__("torch")
    blocks = getattr(getattr(model, "model", None), "layers", None)
    configured_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if (
        not isinstance(blocks, torch.nn.ModuleList)
        or not isinstance(configured_layers, int)
        or isinstance(configured_layers, bool)
        or configured_layers != len(blocks)
        or configured_layers <= 0
    ):
        raise HuggingFaceBlockInfluenceError(
            "block influence requires model.layers to match config.num_hidden_layers"
        )

    chunks = tuple(token_chunks)
    if not chunks or any(
        not chunk
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in chunk
        )
        for chunk in chunks
    ):
        raise HuggingFaceBlockInfluenceError("block influence requires valid token chunks")

    cosine_sums: list[Any | None] = [None] * configured_layers
    token_counts: list[Any | None] = [None] * configured_layers
    nonfinite_activations: list[Any | None] = [None] * configured_layers

    def capture(layer_index: int) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs:
                raise HuggingFaceBlockInfluenceError(
                    f"decoder block {layer_index} received no hidden-state input"
                )
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, (tuple, list)) else output
            if (
                not torch.is_tensor(hidden_in)
                or not torch.is_tensor(hidden_out)
                or hidden_in.ndim != 3
                or hidden_out.shape != hidden_in.shape
            ):
                raise HuggingFaceBlockInfluenceError(
                    f"decoder block {layer_index} returned incompatible hidden states"
                )

            source = hidden_in.detach().float()
            target = hidden_out.detach().float()
            nonfinite = (~torch.isfinite(source)).any() | (~torch.isfinite(target)).any()
            if nonfinite_activations[layer_index] is None:
                nonfinite_activations[layer_index] = nonfinite
            else:
                nonfinite_activations[layer_index] = (
                    nonfinite_activations[layer_index] | nonfinite
                )
            source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
            target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
            similarity = (source * target).sum(dim=-1) / (
                source_norm.squeeze(-1) * target_norm.squeeze(-1)
            )
            # ShortGPT maps the undefined cosine for a zero-norm pair to 0.5.
            # Keep those rows in the mean rather than dropping and reweighting them.
            similarity = torch.nan_to_num(similarity, nan=0.5)
            score_sum = (1.0 - similarity).sum().detach()
            count = score_sum.new_tensor(hidden_in.shape[0] * hidden_in.shape[1], dtype=torch.long)
            if cosine_sums[layer_index] is None:
                cosine_sums[layer_index] = score_sum
                token_counts[layer_index] = count
            else:
                cosine_sums[layer_index] = cosine_sums[layer_index] + score_sum
                token_counts[layer_index] = token_counts[layer_index] + count

        return hook

    module_training_states = tuple((module, module.training) for module in model.modules())
    handles = [block.register_forward_hook(capture(index)) for index, block in enumerate(blocks)]
    try:
        model.eval()
        with torch.inference_mode():
            for chunk in chunks:
                input_ids = torch.tensor((chunk,), dtype=torch.long, device=input_device)
                model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in module_training_states:
            module.training = was_training

    for layer_index, nonfinite in enumerate(nonfinite_activations):
        if nonfinite is not None and bool(nonfinite.detach().cpu().item()):
            raise HuggingFaceBlockInfluenceError(
                f"decoder block {layer_index} produced non-finite hidden states"
            )

    scores: list[BlockInfluenceScore] = []
    for layer_index, (score_sum, token_count) in enumerate(
        zip(cosine_sums, token_counts, strict=True)
    ):
        if score_sum is None or token_count is None:
            raise HuggingFaceBlockInfluenceError(
                f"decoder block {layer_index} produced no hidden-state pairs"
            )
        count = int(token_count.detach().cpu().item())
        value = float((score_sum / token_count).detach().cpu().item())
        if count <= 0 or not math.isfinite(value):
            raise HuggingFaceBlockInfluenceError(
                f"decoder block {layer_index} produced an invalid influence score"
            )
        scores.append(BlockInfluenceScore(layer_index, value, count))
    return tuple(scores)
