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
    epsilon = 1e-12

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
            source_norm = torch.linalg.vector_norm(source, dim=-1)
            target_norm = torch.linalg.vector_norm(target, dim=-1)
            valid = (source_norm > epsilon) & (target_norm > epsilon)
            similarity = torch.nn.functional.cosine_similarity(source, target, dim=-1, eps=epsilon)
            selected = similarity.masked_select(valid)
            if selected.numel() == 0:
                return
            score_sum = (1.0 - selected).sum().detach()
            count = valid.sum().detach()
            if cosine_sums[layer_index] is None:
                cosine_sums[layer_index] = score_sum
                token_counts[layer_index] = count
            else:
                cosine_sums[layer_index] = cosine_sums[layer_index] + score_sum
                token_counts[layer_index] = token_counts[layer_index] + count

        return hook

    was_training = bool(getattr(model, "training", False))
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
        model.train(was_training)

    scores: list[BlockInfluenceScore] = []
    for layer_index, (score_sum, token_count) in enumerate(
        zip(cosine_sums, token_counts, strict=True)
    ):
        if score_sum is None or token_count is None:
            raise HuggingFaceBlockInfluenceError(
                f"decoder block {layer_index} produced no non-zero hidden-state pairs"
            )
        count = int(token_count.detach().cpu().item())
        value = float((score_sum / token_count).detach().cpu().item())
        if count <= 0 or not math.isfinite(value):
            raise HuggingFaceBlockInfluenceError(
                f"decoder block {layer_index} produced an invalid influence score"
            )
        scores.append(BlockInfluenceScore(layer_index, value, count))
    return tuple(scores)
