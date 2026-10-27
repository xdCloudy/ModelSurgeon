"""First-order loss sensitivity for transformer-layer deletion candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast


class HuggingFaceLayerSensitivityError(ValueError):
    """Raised when layer sensitivities cannot be measured safely."""


@dataclass(frozen=True, slots=True)
class LayerSensitivityScore:
    """Predicted held-out-loss change per token if one block becomes an identity."""

    layer_index: int
    predicted_nll_delta: float
    absolute_token_sensitivity: float
    token_count: int


def measure_huggingface_layer_sensitivity(
    model: Any,
    token_chunks: Iterable[tuple[int, ...]],
    *,
    input_device: Any,
) -> tuple[LayerSensitivityScore, ...]:
    """Estimate deletion impact using the first-order loss directional derivative.

    For each decoder block output ``h``, deletion replaces it with the block input
    ``x``. The calibration estimate is ``grad(loss, h) · (x - h)``. Scores are
    accumulated from summed next-token loss and divided by the number of evaluated
    target tokens. The absolute per-token contribution is reported separately to
    expose cancellation in the signed sum. Lower values predict less calibration loss
    increase. This
    approximation does not include nonlinear downstream effects and must be
    compared against physical deletion on independent held-out data.
    """

    torch = __import__("torch")
    blocks = getattr(getattr(model, "model", None), "layers", None)
    configured_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if (
        not isinstance(blocks, torch.nn.ModuleList)
        or not isinstance(configured_layers, int)
        or isinstance(configured_layers, bool)
        or configured_layers <= 0
        or configured_layers != len(blocks)
    ):
        raise HuggingFaceLayerSensitivityError(
            "layer sensitivity requires model.model.layers to match config.num_hidden_layers"
        )

    chunks = tuple(token_chunks)
    if not chunks or any(
        len(chunk) < 2
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in chunk
        )
        for chunk in chunks
    ):
        raise HuggingFaceLayerSensitivityError(
            "layer sensitivity requires nonempty token chunks with at least two valid tokens"
        )

    score_sums = [0.0] * configured_layers
    absolute_score_sums = [0.0] * configured_layers
    token_count = 0
    captured: list[tuple[Any, Any] | None] = [None] * configured_layers
    handles: list[Any] = []

    def capture(layer_index: int) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs:
                raise HuggingFaceLayerSensitivityError(
                    f"decoder block {layer_index} received no hidden-state input"
                )
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, (tuple, list)) else output
            if (
                not torch.is_tensor(hidden_in)
                or not torch.is_tensor(hidden_out)
                or hidden_in.ndim != 3
                or hidden_out.shape != hidden_in.shape
                or not hidden_out.requires_grad
            ):
                raise HuggingFaceLayerSensitivityError(
                    f"decoder block {layer_index} returned incompatible hidden states"
                )
            captured[layer_index] = (hidden_in, hidden_out)

        return hook

    training_states = tuple((module, module.training) for module in model.modules())
    try:
        for layer_index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(capture(layer_index)))
        model.eval()
        for chunk in chunks:
            captured[:] = [None] * configured_layers
            model.zero_grad(set_to_none=True)
            input_ids = torch.tensor((chunk,), dtype=torch.long, device=input_device)
            output = model(input_ids=input_ids, use_cache=False)
            logits = cast(Any, getattr(output, "logits", None))
            if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[1] != len(chunk):
                raise HuggingFaceLayerSensitivityError("model returned incompatible causal logits")
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
                input_ids[:, 1:].contiguous().view(-1),
                reduction="sum",
            )
            if any(pair is None for pair in captured):
                raise HuggingFaceLayerSensitivityError(
                    "one or more decoder blocks produced no sensitivity evidence"
                )
            pairs = tuple(pair for pair in captured if pair is not None)
            gradients = torch.autograd.grad(loss, tuple(pair[1] for pair in pairs))
            current_tokens = int(input_ids[:, 1:].numel())
            for layer_index, (pair, gradient) in enumerate(zip(pairs, gradients, strict=True)):
                hidden_in, hidden_out = pair
                delta = hidden_in.detach().float() - hidden_out.detach().float()
                per_token_contribution = (gradient.detach().float() * delta).sum(dim=-1)
                contribution = per_token_contribution.sum()
                absolute_contribution = per_token_contribution.abs().sum()
                value = float(contribution.cpu().item())
                absolute_value = float(absolute_contribution.cpu().item())
                if not math.isfinite(value) or not math.isfinite(absolute_value):
                    raise HuggingFaceLayerSensitivityError(
                        f"decoder block {layer_index} produced a non-finite sensitivity"
                    )
                score_sums[layer_index] += value
                absolute_score_sums[layer_index] += absolute_value
            token_count += current_tokens
        if token_count <= 0:
            raise HuggingFaceLayerSensitivityError("layer sensitivity observed no target tokens")
    finally:
        for handle in reversed(handles):
            handle.remove()
        model.zero_grad(set_to_none=True)
        for module, was_training in training_states:
            module.training = was_training

    scores = tuple(
        LayerSensitivityScore(index, total / token_count, absolute_score / token_count, token_count)
        for index, (total, absolute_score) in enumerate(
            zip(score_sums, absolute_score_sums, strict=True)
        )
    )
    if any(
        not math.isfinite(score.predicted_nll_delta)
        or not math.isfinite(score.absolute_token_sensitivity)
        for score in scores
    ):
        raise HuggingFaceLayerSensitivityError("layer sensitivity produced a non-finite score")
    return scores
