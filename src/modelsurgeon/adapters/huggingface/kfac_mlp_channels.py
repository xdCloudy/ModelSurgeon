"""Bounded CPU KFAC scoring for globally coupled gated-MLP channel removal."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from modelsurgeon.features.kfac_sensitivity import (
    LinearRemovalAxis,
    score_structured_linear_removals,
)
from modelsurgeon.features.second_moment import (
    RawSecondMomentCollector,
    RawSecondMomentConfig,
    RawSecondMomentKind,
)


class HuggingFaceMLPKFACError(ValueError):
    """Raised when a model, calibration batch, or KFAC budget is unsupported."""


@dataclass(frozen=True, slots=True)
class HuggingFaceMLPKFACReport:
    """Per-layer and aggregate local costs for deleting each shared MLP channel."""

    model_type: str
    layer_count: int
    hidden_size: int
    intermediate_size: int
    calibration_target_tokens: int
    max_workspace_bytes: int
    planned_workspace_bytes: int
    per_layer_costs: tuple[tuple[float, ...], ...]
    per_layer_activation_weighted_scores: tuple[tuple[float, ...], ...]
    activation_factor_sha256: tuple[str, ...]
    gradient_factor_sha256: tuple[str, ...]

    @property
    def aggregate_costs(self) -> tuple[float, ...]:
        return tuple(
            sum(layer_costs[channel] for layer_costs in self.per_layer_costs)
            for channel in range(self.intermediate_size)
        )

    @property
    def aggregate_activation_weighted_scores(self) -> tuple[float, ...]:
        """Sum WANDA-style |down-projection weight| x activation-norm scores by channel."""
        return tuple(
            sum(layer_scores[channel] for layer_scores in self.per_layer_activation_weighted_scores)
            for channel in range(self.intermediate_size)
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": "sum_of_layerwise_uncorrected_down_projection_kfac_costs",
            "model_type": self.model_type,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "calibration_target_tokens": self.calibration_target_tokens,
            "planned_workspace_bytes": self.planned_workspace_bytes,
            "max_workspace_bytes": self.max_workspace_bytes,
            "aggregate_costs": list(self.aggregate_costs),
            "per_layer_costs": [list(costs) for costs in self.per_layer_costs],
            "aggregate_activation_weighted_scores": list(self.aggregate_activation_weighted_scores),
            "activation_score_method": (
                "sum per-layer input-channel scores: sum_output_abs_weight * "
                "calibration_input_l2_norm"
            ),
            "activation_factor_sha256": list(self.activation_factor_sha256),
            "gradient_factor_sha256": list(self.gradient_factor_sha256),
            "limitations": [
                "Sums independent layer-local KFAC costs; cross-layer curvature is ignored.",
                "Models removal as zeroing a down-projection input column; "
                "applies no compensation.",
                "Scores are advisory proxies, not calibrated held-out quality predictions.",
            ],
        }


def score_huggingface_gated_mlp_channels_kfac(
    model: Any,
    calibration_batches: Iterable[Any],
    *,
    max_calibration_target_tokens: int = 8192,
    max_workspace_bytes: int = 4 * 1024 * 1024 * 1024,
    collector_workspace_bytes: int = 256 * 1024 * 1024,
) -> HuggingFaceMLPKFACReport:
    """Collect calibration moments and rank globally coupled Llama/Qwen2 channels.

    Inputs are unpadded CPU ``input_ids`` tensors shaped ``[1, sequence]``.
    Calibration targets are the next-token labels ``input_ids[:, 1:]``. The
    output-gradient rows are rescaled from mean next-token loss to per-token
    gradients before their second moments are accumulated.

    This scores only the local change induced by deleting each channel from the
    down-projection input in every layer. It does not perform surgery, compensate
    weights, estimate cross-layer effects, or measure held-out quality.
    """

    torch = _torch()
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in {"llama", "qwen2"}:
        raise HuggingFaceMLPKFACError("KFAC gated-MLP adapter supports only Llama and Qwen2")
    layer_count = _positive_config_int(config, "num_hidden_layers")
    hidden_size = _positive_config_int(config, "hidden_size")
    intermediate_size = _positive_config_int(config, "intermediate_size")
    for name, value in (
        ("max_calibration_target_tokens", max_calibration_target_tokens),
        ("max_workspace_bytes", max_workspace_bytes),
        ("collector_workspace_bytes", collector_workspace_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise HuggingFaceMLPKFACError(f"{name} must be a positive integer")

    parameters = tuple(model.parameters())
    if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
        raise HuggingFaceMLPKFACError("KFAC collection is CPU-only; move the full model to CPU")
    modules = dict(model.named_modules())
    down_projections: list[Any] = []
    for layer_index in range(layer_count):
        mlp = modules.get(f"model.layers.{layer_index}.mlp")
        if mlp is None or any(
            getattr(mlp, name, None) is None for name in ("gate_proj", "up_proj")
        ):
            raise HuggingFaceMLPKFACError(
                f"layer {layer_index} lacks the supported gated-MLP projection layout"
            )
        down = getattr(mlp, "down_proj", None)
        weight = getattr(down, "weight", None)
        if (
            weight is None
            or getattr(weight, "ndim", None) != 2
            or tuple(weight.shape) != (hidden_size, intermediate_size)
        ):
            raise HuggingFaceMLPKFACError(
                f"layer {layer_index} down_proj shape does not match config dimensions"
            )
        down_projections.append(down)

    max_batch_rows = min(32, max_calibration_target_tokens)
    activation_config = RawSecondMomentConfig(
        kind=RawSecondMomentKind.INPUT_ACTIVATION,
        dimension=intermediate_size,
        max_observations=max_calibration_target_tokens,
        max_batch_rows=max_batch_rows,
        max_workspace_bytes=collector_workspace_bytes,
    )
    gradient_config = RawSecondMomentConfig(
        kind=RawSecondMomentKind.OUTPUT_GRADIENT,
        dimension=hidden_size,
        max_observations=max_calibration_target_tokens,
        max_batch_rows=max_batch_rows,
        max_workspace_bytes=collector_workspace_bytes,
    )
    planned_workspace = layer_count * (
        activation_config.planned_peak_workspace_bytes
        + gradient_config.planned_peak_workspace_bytes
    )
    if planned_workspace > max_workspace_bytes:
        raise HuggingFaceMLPKFACError(
            f"planned KFAC collector workspace {planned_workspace} exceeds budget "
            f"{max_workspace_bytes} bytes"
        )

    activation_collectors = [
        RawSecondMomentCollector(activation_config) for _ in range(layer_count)
    ]
    gradient_collectors = [RawSecondMomentCollector(gradient_config) for _ in range(layer_count)]
    captured_inputs: list[Any | None] = [None] * layer_count
    captured_outputs: list[Any | None] = [None] * layer_count
    handles = []

    for layer_index, module in enumerate(down_projections):

        def capture_input(_module: Any, args: tuple[Any, ...], *, index: int = layer_index) -> None:
            if len(args) != 1 or not torch.is_tensor(args[0]) or args[0].ndim != 3:
                raise HuggingFaceMLPKFACError(
                    "down_proj input must be a rank-three activation tensor"
                )
            captured_inputs[index] = args[0]

        def capture_output(
            _module: Any, _args: tuple[Any, ...], output: Any, *, index: int = layer_index
        ) -> None:
            if not torch.is_tensor(output) or output.ndim != 3:
                raise HuggingFaceMLPKFACError("down_proj output must be a rank-three tensor")
            captured_outputs[index] = output

        handles.append(module.register_forward_pre_hook(capture_input))
        handles.append(module.register_forward_hook(capture_output))

    training_states = tuple((module, module.training) for module in model.modules())
    model.eval()
    target_token_count = 0
    try:
        for token_ids in calibration_batches:
            if (
                not torch.is_tensor(token_ids)
                or token_ids.device.type != "cpu"
                or token_ids.ndim != 2
                or token_ids.shape[0] != 1
                or token_ids.shape[1] < 2
                or token_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise HuggingFaceMLPKFACError(
                    "calibration batches must be unpadded CPU integer tensors shaped [1, seq>=2]"
                )
            batch_target_tokens = int(token_ids.shape[1]) - 1
            if target_token_count + batch_target_tokens > max_calibration_target_tokens:
                raise HuggingFaceMLPKFACError(
                    "calibration batches exceed max_calibration_target_tokens"
                )
            for index in range(layer_count):
                captured_inputs[index] = None
                captured_outputs[index] = None
            result = model(input_ids=token_ids, labels=token_ids, use_cache=False)
            loss: Any = getattr(result, "loss", None)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise HuggingFaceMLPKFACError("model must return a finite scalar causal-LM loss")
            outputs = tuple(captured_outputs)
            if any(output is None for output in outputs):
                raise HuggingFaceMLPKFACError("not every down_proj ran during calibration forward")
            if any(values is None for values in captured_inputs):
                raise HuggingFaceMLPKFACError(
                    "not every down_proj received calibration activations"
                )
            outputs = tuple(output for output in outputs if output is not None)
            inputs = tuple(values for values in captured_inputs if values is not None)
            gradients = torch.autograd.grad(loss, outputs, allow_unused=False)
            for layer_index, (activations, gradients_for_output) in enumerate(
                zip(inputs, gradients, strict=True)
            ):
                expected_shape = (1, token_ids.shape[1], intermediate_size)
                if tuple(activations.shape) != expected_shape:
                    raise HuggingFaceMLPKFACError(
                        f"layer {layer_index} down_proj input shape does not match config"
                    )
                if tuple(gradients_for_output.shape) != (1, token_ids.shape[1], hidden_size):
                    raise HuggingFaceMLPKFACError(
                        f"layer {layer_index} down_proj output gradient shape does not match config"
                    )
                activation_rows = activations.detach()[:, :-1, :].reshape(-1, intermediate_size)
                gradient_rows = (
                    gradients_for_output.detach()[:, :-1, :].reshape(-1, hidden_size)
                    * batch_target_tokens
                )
                activation_collectors[layer_index].update(activation_rows.float().numpy())
                gradient_collectors[layer_index].update(gradient_rows.float().numpy())
            target_token_count += batch_target_tokens
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states:
            module.training = was_training

    if target_token_count <= 0:
        raise HuggingFaceMLPKFACError(
            "at least one calibration batch with a target token is required"
        )

    activation_summaries = []
    gradient_summaries = []
    for activation_collector, gradient_collector in zip(
        activation_collectors, gradient_collectors, strict=True
    ):
        activation_summaries.append(activation_collector.summary())
        gradient_summaries.append(gradient_collector.summary())
    del activation_collectors, gradient_collectors

    singleton_groups = tuple((index,) for index in range(intermediate_size))
    per_layer_costs: list[tuple[float, ...]] = []
    per_layer_activation_weighted_scores: list[tuple[float, ...]] = []
    for down, activation, gradient in zip(
        down_projections, activation_summaries, gradient_summaries, strict=True
    ):
        weights = down.weight.detach().float().numpy()
        activation_l2_norms = (
            activation.matrix.diagonal().clip(min=0.0) * activation.observation_count
        ) ** 0.5
        per_layer_activation_weighted_scores.append(
            tuple((abs(weights) * activation_l2_norms[None, :]).sum(axis=0).tolist())
        )
        report = score_structured_linear_removals(
            weights,
            activation.matrix,
            gradient.matrix,
            singleton_groups,
            axis=LinearRemovalAxis.INPUT_COLUMNS,
            max_workspace_bytes=collector_workspace_bytes,
        )
        per_layer_costs.append(
            tuple(candidate.estimated_loss_increase for candidate in report.candidate_costs)
        )

    return HuggingFaceMLPKFACReport(
        model_type=model_type,
        layer_count=layer_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        calibration_target_tokens=target_token_count,
        max_workspace_bytes=max_workspace_bytes,
        planned_workspace_bytes=planned_workspace,
        per_layer_costs=tuple(per_layer_costs),
        per_layer_activation_weighted_scores=tuple(per_layer_activation_weighted_scores),
        activation_factor_sha256=tuple(item.matrix_sha256 for item in activation_summaries),
        gradient_factor_sha256=tuple(item.matrix_sha256 for item in gradient_summaries),
    )


def _positive_config_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HuggingFaceMLPKFACError(f"model config {name} must be a positive integer")
    return value


def _torch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as error:
        raise HuggingFaceMLPKFACError(
            "Hugging Face KFAC collection requires PyTorch; install the HF extra"
        ) from error
