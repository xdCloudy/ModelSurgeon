"""Sparse REAP-style expert importance collection for Qwen3.5 MoE models.

This module independently implements the REAP scoring rule for the fused
Qwen3.5 expert layout. It evaluates only token/expert pairs selected by the
router. PyTorch remains an optional dependency and is imported only when an
observation is recorded.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any


class Qwen35MoeReapError(ValueError):
    """A Qwen3.5 MoE calibration batch violates the supported tensor contract."""


class Qwen35MoeReapDependencyError(RuntimeError):
    """Raised when the optional PyTorch dependency is not installed."""


@dataclass(frozen=True, slots=True)
class Qwen35MoeReapResult:
    """Layer-major means and routed-token counts from the calibration batches."""

    scores_by_layer: tuple[tuple[float | None, ...], ...]
    routed_counts_by_layer: tuple[tuple[int, ...], ...]

    def to_record(self) -> dict[str, object]:
        """Return JSON-compatible scores without inventing values for unseen experts."""

        return {
            "schema_version": 1,
            "metric": "mean_router_weighted_expert_output_l2",
            "scores_by_layer": [list(layer) for layer in self.scores_by_layer],
            "routed_counts_by_layer": [list(layer) for layer in self.routed_counts_by_layer],
        }

    @classmethod
    def from_record(cls, record: object) -> Qwen35MoeReapResult:
        """Load a score record, rejecting malformed or ambiguous evidence."""

        required = {
            "schema_version",
            "metric",
            "scores_by_layer",
            "routed_counts_by_layer",
        }
        if (
            not isinstance(record, Mapping)
            or set(record) != required
            or isinstance(record.get("schema_version"), bool)
            or not isinstance(record.get("schema_version"), int)
            or record.get("schema_version") != 1
            or record.get("metric") != "mean_router_weighted_expert_output_l2"
        ):
            raise Qwen35MoeReapError("unsupported or malformed Qwen3.5 REAP score record")

        raw_scores = record["scores_by_layer"]
        raw_counts = record["routed_counts_by_layer"]
        if (
            not isinstance(raw_scores, list)
            or not raw_scores
            or not isinstance(raw_counts, list)
            or len(raw_counts) != len(raw_scores)
        ):
            raise Qwen35MoeReapError(
                "REAP score and route-count layers must be non-empty and align"
            )

        scores_by_layer: list[tuple[float | None, ...]] = []
        routed_counts_by_layer: list[tuple[int, ...]] = []
        expert_count: int | None = None
        for layer, (scores, counts) in enumerate(zip(raw_scores, raw_counts, strict=True)):
            if (
                not isinstance(scores, list)
                or not scores
                or not isinstance(counts, list)
                or len(scores) != len(counts)
                or (expert_count is not None and len(scores) != expert_count)
            ):
                raise Qwen35MoeReapError(
                    f"REAP score and route-count experts for layer {layer} must align"
                )
            expert_count = len(scores)
            validated_scores: list[float | None] = []
            validated_counts: list[int] = []
            for expert, (score, count) in enumerate(zip(scores, counts, strict=True)):
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise Qwen35MoeReapError(
                        f"route count for layer {layer}, expert {expert} must be "
                        "a nonnegative integer"
                    )
                if score is None:
                    if count != 0:
                        raise Qwen35MoeReapError(
                            f"unseen expert at layer {layer}, expert {expert} has routed tokens"
                        )
                    validated_scores.append(None)
                else:
                    try:
                        numeric_score = float(score)
                    except (OverflowError, TypeError, ValueError):
                        numeric_score = math.nan
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(numeric_score)
                        or numeric_score < 0
                        or count == 0
                    ):
                        raise Qwen35MoeReapError(
                            f"score for layer {layer}, expert {expert} must be finite, "
                            "nonnegative, and have routed tokens"
                        )
                    validated_scores.append(numeric_score)
                validated_counts.append(count)
            scores_by_layer.append(tuple(validated_scores))
            routed_counts_by_layer.append(tuple(validated_counts))
        return cls(tuple(scores_by_layer), tuple(routed_counts_by_layer))


MaskProvider = Callable[[int, tuple[Any, ...]], Any | None]


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise Qwen35MoeReapDependencyError(
            "Qwen3.5 REAP scoring requires PyTorch; install ModelSurgeon's optional hf extra"
        ) from exc


class Qwen35MoeReapObserver:
    """Collect mean ``router_weight * ||expert_output||_2`` per Qwen3.5 expert.

    Pass the three positional inputs of ``Qwen3_5MoeExperts.forward`` to
    :meth:`update`, or register hooks on the per-layer ``mlp.experts`` modules.
    When using hooks, a mask provider is mandatory so callers explicitly choose
    how padding tokens are handled. Return ``None`` only for batches where every
    routed token is valid.
    """

    def __init__(self, *, num_layers: int, num_experts: int) -> None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise Qwen35MoeReapError("num_layers must be a positive integer")
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
            raise Qwen35MoeReapError("num_experts must be a positive integer")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._weighted_sums = [[0.0] * num_experts for _ in range(num_layers)]
        self._routed_counts = [[0] * num_experts for _ in range(num_layers)]
        self._handles: list[Any] = []

    def update(
        self,
        layer: int,
        experts_module: Any,
        hidden_states: Any,
        top_k_index: Any,
        top_k_weights: Any,
        *,
        valid_token_mask: Any | None,
    ) -> None:
        """Accumulate one routed batch using sparse per-expert token slices."""

        torch = _torch()
        functional = torch.nn.functional
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < self.num_layers
        ):
            raise Qwen35MoeReapError("layer index is outside the observer's configured range")
        if not all(torch.is_tensor(value) for value in (hidden_states, top_k_index, top_k_weights)):
            raise Qwen35MoeReapError(
                "hidden states, expert indices, and router weights must be tensors"
            )
        if hidden_states.ndim != 2:
            raise Qwen35MoeReapError("hidden states must have shape [tokens, hidden_size]")
        if not hidden_states.is_floating_point() or not top_k_weights.is_floating_point():
            raise Qwen35MoeReapError(
                "hidden states and router weights must use floating-point dtypes"
            )
        if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
            raise Qwen35MoeReapError(
                "expert indices and router weights must have matching [tokens, top_k] shapes"
            )
        token_count, hidden_size = hidden_states.shape
        if top_k_index.shape[0] != token_count:
            raise Qwen35MoeReapError("routed token count must match hidden-state count")
        if top_k_index.dtype not in (torch.int32, torch.int64):
            raise Qwen35MoeReapError("expert indices must use int32 or int64 dtype")
        if token_count < 1 or top_k_index.shape[1] < 1:
            raise Qwen35MoeReapError("calibration batches must contain routed tokens")
        if (
            hidden_states.device != top_k_index.device
            or hidden_states.device != top_k_weights.device
        ):
            raise Qwen35MoeReapError(
                "hidden states, indices, and router weights must share a device"
            )

        num_experts = getattr(experts_module, "num_experts", None)
        gate_up_proj = getattr(experts_module, "gate_up_proj", None)
        down_proj = getattr(experts_module, "down_proj", None)
        act_fn = getattr(experts_module, "act_fn", None)
        if num_experts != self.num_experts:
            raise Qwen35MoeReapError(
                "expert module count does not match the observer configuration"
            )
        if not all(torch.is_tensor(value) for value in (gate_up_proj, down_proj)) or not callable(
            act_fn
        ):
            raise Qwen35MoeReapError(
                "expert module must expose fused gate_up_proj, down_proj, and act_fn"
            )
        assert gate_up_proj is not None and down_proj is not None
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise Qwen35MoeReapError("fused expert weights must be rank-three tensors")
        expert_count, gate_up_width, input_width = gate_up_proj.shape
        down_count, output_width, intermediate_width = down_proj.shape
        if (
            expert_count != self.num_experts
            or down_count != self.num_experts
            or input_width != hidden_size
            or gate_up_width % 2
            or intermediate_width * 2 != gate_up_width
            or output_width != hidden_size
        ):
            raise Qwen35MoeReapError(
                "fused expert weight dimensions do not match Qwen3.5 routing inputs"
            )
        if gate_up_proj.device != hidden_states.device or down_proj.device != hidden_states.device:
            raise Qwen35MoeReapError("expert weights and routing inputs must share a device")
        if gate_up_proj.dtype != hidden_states.dtype or down_proj.dtype != hidden_states.dtype:
            raise Qwen35MoeReapError("expert weights and hidden states must share a dtype")

        invalid_index = ((top_k_index < 0) | (top_k_index >= self.num_experts)).any()
        if top_k_index.shape[1] > 1:
            ordered = torch.sort(top_k_index, dim=-1).values
            duplicate_index = (ordered[:, 1:] == ordered[:, :-1]).any()
        else:
            duplicate_index = torch.zeros((), dtype=torch.bool, device=top_k_index.device)
        invalid_weight = (~torch.isfinite(top_k_weights) | (top_k_weights < 0)).any()
        if bool((invalid_index | duplicate_index).item()):
            raise Qwen35MoeReapError("expert indices must be in range and unique per token")
        if bool(invalid_weight.item()):
            raise Qwen35MoeReapError("router weights must be finite and non-negative")

        if valid_token_mask is None:
            valid_token_mask = torch.ones(
                token_count, dtype=torch.bool, device=hidden_states.device
            )
        elif (
            not torch.is_tensor(valid_token_mask)
            or valid_token_mask.shape != (token_count,)
            or valid_token_mask.dtype != torch.bool
            or valid_token_mask.device != hidden_states.device
        ):
            raise Qwen35MoeReapError(
                "valid_token_mask must be a bool tensor matching routed tokens and device"
            )
        assert valid_token_mask is not None

        batch_sums = torch.zeros(self.num_experts, dtype=torch.float64, device=hidden_states.device)
        has_nonfinite_output = torch.zeros((), dtype=torch.bool, device=hidden_states.device)
        with torch.inference_mode():
            top_k = top_k_index.shape[1]
            valid_route_positions = torch.nonzero(
                valid_token_mask.repeat_interleave(top_k), as_tuple=True
            )[0]
            expert_ids = top_k_index.reshape(-1)[valid_route_positions].to(torch.int64)
            route_order = torch.argsort(expert_ids, stable=True)
            sorted_route_positions = valid_route_positions[route_order]
            route_counts = torch.bincount(expert_ids, minlength=self.num_experts).cpu().tolist()

            start = 0
            for expert, routed_count in enumerate(route_counts):
                stop = start + routed_count
                expert_routes = sorted_route_positions[start:stop]
                start = stop
                if routed_count == 0:
                    continue

                token_indices = torch.div(expert_routes, top_k, rounding_mode="floor")
                top_k_positions = torch.remainder(expert_routes, top_k)

                projected = functional.linear(hidden_states[token_indices], gate_up_proj[expert])
                gate, up = projected.chunk(2, dim=-1)
                intermediate = act_fn(gate) * up
                expert_output = functional.linear(intermediate, down_proj[expert])
                output_norm = torch.linalg.vector_norm(expert_output.float(), ord=2, dim=-1)
                weighted_norm = output_norm * top_k_weights[token_indices, top_k_positions].float()
                has_nonfinite_output |= (~torch.isfinite(weighted_norm)).any()
                batch_sums[expert] = weighted_norm.sum(dtype=torch.float64)

        batch_values = (
            torch.cat((batch_sums, has_nonfinite_output.reshape(1).to(torch.float64)))
            .cpu()
            .tolist()
        )
        if bool(batch_values[-1]):
            raise Qwen35MoeReapError("expert output saliency became non-finite")
        for expert, (weighted_sum, routed_count) in enumerate(
            zip(batch_values[:-1], route_counts, strict=True)
        ):
            self._weighted_sums[layer][expert] += float(weighted_sum)
            self._routed_counts[layer][expert] += int(routed_count)

    def register_hooks(
        self,
        experts_by_layer: Mapping[int, Any],
        *,
        valid_token_mask_provider: MaskProvider,
    ) -> None:
        """Attach exception-safe observers to Qwen3.5 per-layer expert modules."""

        if self._handles:
            raise Qwen35MoeReapError("observer hooks are already registered")
        if not callable(valid_token_mask_provider):
            raise Qwen35MoeReapError(
                "a valid_token_mask_provider is required for hook registration"
            )
        if set(experts_by_layer) != set(range(self.num_layers)):
            raise Qwen35MoeReapError("expert modules must be supplied for every configured layer")
        pending: list[Any] = []
        try:
            for layer in range(self.num_layers):
                module = experts_by_layer[layer]
                register = getattr(module, "register_forward_hook", None)
                if not callable(register):
                    raise Qwen35MoeReapError(
                        f"layer {layer} expert module does not support forward hooks"
                    )

                def hook(
                    current_module: Any,
                    args: tuple[Any, ...],
                    _output: Any,
                    *,
                    layer_index: int = layer,
                ) -> None:
                    if len(args) < 3:
                        raise Qwen35MoeReapError(
                            f"layer {layer_index} expert forward must receive hidden_states, "
                            "top_k_index, top_k_weights"
                        )
                    mask = valid_token_mask_provider(layer_index, args)
                    self.update(
                        layer_index,
                        current_module,
                        args[0],
                        args[1],
                        args[2],
                        valid_token_mask=mask,
                    )

                pending.append(register(hook))
        except Exception:
            for handle in reversed(pending):
                handle.remove()
            raise
        self._handles.extend(pending)

    def report(self) -> Qwen35MoeReapResult:
        """Return means per expert; ``None`` marks experts unseen by calibration."""

        scores = tuple(
            tuple(
                self._weighted_sums[layer][expert] / count if count else None
                for expert, count in enumerate(self._routed_counts[layer])
            )
            for layer in range(self.num_layers)
        )
        counts = tuple(tuple(layer_counts) for layer_counts in self._routed_counts)
        return Qwen35MoeReapResult(scores_by_layer=scores, routed_counts_by_layer=counts)

    def close(self) -> None:
        """Remove all observer hooks; safe to call more than once."""

        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> Qwen35MoeReapObserver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
