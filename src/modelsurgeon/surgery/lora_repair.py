"""Bounded post-surgery LoRA repair without source-checkpoint mutation."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any

from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)

LORA_REPAIR_SCHEMA_VERSION = 1


class LoRARepairError(RuntimeError):
    """Raised when bounded LoRA repair cannot preserve its mutation contract."""


class LoRAOutputMode(StrEnum):
    SEPARATE = "separate"
    MERGED = "merged"


@dataclass(frozen=True, slots=True)
class LoRARepairConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    learning_rate: float = 1e-3
    max_steps: int = 16
    seed: int = 0
    output_mode: LoRAOutputMode = LoRAOutputMode.SEPARATE

    def __post_init__(self) -> None:
        if not 1 <= self.rank <= 64 or not 1 <= self.max_steps <= 10_000 or self.seed < 0:
            raise LoRARepairError("LoRA rank/steps must be bounded and seed non-negative")
        if (
            not math.isfinite(self.alpha)
            or self.alpha <= 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise LoRARepairError("LoRA alpha, dropout, and learning rate are invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": LORA_REPAIR_SCHEMA_VERSION,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "output_mode": self.output_mode.value,
        }


@dataclass(frozen=True, slots=True)
class LoRARepairResourceUse:
    wall_seconds: float
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class LoRARepairResult:
    source_checkpoint_id: str
    candidate_checkpoint_id: str
    target_modules: tuple[str, ...]
    trainable_parameters: int
    completed_steps: int
    seed: int
    output_mode: LoRAOutputMode
    losses: tuple[float, ...]
    adapter_sha256: str
    resource_use: LoRARepairResourceUse

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": LORA_REPAIR_SCHEMA_VERSION,
            "source_checkpoint_id": self.source_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "target_modules": list(self.target_modules),
            "trainable_parameters": self.trainable_parameters,
            "completed_steps": self.completed_steps,
            "seed": self.seed,
            "output_mode": self.output_mode.value,
            "losses": list(self.losses),
            "adapter_sha256": self.adapter_sha256,
            "resource_use": self.resource_use.to_record(),
        }


def _resolve_parent(model: Any, name: str) -> tuple[Any, str, Any]:
    parts = name.split(".")
    if not parts or any(not part for part in parts):
        raise LoRARepairError("LoRA target module names must be canonical dotted paths")
    parent = model
    try:
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child = getattr(parent, parts[-1])
    except AttributeError as error:
        raise LoRARepairError(f"LoRA target module {name!r} does not exist") from error
    return parent, parts[-1], child


def _make_wrapper(torch: Any, base: Any, config: LoRARepairConfig) -> Any:
    if not isinstance(base, torch.nn.Linear):
        raise LoRARepairError("LoRA repair currently supports exact torch Linear targets")

    class _LoRALinear(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.lora_a = torch.nn.Parameter(
                torch.empty(
                    config.rank,
                    base.in_features,
                    device=base.weight.device,
                    dtype=base.weight.dtype,
                )
            )
            self.lora_b = torch.nn.Parameter(
                torch.zeros(
                    base.out_features,
                    config.rank,
                    device=base.weight.device,
                    dtype=base.weight.dtype,
                )
            )
            self.dropout = torch.nn.Dropout(config.dropout)
            self.scaling = config.alpha / config.rank
            torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self._modelsurgeon_lora = True

        def forward(self, inputs: Any) -> Any:
            return (
                self.base(inputs)
                + self.dropout(inputs) @ self.lora_a.T @ self.lora_b.T * self.scaling
            )

    return _LoRALinear()


def _adapter_digest(wrappers: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for name, wrapper in wrappers:
        digest.update(name.encode())
        for label, tensor in (("lora_a", wrapper.lora_a), ("lora_b", wrapper.lora_b)):
            host = tensor.detach().float().cpu().contiguous()
            digest.update(label.encode())
            digest.update(str(tuple(host.shape)).encode())
            digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def lora_adapter_state_dict(
    model: Any,
    target_modules: tuple[str, ...],
) -> dict[str, Any]:
    """Return tensor-only separate adapter state suitable for safetensors export."""

    state: dict[str, Any] = {}
    for name in target_modules:
        _, _, wrapper = _resolve_parent(model, name)
        if not getattr(wrapper, "_modelsurgeon_lora", False):
            raise LoRARepairError(f"target {name!r} does not retain a separate LoRA adapter")
        state[f"{name}.lora_a"] = wrapper.lora_a.detach().cpu().contiguous()
        state[f"{name}.lora_b"] = wrapper.lora_b.detach().cpu().contiguous()
    return state


def run_bounded_lora_repair(
    model: Any,
    repair_examples: tuple[dict[str, Any], ...],
    target_modules: tuple[str, ...],
    config: LoRARepairConfig,
    *,
    source_checkpoint_id: str,
    candidate_checkpoint_id: str,
) -> LoRARepairResult:
    """Train bounded adapters on a mutated candidate and merge or retain them."""

    if not source_checkpoint_id.startswith("checkpoint_") or not candidate_checkpoint_id.startswith(
        "checkpoint_"
    ):
        raise LoRARepairError("LoRA repair requires source and candidate checkpoint identities")
    if source_checkpoint_id == candidate_checkpoint_id:
        raise LoRARepairError("LoRA repair candidate must not alias its immutable source")
    if not repair_examples:
        raise LoRARepairError("LoRA repair requires a non-empty repair set")
    if target_modules != tuple(sorted(set(target_modules))) or not target_modules:
        raise LoRARepairError("LoRA targets must be non-empty, unique, and canonical")
    torch = import_module("torch")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    original_training = bool(model.training)
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    replacements: list[tuple[Any, str, Any]] = []
    wrappers: list[tuple[str, Any]] = []
    completed = False
    losses: list[float] = []
    report_box: list[MemoryTelemetryReport] = []
    started = time.perf_counter()
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in target_modules:
            parent, child_name, base = _resolve_parent(model, name)
            wrapper = _make_wrapper(torch, base, config)
            setattr(parent, child_name, wrapper)
            replacements.append((parent, child_name, base))
            wrappers.append((name, wrapper))
        adapter_parameters = tuple(
            parameter for _, wrapper in wrappers for parameter in (wrapper.lora_a, wrapper.lora_b)
        )
        trainable_parameters = sum(parameter.numel() for parameter in adapter_parameters)
        optimizer = torch.optim.AdamW(adapter_parameters, lr=config.learning_rate)
        device = next(model.parameters()).device
        cuda = TorchCudaMemoryProvider(device) if str(device).startswith("cuda") else None

        def train() -> object:
            model.train()
            for step in range(config.max_steps):
                raw = repair_examples[step % len(repair_examples)]
                if "labels" not in raw:
                    raise LoRARepairError("every repair example requires labels")
                inputs = {
                    name: value.to(device) if hasattr(value, "to") else value
                    for name, value in raw.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                loss = getattr(output, "loss", None)
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise LoRARepairError("repair forward did not return a finite loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
            return None

        collect_memory_telemetry(
            "lora_repair",
            train,
            MemoryTelemetryConfig(True, 0.01, 8192),
            cuda=cuda,
            report_callback=report_box.append,
        )
        if len(report_box) != 1:
            raise LoRARepairError("LoRA repair did not produce one resource report")
        report = report_box[0]
        adapter_sha256 = _adapter_digest(tuple(wrappers))
        if config.output_mode is LoRAOutputMode.MERGED:
            with torch.no_grad():
                for (parent, child_name, base), (_, wrapper) in zip(
                    replacements, wrappers, strict=True
                ):
                    delta = wrapper.lora_b @ wrapper.lora_a * wrapper.scaling
                    base.weight.add_(delta.to(dtype=base.weight.dtype))
                    setattr(parent, child_name, base)
        else:
            for _, wrapper in wrappers:
                wrapper.lora_a.requires_grad_(False)
                wrapper.lora_b.requires_grad_(False)
        completed = True
        return LoRARepairResult(
            source_checkpoint_id,
            candidate_checkpoint_id,
            target_modules,
            trainable_parameters,
            len(losses),
            config.seed,
            config.output_mode,
            tuple(losses),
            adapter_sha256,
            LoRARepairResourceUse(
                time.perf_counter() - started,
                report.peak_rss_bytes,
                report.peak_cuda_allocated_bytes,
                report.peak_cuda_reserved_bytes,
            ),
        )
    finally:
        if not completed:
            for parent, child_name, base in replacements:
                setattr(parent, child_name, base)
        for parameter in model.parameters():
            previous = original_requires_grad.get(id(parameter))
            if previous is not None:
                parameter.requires_grad_(previous)
        model.train(original_training)
