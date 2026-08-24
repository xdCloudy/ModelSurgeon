"""Transactional bounded short fine-tuning repair with overfit rejection."""

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

SHORT_FINETUNE_SCHEMA_VERSION = 1


class ShortFineTuneError(RuntimeError):
    """Raised when fine-tuning repair cannot preserve its bounded contract."""


class FineTuneParameterMode(StrEnum):
    FULL = "full"
    SELECTED = "selected"


class FineTuneStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_OVERFIT = "rejected_overfit"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class ShortFineTuneConfig:
    parameter_mode: FineTuneParameterMode = FineTuneParameterMode.SELECTED
    parameter_names: tuple[str, ...] = ()
    learning_rate: float = 1e-4
    max_steps: int = 16
    validation_patience: int = 3
    min_validation_improvement: float = 0.0
    max_validation_loss_increase: float = 0.0
    max_trainable_parameters: int = 10_000_000
    max_wall_seconds: float = 300.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.parameter_mode is FineTuneParameterMode.SELECTED:
            if not self.parameter_names or self.parameter_names != tuple(
                sorted(set(self.parameter_names))
            ):
                raise ShortFineTuneError("selected fine-tuning names must be canonical")
        elif self.parameter_names:
            raise ShortFineTuneError("full fine-tuning cannot also name selected parameters")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not 1 <= self.max_steps <= 10_000
            or self.validation_patience <= 0
            or not math.isfinite(self.min_validation_improvement)
            or self.min_validation_improvement < 0
            or not math.isfinite(self.max_validation_loss_increase)
            or self.max_validation_loss_increase < 0
            or self.max_trainable_parameters <= 0
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
            or self.seed < 0
        ):
            raise ShortFineTuneError("short fine-tuning limits are invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SHORT_FINETUNE_SCHEMA_VERSION,
            "parameter_mode": self.parameter_mode.value,
            "parameter_names": list(self.parameter_names),
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "validation_patience": self.validation_patience,
            "min_validation_improvement": self.min_validation_improvement,
            "max_validation_loss_increase": self.max_validation_loss_increase,
            "max_trainable_parameters": self.max_trainable_parameters,
            "max_wall_seconds": self.max_wall_seconds,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ShortFineTuneResult:
    status: FineTuneStatus
    source_checkpoint_id: str
    candidate_parent_checkpoint_id: str
    output_checkpoint_id: str | None
    trainable_parameters: int
    completed_steps: int
    seed: int
    no_repair_validation_loss: float
    repaired_validation_loss: float | None
    validation_loss_delta: float | None
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    early_stopped: bool
    weights_restored: bool
    wall_seconds: float
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SHORT_FINETUNE_SCHEMA_VERSION,
            "status": self.status.value,
            "source_checkpoint_id": self.source_checkpoint_id,
            "candidate_parent_checkpoint_id": self.candidate_parent_checkpoint_id,
            "output_checkpoint_id": self.output_checkpoint_id,
            "trainable_parameters": self.trainable_parameters,
            "completed_steps": self.completed_steps,
            "seed": self.seed,
            "no_repair_validation_loss": self.no_repair_validation_loss,
            "repaired_validation_loss": self.repaired_validation_loss,
            "validation_loss_delta": self.validation_loss_delta,
            "training_losses": list(self.training_losses),
            "validation_losses": list(self.validation_losses),
            "early_stopped": self.early_stopped,
            "weights_restored": self.weights_restored,
            "resource_use": {
                "wall_seconds": self.wall_seconds,
                "peak_rss_bytes": self.peak_rss_bytes,
                "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
                "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            },
        }


@dataclass(frozen=True, slots=True)
class _FineTuneRunState:
    status: FineTuneStatus
    baseline: float
    repaired: float | None
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    early_stopped: bool
    output_checkpoint_id: str | None


def _finite_loss(torch: Any, output: object, label: str) -> Any:
    loss = getattr(output, "loss", None)
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise ShortFineTuneError(f"{label} did not return a finite loss")
    return loss


def _move(example: dict[str, Any], device: Any) -> dict[str, Any]:
    if "labels" not in example:
        raise ShortFineTuneError("fine-tuning examples require labels")
    return {
        name: value.to(device) if hasattr(value, "to") else value for name, value in example.items()
    }


def _validation_loss(
    torch: Any,
    model: Any,
    examples: tuple[dict[str, Any], ...],
    device: Any,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for example in examples:
            loss = _finite_loss(torch, model(**_move(example, device)), "validation forward")
            losses.append(float(loss.detach().item()))
    return math.fsum(losses) / len(losses)


def _parameter_digest(parameters: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        host = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(host.shape)).encode())
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def run_short_finetune_repair(
    model: Any,
    training_examples: tuple[dict[str, Any], ...],
    validation_examples: tuple[dict[str, Any], ...],
    config: ShortFineTuneConfig,
    *,
    source_checkpoint_id: str,
    candidate_parent_checkpoint_id: str,
) -> ShortFineTuneResult:
    """Fine-tune a candidate briefly and keep only validation-safe best weights."""

    if not source_checkpoint_id.startswith(
        "checkpoint_"
    ) or not candidate_parent_checkpoint_id.startswith("checkpoint_"):
        raise ShortFineTuneError("fine-tuning repair requires source and parent checkpoint IDs")
    if source_checkpoint_id == candidate_parent_checkpoint_id:
        raise ShortFineTuneError("fine-tuning candidate must not alias the immutable source")
    if not training_examples or not validation_examples:
        raise ShortFineTuneError("fine-tuning requires non-empty train and validation sets")
    torch = import_module("torch")
    named = tuple(model.named_parameters())
    by_name = {name: parameter for name, parameter in named}
    if config.parameter_mode is FineTuneParameterMode.FULL:
        selected = named
    else:
        missing = tuple(name for name in config.parameter_names if name not in by_name)
        if missing:
            raise ShortFineTuneError("unknown selected parameters: " + ", ".join(missing))
        selected = tuple((name, by_name[name]) for name in config.parameter_names)
    trainable_parameters = sum(parameter.numel() for _, parameter in selected)
    if trainable_parameters > config.max_trainable_parameters:
        raise ShortFineTuneError(
            f"trainable parameter budget exceeded: {trainable_parameters} > "
            f"{config.max_trainable_parameters}"
        )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    original_training = bool(model.training)
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for _, parameter in named
    }
    snapshots: dict[str, Any] = {}
    result_box: list[_FineTuneRunState] = []
    report_box: list[MemoryTelemetryReport] = []
    started = time.perf_counter()
    completed = False
    device = next(model.parameters()).device
    cuda = TorchCudaMemoryProvider(device) if str(device).startswith("cuda") else None

    def operation() -> object:
        snapshots.update({name: parameter.detach().clone() for name, parameter in selected})
        for _, parameter in named:
            parameter.requires_grad_(False)
        for _, parameter in selected:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            tuple(parameter for _, parameter in selected), lr=config.learning_rate
        )
        baseline = _validation_loss(torch, model, validation_examples, device)
        training_losses: list[float] = []
        validation_losses: list[float] = []
        best_loss = math.inf
        best_weights: dict[str, Any] | None = None
        stale = 0
        early_stopped = False
        budget_exhausted = False
        for step in range(config.max_steps):
            if time.perf_counter() - started >= config.max_wall_seconds:
                budget_exhausted = True
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            example = training_examples[step % len(training_examples)]
            loss = _finite_loss(torch, model(**_move(example, device)), "training forward")
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().item()))
            validation = _validation_loss(torch, model, validation_examples, device)
            validation_losses.append(validation)
            if validation < best_loss - config.min_validation_improvement:
                best_loss = validation
                best_weights = {name: parameter.detach().clone() for name, parameter in selected}
                stale = 0
            else:
                stale += 1
                if stale >= config.validation_patience:
                    early_stopped = True
                    break
            if time.perf_counter() - started >= config.max_wall_seconds:
                budget_exhausted = True
                break

        if budget_exhausted:
            status = FineTuneStatus.BUDGET_EXHAUSTED
            repaired: float | None = None
            restore = snapshots
        elif best_weights is None or best_loss > baseline + config.max_validation_loss_increase:
            status = FineTuneStatus.REJECTED_OVERFIT
            repaired = None if best_weights is None else best_loss
            restore = snapshots
        else:
            status = FineTuneStatus.ACCEPTED
            repaired = best_loss
            restore = best_weights
        with torch.no_grad():
            for name, parameter in selected:
                parameter.copy_(restore[name])
        output_checkpoint_id = None
        if status is FineTuneStatus.ACCEPTED:
            canonical_identity_json = import_module(
                "modelsurgeon.experiments.identity"
            ).canonical_identity_json
            payload = canonical_identity_json(
                {
                    "source_checkpoint_id": source_checkpoint_id,
                    "parent_checkpoint_id": candidate_parent_checkpoint_id,
                    "config": config.to_record(),
                    "parameter_sha256": _parameter_digest(selected),
                }
            ).encode()
            output_checkpoint_id = f"checkpoint_{hashlib.sha256(payload).hexdigest()}"
        result_box.append(
            _FineTuneRunState(
                status,
                baseline,
                repaired,
                tuple(training_losses),
                tuple(validation_losses),
                early_stopped,
                output_checkpoint_id,
            )
        )
        return None

    try:
        collect_memory_telemetry(
            "short_finetune_repair",
            operation,
            MemoryTelemetryConfig(True, 0.01, 8192),
            cuda=cuda,
            report_callback=report_box.append,
        )
        if len(result_box) != 1 or len(report_box) != 1:
            raise ShortFineTuneError("fine-tuning repair did not produce one result/report")
        state = result_box[0]
        report = report_box[0]
        result = ShortFineTuneResult(
            state.status,
            source_checkpoint_id,
            candidate_parent_checkpoint_id,
            state.output_checkpoint_id,
            trainable_parameters,
            len(state.training_losses),
            config.seed,
            state.baseline,
            state.repaired,
            None if state.repaired is None else state.repaired - state.baseline,
            state.training_losses,
            state.validation_losses,
            state.early_stopped,
            state.status is not FineTuneStatus.ACCEPTED,
            time.perf_counter() - started,
            report.peak_rss_bytes,
            report.peak_cuda_allocated_bytes,
            report.peak_cuda_reserved_bytes,
        )
        completed = True
        return result
    finally:
        if not completed and snapshots:
            with torch.no_grad():
                for name, parameter in selected:
                    parameter.copy_(snapshots[name])
        for _, parameter in named:
            parameter.requires_grad_(original_requires_grad[id(parameter)])
        model.train(original_training)
