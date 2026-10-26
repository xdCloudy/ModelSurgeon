"""Bounded post-surgery LoRA repair without source-checkpoint mutation."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any, cast

from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)
from modelsurgeon.surgery.module_training import (
    restore_module_training_modes,
    snapshot_module_training_modes,
)

LORA_REPAIR_SCHEMA_VERSION = 5
VALIDATION_LOSS_AGGREGATION = "token_weighted_causal_targets"


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
    max_trainable_parameters: int = 10_000_000
    seed: int = 0
    output_mode: LoRAOutputMode = LoRAOutputMode.SEPARATE

    def __post_init__(self) -> None:
        if (
            not 1 <= self.rank <= 64
            or not 1 <= self.max_steps <= 10_000
            or self.max_trainable_parameters <= 0
            or self.seed < 0
        ):
            raise LoRARepairError(
                "LoRA rank, steps, and trainable-parameter budget must be bounded"
            )
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
            "max_trainable_parameters": self.max_trainable_parameters,
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
    max_trainable_parameters: int
    completed_steps: int
    seed: int
    output_mode: LoRAOutputMode
    losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    selected_step: int
    baseline_validation_loss: float | None
    selected_validation_loss: float | None
    validation_improved: bool | None
    validation_loss_aggregation: str | None
    training_examples_sha256: str
    validation_examples_sha256: str | None
    validation_dataset_sha256: str | None
    evaluation_source: str
    adapter_sha256: str
    resource_use: LoRARepairResourceUse

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": LORA_REPAIR_SCHEMA_VERSION,
            "source_checkpoint_id": self.source_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "target_modules": list(self.target_modules),
            "trainable_parameters": self.trainable_parameters,
            "max_trainable_parameters": self.max_trainable_parameters,
            "completed_steps": self.completed_steps,
            "seed": self.seed,
            "output_mode": self.output_mode.value,
            "losses": list(self.losses),
            "validation_losses": list(self.validation_losses),
            "selected_step": self.selected_step,
            "baseline_validation_loss": self.baseline_validation_loss,
            "selected_validation_loss": self.selected_validation_loss,
            "validation_improved": self.validation_improved,
            "validation_loss_aggregation": self.validation_loss_aggregation,
            "training_examples_sha256": self.training_examples_sha256,
            "validation_examples_sha256": self.validation_examples_sha256,
            "validation_dataset_sha256": self.validation_dataset_sha256,
            "evaluation_source": self.evaluation_source,
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


def _example_input_ids(example: dict[str, Any]) -> bytes:
    input_ids = example.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "detach"):
        raise LoRARepairError("repair examples require tensor input_ids")
    return cast(bytes, input_ids.detach().cpu().contiguous().numpy().tobytes())


def _examples_sha256(examples: tuple[dict[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        encoded = _example_input_ids(example)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _adapter_matrices(wrapper: Any) -> tuple[Any, Any]:
    try:
        adapter_a = wrapper.lora_A["default"].weight
        adapter_b = wrapper.lora_B["default"].weight
    except (AttributeError, KeyError) as error:
        raise LoRARepairError("target does not retain the default PEFT LoRA adapter") from error
    return adapter_a, adapter_b


def _adapter_digest(wrappers: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for name, wrapper in wrappers:
        digest.update(name.encode())
        adapter_a, adapter_b = _adapter_matrices(wrapper)
        for label, tensor in (("lora_a", adapter_a), ("lora_b", adapter_b)):
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
        try:
            adapter_a, adapter_b = _adapter_matrices(wrapper)
        except LoRARepairError as error:
            raise LoRARepairError(
                f"target {name!r} does not retain a separate LoRA adapter"
            ) from error
        state[f"{name}.lora_a"] = adapter_a.detach().cpu().contiguous()
        state[f"{name}.lora_b"] = adapter_b.detach().cpu().contiguous()
    return state


def run_bounded_lora_repair(
    model: Any,
    repair_examples: tuple[dict[str, Any], ...],
    target_modules: tuple[str, ...],
    config: LoRARepairConfig,
    *,
    source_checkpoint_id: str,
    candidate_checkpoint_id: str,
    validation_examples: tuple[dict[str, Any], ...] | None = None,
    validation_dataset_sha256: str | None = None,
    evaluation_source: str | None = None,
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
    if validation_examples is not None and not validation_examples:
        raise LoRARepairError("validation examples must be non-empty when supplied")
    if validation_examples is None and validation_dataset_sha256 is not None:
        raise LoRARepairError("validation dataset digest requires validation examples")
    if validation_dataset_sha256 is not None:
        if len(validation_dataset_sha256) != 64:
            raise LoRARepairError("validation dataset digest must be SHA-256")
        try:
            int(validation_dataset_sha256, 16)
        except ValueError as error:
            raise LoRARepairError("validation dataset digest must be SHA-256") from error
        validation_dataset_sha256 = validation_dataset_sha256.lower()
    selected_evaluation_source = (
        ("training_fallback" if validation_examples is None else "provided_validation")
        if evaluation_source is None
        else evaluation_source
    )
    if not selected_evaluation_source.strip():
        raise LoRARepairError("evaluation source cannot be blank")
    training_examples_sha256 = _examples_sha256(repair_examples)
    validation_examples_sha256 = (
        None if validation_examples is None else _examples_sha256(validation_examples)
    )
    if validation_examples is not None:
        training_keys = {_example_input_ids(example) for example in repair_examples}
        if any(_example_input_ids(example) in training_keys for example in validation_examples):
            raise LoRARepairError("training and validation examples must be disjoint")
    if target_modules != tuple(sorted(set(target_modules))) or not target_modules:
        raise LoRARepairError("LoRA targets must be non-empty, unique, and canonical")
    torch = import_module("torch")
    try:
        peft = import_module("peft")
    except ImportError as error:
        raise LoRARepairError("install the modelsurgeon HF extra with PEFT support") from error
    original_training_modes = snapshot_module_training_modes(model)
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    expected_trainable_parameters = 0
    for name in target_modules:
        _, _, base = _resolve_parent(model, name)
        if not isinstance(base, torch.nn.Linear):
            raise LoRARepairError("LoRA repair currently supports exact torch Linear targets")
        expected_trainable_parameters += config.rank * (base.in_features + base.out_features)
    if expected_trainable_parameters > config.max_trainable_parameters:
        raise LoRARepairError("LoRA trainable-parameter budget exceeded")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    replacements: list[tuple[Any, str, Any]] = []
    wrappers: list[tuple[str, Any]] = []
    completed = False
    losses: list[float] = []
    validation_losses: list[float] = []
    selected_step = 0
    baseline_validation_loss: float | None = None
    selected_validation_loss: float | None = None
    validation_improved: bool | None = None
    selected_adapter_state: tuple[Any, ...] | None = None
    best_validation_loss = math.inf
    report_box: list[MemoryTelemetryReport] = []
    started = time.perf_counter()
    try:
        for name in target_modules:
            parent, child_name, base = _resolve_parent(model, name)
            replacements.append((parent, child_name, base))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        peft_config = peft.LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            target_modules=list(target_modules),
            lora_dropout=config.dropout,
            bias="none",
            task_type=None,
        )
        peft_model = peft.get_peft_model(model, peft_config)
        wrappers = [(name, _resolve_parent(model, name)[2]) for name in target_modules]
        adapter_parameters = tuple(
            parameter for parameter in peft_model.parameters() if parameter.requires_grad
        )
        trainable_parameters = sum(parameter.numel() for parameter in adapter_parameters)
        if trainable_parameters != expected_trainable_parameters:
            raise LoRARepairError("PEFT adapter parameter count does not match selected targets")
        optimizer = torch.optim.AdamW(adapter_parameters, lr=config.learning_rate)
        device = next(peft_model.parameters()).device
        cuda = TorchCudaMemoryProvider(device) if str(device).startswith("cuda") else None

        def evaluate_validation() -> float:
            if validation_examples is None:
                raise LoRARepairError("validation examples are required for evaluation")
            peft_model.eval()
            weighted_loss = 0.0
            valid_token_count = 0
            with torch.inference_mode():
                for validation_example in validation_examples:
                    if "labels" not in validation_example:
                        raise LoRARepairError("every validation example requires labels")
                    validation_inputs = {
                        name: value.to(device) if hasattr(value, "to") else value
                        for name, value in validation_example.items()
                    }
                    validation_output = peft_model(**validation_inputs)
                    validation_loss = getattr(validation_output, "loss", None)
                    if validation_loss is None or not bool(torch.isfinite(validation_loss).item()):
                        raise LoRARepairError("validation forward did not return a finite loss")
                    labels = validation_inputs["labels"]
                    if labels.ndim == 0 or labels.shape[-1] < 2:
                        raise LoRARepairError(
                            "validation examples require at least two causal label positions"
                        )
                    example_token_count = int((labels[..., 1:] != -100).sum().item())
                    if example_token_count <= 0:
                        raise LoRARepairError(
                            "validation examples require at least one valid prediction target"
                        )
                    weighted_loss += float(validation_loss.detach().float().item()) * (
                        example_token_count
                    )
                    valid_token_count += example_token_count
            if valid_token_count <= 0:
                raise LoRARepairError("validation set has no valid prediction targets")
            return weighted_loss / valid_token_count

        def train() -> object:
            nonlocal best_validation_loss, selected_step, selected_adapter_state
            nonlocal baseline_validation_loss
            if validation_examples is not None:
                baseline_validation_loss = evaluate_validation()
                best_validation_loss = baseline_validation_loss
                selected_step = 0
                selected_adapter_state = tuple(
                    parameter.detach().clone() for parameter in adapter_parameters
                )
            for step in range(config.max_steps):
                model.train()
                raw = repair_examples[step % len(repair_examples)]
                if "labels" not in raw:
                    raise LoRARepairError("every repair example requires labels")
                inputs = {
                    name: value.to(device) if hasattr(value, "to") else value
                    for name, value in raw.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = peft_model(**inputs)
                loss = getattr(output, "loss", None)
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise LoRARepairError("repair forward did not return a finite loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
                if validation_examples is not None:
                    mean_validation_loss = evaluate_validation()
                    validation_losses.append(mean_validation_loss)
                    if mean_validation_loss < best_validation_loss:
                        best_validation_loss = mean_validation_loss
                        selected_step = step + 1
                        selected_adapter_state = tuple(
                            parameter.detach().clone() for parameter in adapter_parameters
                        )
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
        if validation_examples is not None:
            if selected_adapter_state is None or baseline_validation_loss is None:
                raise LoRARepairError("validation did not retain a candidate checkpoint")
            with torch.no_grad():
                for parameter, selected_parameter in zip(
                    adapter_parameters, selected_adapter_state, strict=True
                ):
                    parameter.copy_(selected_parameter)
            selected_validation_loss = best_validation_loss
            validation_improved = best_validation_loss < baseline_validation_loss
        else:
            selected_step = len(losses)
        adapter_sha256 = _adapter_digest(tuple(wrappers))
        if config.output_mode is LoRAOutputMode.MERGED:
            peft_model.merge_and_unload(safe_merge=True)
        else:
            for _, wrapper in wrappers:
                adapter_a, adapter_b = _adapter_matrices(wrapper)
                adapter_a.requires_grad_(False)
                adapter_b.requires_grad_(False)
        completed = True
        return LoRARepairResult(
            source_checkpoint_id=source_checkpoint_id,
            candidate_checkpoint_id=candidate_checkpoint_id,
            target_modules=target_modules,
            trainable_parameters=trainable_parameters,
            max_trainable_parameters=config.max_trainable_parameters,
            completed_steps=len(losses),
            seed=config.seed,
            output_mode=config.output_mode,
            losses=tuple(losses),
            validation_losses=tuple(validation_losses),
            selected_step=selected_step,
            baseline_validation_loss=baseline_validation_loss,
            selected_validation_loss=selected_validation_loss,
            validation_improved=validation_improved,
            validation_loss_aggregation=(
                VALIDATION_LOSS_AGGREGATION if validation_examples is not None else None
            ),
            training_examples_sha256=training_examples_sha256,
            validation_examples_sha256=validation_examples_sha256,
            validation_dataset_sha256=validation_dataset_sha256,
            evaluation_source=selected_evaluation_source,
            adapter_sha256=adapter_sha256,
            resource_use=LoRARepairResourceUse(
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
        restore_module_training_modes(
            model,
            original_training_modes,
            subtree_roots=target_modules,
        )
