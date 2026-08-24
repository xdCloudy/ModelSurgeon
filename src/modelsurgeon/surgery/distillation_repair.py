"""Bounded teacher-to-candidate logit distillation repair."""

from __future__ import annotations

import hashlib
import json
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

DISTILLATION_REPAIR_SCHEMA_VERSION = 1


class DistillationRepairError(RuntimeError):
    """Raised when distillation cannot preserve its bounded repair contract."""


class TeacherLogitSource(StrEnum):
    PRECOMPUTED = "precomputed"
    TEACHER_INFERENCE = "teacher_inference"


class DistillationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_NO_IMPROVEMENT = "rejected_no_improvement"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class TokenizerSignature:
    vocabulary_size: int
    vocabulary_sha256: str
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0 or len(self.vocabulary_sha256) != 64:
            raise DistillationRepairError("tokenizer signature is invalid")
        try:
            int(self.vocabulary_sha256, 16)
        except ValueError as error:
            raise DistillationRepairError("tokenizer vocabulary digest is invalid") from error
        for token_id in (
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        ):
            if token_id is not None and not 0 <= token_id < self.vocabulary_size:
                raise DistillationRepairError("special token ID is outside the vocabulary")

    @classmethod
    def from_tokenizer(cls, tokenizer: object) -> TokenizerSignature:
        """Create a compatibility signature from the tokenizer's effective vocabulary."""

        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise DistillationRepairError("tokenizer must expose get_vocab()")
        vocabulary = get_vocab()
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise DistillationRepairError("tokenizer vocabulary must be a non-empty mapping")
        if any(
            not isinstance(token, str) or not isinstance(index, int)
            for token, index in vocabulary.items()
        ):
            raise DistillationRepairError("tokenizer vocabulary entries must be string-to-int")
        encoded = json.dumps(
            sorted(vocabulary.items()),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        return cls(
            len(vocabulary),
            hashlib.sha256(encoded).hexdigest(),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "vocabulary_sha256": self.vocabulary_sha256,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }


@dataclass(frozen=True, slots=True)
class DistillationRepairConfig:
    parameter_names: tuple[str, ...]
    temperature: float = 2.0
    distillation_weight: float = 1.0
    supervised_weight: float = 0.0
    learning_rate: float = 1e-4
    max_steps: int = 16
    min_loss_improvement: float = 0.0
    max_trainable_parameters: int = 10_000_000
    max_teacher_logit_bytes: int = 1_000_000_000
    max_wall_seconds: float = 300.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.parameter_names or self.parameter_names != tuple(
            sorted(set(self.parameter_names))
        ):
            raise DistillationRepairError("distillation parameter names must be canonical")
        values = (
            self.temperature,
            self.distillation_weight,
            self.supervised_weight,
            self.learning_rate,
            self.min_loss_improvement,
            self.max_wall_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise DistillationRepairError("distillation numeric limits must be finite")
        if (
            self.temperature <= 0
            or self.distillation_weight < 0
            or self.supervised_weight < 0
            or not math.isclose(
                self.distillation_weight + self.supervised_weight,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.learning_rate <= 0
            or not 1 <= self.max_steps <= 10_000
            or self.min_loss_improvement < 0
            or self.max_trainable_parameters <= 0
            or self.max_teacher_logit_bytes <= 0
            or self.max_wall_seconds <= 0
            or self.seed < 0
        ):
            raise DistillationRepairError("distillation limits or loss mix are invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": DISTILLATION_REPAIR_SCHEMA_VERSION,
            "parameter_names": list(self.parameter_names),
            "temperature": self.temperature,
            "distillation_weight": self.distillation_weight,
            "supervised_weight": self.supervised_weight,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "min_loss_improvement": self.min_loss_improvement,
            "max_trainable_parameters": self.max_trainable_parameters,
            "max_teacher_logit_bytes": self.max_teacher_logit_bytes,
            "max_wall_seconds": self.max_wall_seconds,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class DistillationResourceUse:
    wall_seconds: float
    teacher_capture_seconds: float
    training_seconds: float
    teacher_logit_bytes: int
    token_rows: int
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "teacher_capture_seconds": self.teacher_capture_seconds,
            "training_seconds": self.training_seconds,
            "teacher_logit_bytes": self.teacher_logit_bytes,
            "token_rows": self.token_rows,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class DistillationRepairResult:
    status: DistillationStatus
    source_checkpoint_id: str
    candidate_parent_checkpoint_id: str
    output_checkpoint_id: str | None
    teacher_source: TeacherLogitSource
    teacher_logits_sha256: str
    tokenizer_signature: TokenizerSignature
    trainable_parameters: int
    completed_steps: int
    baseline_loss: float
    repaired_loss: float | None
    training_losses: tuple[float, ...]
    evaluation_losses: tuple[float, ...]
    config: DistillationRepairConfig
    resource_use: DistillationResourceUse

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": DISTILLATION_REPAIR_SCHEMA_VERSION,
            "status": self.status.value,
            "source_checkpoint_id": self.source_checkpoint_id,
            "candidate_parent_checkpoint_id": self.candidate_parent_checkpoint_id,
            "output_checkpoint_id": self.output_checkpoint_id,
            "teacher_source": self.teacher_source.value,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "tokenizer_signature": self.tokenizer_signature.to_record(),
            "trainable_parameters": self.trainable_parameters,
            "completed_steps": self.completed_steps,
            "baseline_loss": self.baseline_loss,
            "repaired_loss": self.repaired_loss,
            "training_losses": list(self.training_losses),
            "evaluation_losses": list(self.evaluation_losses),
            "config": self.config.to_record(),
            "resource_use": self.resource_use.to_record(),
        }


@dataclass(frozen=True, slots=True)
class _RunState:
    status: DistillationStatus
    repaired_loss: float | None
    output_checkpoint_id: str | None
    training_losses: tuple[float, ...]
    evaluation_losses: tuple[float, ...]


def _move(example: dict[str, Any], device: Any, *, include_labels: bool = True) -> dict[str, Any]:
    return {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in example.items()
        if include_labels or name != "labels"
    }


def _logit_digest(logits: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in logits:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_digest(parameters: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        host = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(host.shape)).encode())
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def _capture_teacher_logits(
    torch: Any,
    examples: tuple[dict[str, Any], ...],
    config: DistillationRepairConfig,
    signature: TokenizerSignature,
    *,
    teacher_model: Any | None,
    baseline_logits: tuple[Any, ...] | None,
) -> tuple[TeacherLogitSource, tuple[Any, ...], int]:
    if (teacher_model is None) == (baseline_logits is None):
        raise DistillationRepairError("provide exactly one teacher model or baseline-logit set")
    captured: list[Any] = []
    if teacher_model is not None:
        original_training = bool(teacher_model.training)
        teacher_device = next(teacher_model.parameters()).device
        try:
            teacher_model.eval()
            with torch.inference_mode():
                for example in examples:
                    output = teacher_model(**_move(example, teacher_device, include_labels=False))
                    logits = getattr(output, "logits", None)
                    if logits is None:
                        raise DistillationRepairError("teacher inference did not return logits")
                    captured.append(logits.detach().float().cpu().contiguous().clone())
        finally:
            teacher_model.train(original_training)
        source = TeacherLogitSource.TEACHER_INFERENCE
    else:
        if baseline_logits is None or len(baseline_logits) != len(examples):
            raise DistillationRepairError("baseline logits must align one-to-one with examples")
        for logits in baseline_logits:
            if not hasattr(logits, "detach"):
                raise DistillationRepairError("baseline logits must be tensors")
            captured.append(logits.detach().float().cpu().contiguous().clone())
        source = TeacherLogitSource.PRECOMPUTED
    total_bytes = 0
    for example, logits in zip(examples, captured, strict=True):
        input_ids = example.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise DistillationRepairError("distillation examples require tensor input_ids")
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
            raise DistillationRepairError("teacher logits must align to input batch and sequence")
        if int(logits.shape[-1]) != signature.vocabulary_size:
            raise DistillationRepairError("teacher logits do not match tokenizer vocabulary")
        if not bool(torch.isfinite(logits).all().item()):
            raise DistillationRepairError("teacher logits must be finite")
        total_bytes += int(logits.numel() * logits.element_size())
        if total_bytes > config.max_teacher_logit_bytes:
            raise DistillationRepairError("teacher-logit byte budget exceeded")
    return source, tuple(captured), total_bytes


def _objective(
    torch: Any,
    model: Any,
    example: dict[str, Any],
    teacher_logits: Any,
    config: DistillationRepairConfig,
    device: Any,
) -> Any:
    if config.supervised_weight > 0 and "labels" not in example:
        raise DistillationRepairError("supervised distillation mix requires labels")
    output = model(**_move(example, device))
    candidate_logits = getattr(output, "logits", None)
    if candidate_logits is None or tuple(candidate_logits.shape) != tuple(teacher_logits.shape):
        raise DistillationRepairError("candidate logits do not align with immutable teacher logits")
    teacher = teacher_logits.to(device=device, dtype=torch.float32)
    candidate = candidate_logits.float()
    attention_mask = example.get("attention_mask")
    if attention_mask is None:
        valid = torch.ones(candidate.shape[:-1], dtype=torch.bool, device=device)
    else:
        valid = attention_mask.to(device=device, dtype=torch.bool)
        if tuple(valid.shape) != tuple(candidate.shape[:-1]):
            raise DistillationRepairError("attention mask does not align with logits")
    if not bool(valid.any().item()):
        raise DistillationRepairError("distillation requires at least one unmasked token row")
    student_rows = candidate[valid]
    teacher_rows = teacher[valid]
    distillation = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_rows / config.temperature, dim=-1),
        torch.nn.functional.softmax(teacher_rows / config.temperature, dim=-1),
        reduction="batchmean",
    ) * (config.temperature**2)
    supervised = getattr(output, "loss", None)
    if config.supervised_weight > 0 and supervised is None:
        raise DistillationRepairError("candidate did not return supervised loss")
    total = config.distillation_weight * distillation
    if supervised is not None:
        total = total + config.supervised_weight * supervised
    if not bool(torch.isfinite(total).item()):
        raise DistillationRepairError("distillation objective is not finite")
    return total


def _evaluate(
    torch: Any,
    model: Any,
    examples: tuple[dict[str, Any], ...],
    teacher_logits: tuple[Any, ...],
    config: DistillationRepairConfig,
    device: Any,
) -> float:
    original_training = bool(model.training)
    values: list[float] = []
    try:
        model.eval()
        with torch.inference_mode():
            for example, logits in zip(examples, teacher_logits, strict=True):
                value = _objective(torch, model, example, logits, config, device)
                values.append(float(value.detach().item()))
    finally:
        model.train(original_training)
    return math.fsum(values) / len(values)


def run_distillation_repair(
    candidate_model: Any,
    examples: tuple[dict[str, Any], ...],
    config: DistillationRepairConfig,
    *,
    teacher_tokenizer: TokenizerSignature,
    candidate_tokenizer: TokenizerSignature,
    source_checkpoint_id: str,
    candidate_parent_checkpoint_id: str,
    teacher_model: Any | None = None,
    baseline_logits: tuple[Any, ...] | None = None,
) -> DistillationRepairResult:
    """Repair selected candidate parameters toward immutable teacher logits."""

    if teacher_tokenizer != candidate_tokenizer:
        raise DistillationRepairError("teacher and candidate tokenizers are incompatible")
    if not source_checkpoint_id.startswith(
        "checkpoint_"
    ) or not candidate_parent_checkpoint_id.startswith("checkpoint_"):
        raise DistillationRepairError("distillation requires source and parent checkpoint IDs")
    if source_checkpoint_id == candidate_parent_checkpoint_id:
        raise DistillationRepairError("candidate parent must not alias the immutable source")
    if not examples:
        raise DistillationRepairError("distillation requires a non-empty example set")
    if teacher_model is candidate_model:
        raise DistillationRepairError("teacher and candidate models must be distinct")
    torch = import_module("torch")
    named = tuple(candidate_model.named_parameters())
    by_name = {name: parameter for name, parameter in named}
    missing = tuple(name for name in config.parameter_names if name not in by_name)
    if missing:
        raise DistillationRepairError("unknown selected parameters: " + ", ".join(missing))
    selected = tuple((name, by_name[name]) for name in config.parameter_names)
    trainable_parameters = sum(parameter.numel() for _, parameter in selected)
    if trainable_parameters > config.max_trainable_parameters:
        raise DistillationRepairError("trainable parameter budget exceeded")

    started = time.perf_counter()
    teacher_started = time.perf_counter()
    teacher_source, immutable_logits, teacher_logit_bytes = _capture_teacher_logits(
        torch,
        examples,
        config,
        teacher_tokenizer,
        teacher_model=teacher_model,
        baseline_logits=baseline_logits,
    )
    teacher_capture_seconds = time.perf_counter() - teacher_started
    teacher_logits_sha256 = _logit_digest(immutable_logits)
    device = next(candidate_model.parameters()).device
    baseline_loss = _evaluate(torch, candidate_model, examples, immutable_logits, config, device)
    token_rows = sum(
        int(example["attention_mask"].sum().item())
        if "attention_mask" in example
        else int(example["input_ids"].numel())
        for example in examples
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    original_training = bool(candidate_model.training)
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for _, parameter in named
    }
    snapshots: dict[str, Any] = {}
    result_box: list[_RunState] = []
    report_box: list[MemoryTelemetryReport] = []
    completed = False
    training_started = time.perf_counter()
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
        training_losses: list[float] = []
        evaluation_losses: list[float] = []
        best_loss = math.inf
        best_weights: dict[str, Any] | None = None
        budget_exhausted = False
        for step in range(config.max_steps):
            if time.perf_counter() - started >= config.max_wall_seconds:
                budget_exhausted = True
                break
            candidate_model.train()
            optimizer.zero_grad(set_to_none=True)
            index = step % len(examples)
            loss = _objective(
                torch,
                candidate_model,
                examples[index],
                immutable_logits[index],
                config,
                device,
            )
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().item()))
            evaluated = _evaluate(
                torch, candidate_model, examples, immutable_logits, config, device
            )
            evaluation_losses.append(evaluated)
            if evaluated < best_loss:
                best_loss = evaluated
                best_weights = {name: parameter.detach().clone() for name, parameter in selected}
            if time.perf_counter() - started >= config.max_wall_seconds:
                budget_exhausted = True
                break
        if budget_exhausted:
            status = DistillationStatus.BUDGET_EXHAUSTED
            repaired_loss: float | None = None
            restore = snapshots
        elif best_weights is None or best_loss >= baseline_loss - config.min_loss_improvement:
            status = DistillationStatus.REJECTED_NO_IMPROVEMENT
            repaired_loss = None if best_weights is None else best_loss
            restore = snapshots
        else:
            status = DistillationStatus.ACCEPTED
            repaired_loss = best_loss
            restore = best_weights
        with torch.no_grad():
            for name, parameter in selected:
                parameter.copy_(restore[name])
        output_checkpoint_id = None
        if status is DistillationStatus.ACCEPTED:
            canonical_identity_json = import_module(
                "modelsurgeon.experiments.identity"
            ).canonical_identity_json
            payload = canonical_identity_json(
                {
                    "source_checkpoint_id": source_checkpoint_id,
                    "parent_checkpoint_id": candidate_parent_checkpoint_id,
                    "teacher_logits_sha256": teacher_logits_sha256,
                    "config": config.to_record(),
                    "parameter_sha256": _parameter_digest(selected),
                }
            ).encode()
            output_checkpoint_id = f"checkpoint_{hashlib.sha256(payload).hexdigest()}"
        result_box.append(
            _RunState(
                status,
                repaired_loss,
                output_checkpoint_id,
                tuple(training_losses),
                tuple(evaluation_losses),
            )
        )
        return None

    try:
        collect_memory_telemetry(
            "distillation_repair",
            operation,
            MemoryTelemetryConfig(True, 0.01, 8192),
            cuda=cuda,
            report_callback=report_box.append,
        )
        training_seconds = time.perf_counter() - training_started
        if len(result_box) != 1 or len(report_box) != 1:
            raise DistillationRepairError("distillation did not produce one result/report")
        state = result_box[0]
        report = report_box[0]
        result = DistillationRepairResult(
            state.status,
            source_checkpoint_id,
            candidate_parent_checkpoint_id,
            state.output_checkpoint_id,
            teacher_source,
            teacher_logits_sha256,
            candidate_tokenizer,
            trainable_parameters,
            len(state.training_losses),
            baseline_loss,
            state.repaired_loss,
            state.training_losses,
            state.evaluation_losses,
            config,
            DistillationResourceUse(
                time.perf_counter() - started,
                teacher_capture_seconds,
                training_seconds,
                teacher_logit_bytes,
                token_rows,
                report.peak_rss_bytes,
                report.peak_cuda_allocated_bytes,
                report.peak_cuda_reserved_bytes,
            ),
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
        candidate_model.train(original_training)
