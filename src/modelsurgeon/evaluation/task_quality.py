"""Pinned, user-supplied task-quality evaluation for first-party runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, cast

from modelsurgeon.task_quality_limits import (
    MAX_TASK_QUALITY_DATASET_BYTES,
    MAX_TASK_QUALITY_EXAMPLES,
    MAX_TASK_QUALITY_ROW_BYTES,
    MAX_TASK_QUALITY_ROWS,
)

TASK_QUALITY_EVALUATOR_VERSION = "code_exact_match_v2"
MAX_TASK_QUALITY_NEW_TOKENS = 4096
MAX_TASK_QUALITY_TOTAL_NEW_TOKENS = 1_000_000
MULTIPLE_CHOICE_EVALUATOR_VERSION = "multiple_choice_loglikelihood_v1"
MAX_MULTIPLE_CHOICE_DATASET_BYTES = 16 * 1024 * 1024
MAX_MULTIPLE_CHOICE_OPTIONS = 16
MAX_MULTIPLE_CHOICE_TEXT_CHARS = 16_384
MAX_MULTIPLE_CHOICE_INPUT_TOKENS = 8192
MAX_MULTIPLE_CHOICE_FORWARD_EVALUATIONS = 16_384
MAX_MULTIPLE_CHOICE_TOTAL_SCORED_TOKENS = 1_000_000


class TaskQualityEvaluationError(ValueError):
    """Raised when task-quality evidence cannot be measured safely."""


def _preserve_model_training_modes[**P, Measurement](
    function: Callable[Concatenate[Any, P], Measurement],
) -> Callable[Concatenate[Any, P], Measurement]:
    """Restore every module's mode after evaluation, including when it fails."""

    @wraps(function)
    def wrapped(model: Any, *args: P.args, **kwargs: P.kwargs) -> Measurement:
        original_modes = tuple((module, bool(module.training)) for module in model.modules())
        try:
            return function(model, *args, **kwargs)
        finally:
            for module, was_training in original_modes:
                module.training = was_training

    return cast(Callable[Concatenate[Any, P], Measurement], wrapped)


def validate_task_quality_generation_budget(
    max_new_tokens: int,
    *,
    sample_count: int | None = None,
) -> None:
    """Reject task-quality generation settings that exceed the shared workload budget."""

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TaskQualityEvaluationError("task-quality max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= MAX_TASK_QUALITY_NEW_TOKENS:
        raise TaskQualityEvaluationError(
            f"task-quality max_new_tokens must be between 1 and {MAX_TASK_QUALITY_NEW_TOKENS}"
        )
    if sample_count is None:
        return
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise TaskQualityEvaluationError("task-quality sample count must be a positive integer")
    if sample_count > MAX_TASK_QUALITY_EXAMPLES:
        raise TaskQualityEvaluationError(
            f"task-quality sample count must not exceed {MAX_TASK_QUALITY_EXAMPLES}"
        )
    if sample_count * max_new_tokens > MAX_TASK_QUALITY_TOTAL_NEW_TOKENS:
        raise TaskQualityEvaluationError(
            "task-quality generation budget exceeds "
            f"{MAX_TASK_QUALITY_TOTAL_NEW_TOKENS} total new tokens"
        )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskQualityEvaluationError(f"{label} must be a non-empty string")
    return value


def _normalise_code(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        normalized = "\n".join(lines[1:-1]).strip()
    return normalized


@dataclass(frozen=True, slots=True)
class CodeExactMatchExample:
    """One prompt/reference pair from a pinned JSONL task set."""

    example_id: str
    prompt: str
    reference: str

    def to_record(self) -> dict[str, str]:
        return {
            "id": self.example_id,
            "prompt": self.prompt,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class CodeExactMatchDataset:
    """Content-addressed code benchmark loaded from a local UTF-8 JSONL file."""

    path: Path
    revision: str | None
    split: str
    digest: str
    examples: tuple[CodeExactMatchExample, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        revision: str | None = None,
        split: str = "test",
        max_samples: int | None = None,
    ) -> CodeExactMatchDataset:
        resolved = Path(path).expanduser().absolute().resolve(strict=False)
        if not resolved.is_file():
            raise TaskQualityEvaluationError(f"task-quality dataset does not exist: {resolved}")
        if not split.strip():
            raise TaskQualityEvaluationError("task-quality dataset split cannot be blank")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or not 1 <= max_samples <= MAX_TASK_QUALITY_EXAMPLES
        ):
            raise TaskQualityEvaluationError(
                f"code exact-match max_samples must be within 1..{MAX_TASK_QUALITY_EXAMPLES}"
            )
        digest = hashlib.sha256()
        examples: list[CodeExactMatchExample] = []
        seen: set[str] = set()
        try:
            with resolved.open("rb") as dataset_file:
                total_bytes = 0
                line_number = 0
                while payload_line := dataset_file.readline(MAX_TASK_QUALITY_ROW_BYTES + 1):
                    line_number += 1
                    if line_number > MAX_TASK_QUALITY_ROWS:
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset exceeds {MAX_TASK_QUALITY_ROWS} rows"
                        )
                    if len(payload_line) > MAX_TASK_QUALITY_ROW_BYTES:
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset line {line_number} exceeds "
                            f"{MAX_TASK_QUALITY_ROW_BYTES} bytes"
                        )
                    total_bytes += len(payload_line)
                    if total_bytes > MAX_TASK_QUALITY_DATASET_BYTES:
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset exceeds {MAX_TASK_QUALITY_DATASET_BYTES} bytes"
                        )
                    digest.update(payload_line)
                    try:
                        line = payload_line.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise TaskQualityEvaluationError(
                            "task-quality dataset must be UTF-8"
                        ) from error
                    if not line.strip():
                        continue
                    if max_samples is None and len(examples) >= MAX_TASK_QUALITY_EXAMPLES:
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset exceeds {MAX_TASK_QUALITY_EXAMPLES} examples"
                        )
                    try:
                        raw = json.loads(line)
                    except (json.JSONDecodeError, RecursionError) as error:
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset line {line_number} is not valid JSON"
                        ) from error
                    if not isinstance(raw, dict):
                        raise TaskQualityEvaluationError(
                            f"task-quality dataset line {line_number} must be an object"
                        )
                    example_id = _text(raw.get("id"), f"task-quality dataset line {line_number}.id")
                    if example_id in seen:
                        raise TaskQualityEvaluationError(
                            f"duplicate task-quality example id: {example_id}"
                        )
                    seen.add(example_id)
                    prompt = _text(
                        raw.get("prompt"),
                        f"task-quality dataset line {line_number}.prompt",
                    )
                    reference = _text(
                        raw.get("reference"),
                        f"task-quality dataset line {line_number}.reference",
                    )
                    if max_samples is None or len(examples) < max_samples:
                        examples.append(CodeExactMatchExample(example_id, prompt, reference))
        except OSError as error:
            raise TaskQualityEvaluationError(
                f"cannot read task-quality dataset: {error}"
            ) from error
        if not examples:
            raise TaskQualityEvaluationError(
                "task-quality dataset must contain at least one example"
            )
        return cls(resolved, revision, split, f"sha256:{digest.hexdigest()}", tuple(examples))

    def to_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "revision": self.revision,
            "split": self.split,
            "digest": self.digest,
            "example_count": len(self.examples),
            "example_ids": [example.example_id for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class CodeExactMatchMeasurement:
    dataset: CodeExactMatchDataset
    matches: int
    total: int
    output_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.total != len(self.dataset.examples) or self.total <= 0:
            raise TaskQualityEvaluationError("task-quality measurement has an invalid sample count")
        if not 0 <= self.matches <= self.total:
            raise TaskQualityEvaluationError("task-quality measurement has an invalid match count")

    @property
    def score(self) -> float:
        return self.matches / self.total

    def to_record(self) -> dict[str, object]:
        return {
            "metric": "code_exact_match",
            "evaluator": TASK_QUALITY_EVALUATOR_VERSION,
            "measurement_authority": "physical_model_evaluation",
            "dataset": self.dataset.to_record(),
            "matches": self.matches,
            "total": self.total,
            "exact_match_accuracy": self.score,
            "output_digests": [
                {"id": example_id, "digest": digest} for example_id, digest in self.output_digests
            ],
        }


@dataclass(frozen=True, slots=True)
class MultipleChoiceExample:
    """One prompt, finite answer set, and correct answer index."""

    example_id: str
    prompt: str
    choices: tuple[str, ...]
    answer_index: int

    def __post_init__(self) -> None:
        _text(self.example_id, "multiple-choice example ID")
        if not isinstance(self.prompt, str):
            raise TaskQualityEvaluationError(
                f"multiple-choice prompt for {self.example_id} must be a string"
            )
        if not 2 <= len(self.choices) <= MAX_MULTIPLE_CHOICE_OPTIONS:
            raise TaskQualityEvaluationError(
                f"multiple-choice example {self.example_id} must have 2.."
                f"{MAX_MULTIPLE_CHOICE_OPTIONS} choices"
            )
        if any(not isinstance(choice, str) or not choice.strip() for choice in self.choices):
            raise TaskQualityEvaluationError(
                f"multiple-choice example {self.example_id} has a blank choice"
            )
        if len(set(self.choices)) != len(self.choices):
            raise TaskQualityEvaluationError(
                f"multiple-choice example {self.example_id} has duplicate choices"
            )
        if (
            isinstance(self.answer_index, bool)
            or not isinstance(self.answer_index, int)
            or not 0 <= self.answer_index < len(self.choices)
        ):
            raise TaskQualityEvaluationError(
                f"multiple-choice example {self.example_id} has an invalid answer index"
            )
        if any(
            len(value) > MAX_MULTIPLE_CHOICE_TEXT_CHARS for value in (self.prompt, *self.choices)
        ):
            raise TaskQualityEvaluationError(
                f"multiple-choice example {self.example_id} exceeds the text-size limit"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.example_id,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "answer_index": self.answer_index,
        }


@dataclass(frozen=True, slots=True)
class MultipleChoiceDataset:
    """Bounded, content-addressed multiple-choice JSONL benchmark."""

    path: Path
    revision: str | None
    split: str
    digest: str
    examples: tuple[MultipleChoiceExample, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        revision: str | None = None,
        split: str = "test",
        max_samples: int | None = None,
    ) -> MultipleChoiceDataset:
        resolved = Path(path).expanduser().absolute().resolve(strict=False)
        if not resolved.is_file():
            raise TaskQualityEvaluationError(f"task-quality dataset does not exist: {resolved}")
        if not isinstance(split, str) or not split.strip():
            raise TaskQualityEvaluationError("task-quality dataset split cannot be blank")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or not 1 <= max_samples <= MAX_TASK_QUALITY_EXAMPLES
        ):
            raise TaskQualityEvaluationError(
                f"multiple-choice max_samples must be within 1..{MAX_TASK_QUALITY_EXAMPLES}"
            )
        try:
            with resolved.open("rb") as stream:
                payload = stream.read(MAX_MULTIPLE_CHOICE_DATASET_BYTES + 1)
            if len(payload) > MAX_MULTIPLE_CHOICE_DATASET_BYTES:
                raise TaskQualityEvaluationError("multiple-choice dataset exceeds 16 MiB")
            text = payload.decode("utf-8")
        except OSError as error:
            raise TaskQualityEvaluationError(
                f"cannot read task-quality dataset: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise TaskQualityEvaluationError("task-quality dataset must be UTF-8") from error
        examples: list[MultipleChoiceExample] = []
        seen: set[str] = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number} is not valid JSON"
                ) from error
            if not isinstance(raw, dict):
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number} must be an object"
                )
            example_id = _text(raw.get("id"), f"task-quality dataset line {line_number}.id")
            if example_id in seen:
                raise TaskQualityEvaluationError(f"duplicate task-quality example id: {example_id}")
            seen.add(example_id)
            choices_value = raw.get("choices")
            if not isinstance(choices_value, list) or any(
                not isinstance(choice, str) for choice in choices_value
            ):
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number}.choices must be a string array"
                )
            prompt_value = raw.get("prompt")
            if not isinstance(prompt_value, str):
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number}.prompt must be a string"
                )
            answer_index = raw.get("answer_index")
            if isinstance(answer_index, bool) or not isinstance(answer_index, int):
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number}.answer_index must be an integer"
                )
            try:
                examples.append(
                    MultipleChoiceExample(
                        example_id=example_id,
                        prompt=prompt_value,
                        choices=tuple(choices_value),
                        answer_index=answer_index,
                    )
                )
            except TaskQualityEvaluationError as error:
                raise TaskQualityEvaluationError(
                    f"task-quality dataset line {line_number}: {error}"
                ) from error
            if len(examples) > MAX_TASK_QUALITY_EXAMPLES:
                raise TaskQualityEvaluationError(
                    f"multiple-choice datasets cannot exceed {MAX_TASK_QUALITY_EXAMPLES} examples"
                )
        if max_samples is not None:
            examples = examples[:max_samples]
        if not examples:
            raise TaskQualityEvaluationError(
                "task-quality dataset must contain at least one example"
            )
        return cls(resolved, revision, split, f"sha256:{_sha256(payload)}", tuple(examples))

    def to_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "revision": self.revision,
            "split": self.split,
            "digest": self.digest,
            "example_count": len(self.examples),
            "example_ids": [example.example_id for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class MultipleChoiceMeasurement:
    dataset: MultipleChoiceDataset
    correct: int
    total: int
    predictions: tuple[tuple[str, int, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        if self.total != len(self.dataset.examples) or self.total <= 0:
            raise TaskQualityEvaluationError("task-quality measurement has an invalid sample count")
        if not 0 <= self.correct <= self.total or len(self.predictions) != self.total:
            raise TaskQualityEvaluationError("task-quality measurement has invalid predictions")
        examples = {example.example_id: example for example in self.dataset.examples}
        observed_ids = tuple(example_id for example_id, _, _ in self.predictions)
        if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(examples):
            raise TaskQualityEvaluationError("task-quality predictions do not match the dataset")
        counted_correct = 0
        for example_id, predicted, scores in self.predictions:
            example = examples[example_id]
            if not 0 <= predicted < len(example.choices) or len(scores) != len(example.choices):
                raise TaskQualityEvaluationError(
                    "task-quality prediction has invalid choice scores"
                )
            if any(not math.isfinite(score) for score in scores):
                raise TaskQualityEvaluationError("task-quality choice scores must be finite")
            counted_correct += int(predicted == example.answer_index)
        if counted_correct != self.correct:
            raise TaskQualityEvaluationError(
                "task-quality correct count disagrees with predictions"
            )

    @property
    def score(self) -> float:
        return self.correct / self.total

    def to_record(self) -> dict[str, object]:
        examples = {example.example_id: example for example in self.dataset.examples}
        return {
            "metric": "multiple_choice_accuracy",
            "evaluator": MULTIPLE_CHOICE_EVALUATOR_VERSION,
            "measurement_authority": "physical_model_evaluation",
            "dataset": self.dataset.to_record(),
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.score,
            "predictions": [
                {
                    "id": example_id,
                    "answer_index": examples[example_id].answer_index,
                    "predicted_index": predicted,
                    "choice_log_likelihoods": list(log_likelihoods),
                }
                for example_id, predicted, log_likelihoods in self.predictions
            ],
        }


def _encoded_ids(tokenizer: Any, text: str, torch: Any) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    value = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if value is not None and torch.is_tensor(value):
        if value.ndim == 2 and int(value.shape[0]) == 1:
            value = value[0].tolist()
        elif value.ndim == 1:
            value = value.tolist()
        else:
            raise TaskQualityEvaluationError("tokenizer returned invalid multiple-choice input IDs")
    if not isinstance(value, (tuple, list)) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise TaskQualityEvaluationError("tokenizer returned invalid multiple-choice input IDs")
    return list(value)


def _choice_tokens(
    prompt: str, choice: str, tokenizer: Any, torch: Any
) -> tuple[list[int], list[int]]:
    context = prompt.rstrip()
    continuation = prompt[len(context) :] + choice
    context_ids = _encoded_ids(tokenizer, context, torch)
    combined_ids = _encoded_ids(tokenizer, context + continuation, torch)
    shared_prefix_length = 0
    for context_id, combined_id in zip(context_ids, combined_ids, strict=False):
        if context_id != combined_id:
            break
        shared_prefix_length += 1
    context_ids = combined_ids[:shared_prefix_length]
    continuation_ids = combined_ids[shared_prefix_length:]
    if not context_ids:
        prefix_token = getattr(tokenizer, "bos_token_id", None)
        if not isinstance(prefix_token, int):
            prefix_token = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(prefix_token, int) or prefix_token < 0:
            raise TaskQualityEvaluationError(
                "multiple-choice tokenizer requires a BOS or EOS prefix token for empty prompts"
            )
        context_ids = [prefix_token]
    if not continuation_ids:
        raise TaskQualityEvaluationError(
            "multiple-choice option tokenized to an empty continuation"
        )
    return context_ids, continuation_ids


def _choice_log_likelihood(
    model: Any,
    torch: Any,
    context_ids: list[int],
    continuation_ids: list[int],
    *,
    input_device: Any,
    max_input_tokens: int,
) -> float:
    sequence = [*context_ids, *continuation_ids]
    if len(sequence) > max_input_tokens:
        raise TaskQualityEvaluationError(
            f"multiple-choice input has {len(sequence)} tokens; limit is {max_input_tokens}"
        )
    if len(sequence) > MAX_MULTIPLE_CHOICE_INPUT_TOKENS:
        raise TaskQualityEvaluationError("multiple-choice input exceeds the hard token limit")
    input_ids = torch.tensor([sequence], dtype=torch.long, device=input_device)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids)
    except Exception as error:
        raise TaskQualityEvaluationError(
            f"multiple-choice scoring failed: {type(error).__name__}"
        ) from error
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if (
        logits is None
        or not torch.is_tensor(logits)
        or logits.ndim != 3
        or int(logits.shape[0]) != 1
        or int(logits.shape[1]) != len(sequence)
    ):
        raise TaskQualityEvaluationError(
            "model returned invalid logits for multiple-choice scoring"
        )
    start = len(context_ids) - 1
    stop = start + len(continuation_ids)
    if stop > len(sequence) - 1:
        raise TaskQualityEvaluationError("model logits do not cover the full choice continuation")
    selected_logits = logits[0, start:stop].float()
    if any(token_id >= int(logits.shape[-1]) for token_id in continuation_ids):
        raise TaskQualityEvaluationError(
            "tokenizer returned a choice ID outside the model vocabulary"
        )
    target_ids = torch.tensor(continuation_ids, dtype=torch.long, device=logits.device)
    log_probabilities = torch.log_softmax(selected_logits, dim=-1)
    score = log_probabilities.gather(1, target_ids.unsqueeze(1)).sum()
    value = float(score.item())
    if not math.isfinite(value):
        raise TaskQualityEvaluationError("model returned a non-finite choice log-likelihood")
    return value


@_preserve_model_training_modes
def evaluate_multiple_choice_accuracy(
    model: Any,
    tokenizer: Any,
    torch: Any,
    dataset: MultipleChoiceDataset,
    *,
    max_input_tokens: int = 4096,
    input_device: Any | None = None,
) -> MultipleChoiceMeasurement:
    """Choose the answer with highest teacher-forced continuation log-likelihood."""

    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not 1 <= max_input_tokens <= MAX_MULTIPLE_CHOICE_INPUT_TOKENS
    ):
        raise TaskQualityEvaluationError(
            f"multiple-choice max_input_tokens must be within 1..{MAX_MULTIPLE_CHOICE_INPUT_TOKENS}"
        )
    if input_device is None:
        try:
            embeddings = model.get_input_embeddings()
            input_device = embeddings.weight.device
        except (AttributeError, TypeError, StopIteration):
            try:
                input_device = next(iter(model.parameters())).device
            except (AttributeError, StopIteration) as error:
                raise TaskQualityEvaluationError(
                    "multiple-choice model has no input parameter device"
                ) from error
    model.eval()
    prepared: list[tuple[MultipleChoiceExample, tuple[tuple[list[int], list[int]], ...]]] = []
    total_evaluations = 0
    total_scored_tokens = 0
    for example in dataset.examples:
        options = tuple(
            _choice_tokens(example.prompt, choice, tokenizer, torch) for choice in example.choices
        )
        if any(
            len(context_ids) + len(continuation_ids) > max_input_tokens
            for context_ids, continuation_ids in options
        ):
            raise TaskQualityEvaluationError(
                f"multiple-choice example {example.example_id} exceeds max_input_tokens"
            )
        total_evaluations += len(options)
        total_scored_tokens += sum(
            len(context_ids) + len(continuation_ids) for context_ids, continuation_ids in options
        )
        if total_evaluations > MAX_MULTIPLE_CHOICE_FORWARD_EVALUATIONS:
            raise TaskQualityEvaluationError(
                "multiple-choice benchmark exceeds the forward-evaluation budget"
            )
        if total_scored_tokens > MAX_MULTIPLE_CHOICE_TOTAL_SCORED_TOKENS:
            raise TaskQualityEvaluationError(
                "multiple-choice benchmark exceeds the total scored-token budget"
            )
        prepared.append((example, options))
    correct = 0
    predictions: list[tuple[str, int, tuple[float, ...]]] = []
    for example, options in prepared:
        scores = tuple(
            _choice_log_likelihood(
                model,
                torch,
                context_ids,
                continuation_ids,
                input_device=input_device,
                max_input_tokens=max_input_tokens,
            )
            for context_ids, continuation_ids in options
        )
        predicted = max(range(len(scores)), key=lambda index: (scores[index], -index))
        correct += int(predicted == example.answer_index)
        predictions.append((example.example_id, predicted, scores))
    return MultipleChoiceMeasurement(dataset, correct, len(dataset.examples), tuple(predictions))


@_preserve_model_training_modes
def evaluate_code_exact_match(
    model: Any,
    tokenizer: Any,
    torch: Any,
    dataset: CodeExactMatchDataset,
    *,
    max_new_tokens: int,
) -> CodeExactMatchMeasurement:
    """Run deterministic greedy generation and compare normalized completions."""

    validate_task_quality_generation_budget(max_new_tokens, sample_count=len(dataset.examples))
    try:
        input_device = next(iter(model.parameters())).device
    except (AttributeError, StopIteration) as error:
        raise TaskQualityEvaluationError("task-quality model has no parameter device") from error
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad_token_id, int):
        raise TaskQualityEvaluationError("task-quality tokenizer has no pad or EOS token")
    matches = 0
    output_digests: list[tuple[str, str]] = []
    model.eval()
    for example in dataset.examples:
        encoded = tokenizer(example.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids_value = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if (
            input_ids_value is None
            or not torch.is_tensor(input_ids_value)
            or input_ids_value.ndim != 2
            or int(input_ids_value.shape[0]) != 1
        ):
            raise TaskQualityEvaluationError(
                f"task-quality tokenizer returned invalid input_ids for {example.example_id}"
            )
        input_ids = cast(Any, input_ids_value)
        input_ids = input_ids.to(input_device)
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, Mapping) else None
        if torch.is_tensor(attention_mask):
            attention_mask = cast(Any, attention_mask).to(input_device)
        kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            with torch.inference_mode():
                generated = model.generate(**kwargs)
        except Exception as error:
            raise TaskQualityEvaluationError(
                f"task-quality generation failed for {example.example_id}: {type(error).__name__}"
            ) from error
        if not torch.is_tensor(generated) or generated.ndim != 2 or int(generated.shape[0]) != 1:
            raise TaskQualityEvaluationError(
                f"task-quality generation returned invalid output for {example.example_id}"
            )
        completion = tokenizer.decode(
            generated[0, int(input_ids.shape[-1]) :], skip_special_tokens=True
        )
        if not isinstance(completion, str):
            raise TaskQualityEvaluationError(
                f"task-quality tokenizer returned a non-text completion for {example.example_id}"
            )
        normalized = _normalise_code(completion)
        matched = normalized == _normalise_code(example.reference)
        matches += int(matched)
        output_digests.append((example.example_id, f"sha256:{_sha256(normalized.encode('utf-8'))}"))
    return CodeExactMatchMeasurement(dataset, matches, len(dataset.examples), tuple(output_digests))


def evaluate_code_exact_match_with_generator(
    dataset: CodeExactMatchDataset,
    generate: Callable[[str], str],
    *,
    max_new_tokens: int,
) -> CodeExactMatchMeasurement:
    """Measure exact-match quality through a runtime-specific text generator."""

    validate_task_quality_generation_budget(max_new_tokens, sample_count=len(dataset.examples))
    matches = 0
    output_digests: list[tuple[str, str]] = []
    for example in dataset.examples:
        completion = generate(example.prompt)
        if not isinstance(completion, str):
            raise TaskQualityEvaluationError(
                f"task-quality generator returned non-text output for {example.example_id}"
            )
        normalized = _normalise_code(completion)
        matches += int(normalized == _normalise_code(example.reference))
        output_digests.append((example.example_id, f"sha256:{_sha256(normalized.encode('utf-8'))}"))
    return CodeExactMatchMeasurement(dataset, matches, len(dataset.examples), tuple(output_digests))


def evaluate_code_quality_gate(
    baseline_score: float,
    candidate_score: float,
    *,
    min_quality_retention_ratio: float,
) -> dict[str, object]:
    """Apply the declared higher-is-better retention bound to code accuracy."""

    values = (baseline_score, candidate_score, min_quality_retention_ratio)
    if any(not math.isfinite(value) for value in values):
        raise TaskQualityEvaluationError("task-quality gate values must be finite")
    if not 0.0 <= baseline_score <= 1.0 or not 0.0 <= candidate_score <= 1.0:
        raise TaskQualityEvaluationError("task-quality scores must be between zero and one")
    if not 0.0 <= min_quality_retention_ratio <= 1.0:
        raise TaskQualityEvaluationError("min_quality_retention_ratio must be between zero and one")
    if baseline_score <= 0.0:
        return {
            "metric": "code_exact_match",
            "measurement_authority": "physical_model_evaluation",
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": candidate_score - baseline_score,
            "quality_retention_ratio": None,
            "min_quality_retention_ratio": min_quality_retention_ratio,
            "accepted": False,
            "inconclusive": True,
            "rejection_reasons": [
                "task-quality retention cannot be evaluated from a zero baseline score"
            ],
        }
    retention_ratio = candidate_score / baseline_score
    passed = retention_ratio >= min_quality_retention_ratio
    return {
        "metric": "code_exact_match",
        "measurement_authority": "physical_model_evaluation",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - baseline_score,
        "quality_retention_ratio": retention_ratio,
        "min_quality_retention_ratio": min_quality_retention_ratio,
        "accepted": passed,
        "rejection_reasons": [] if passed else ["minimum task-quality retention ratio violated"],
    }


def evaluate_task_quality_gate(
    baseline_score: float,
    candidate_score: float,
    *,
    metric: str,
    min_quality_retention_ratio: float,
) -> dict[str, object]:
    """Apply the measured retention rule to a supported higher-is-better metric."""

    if metric == "code_exact_match":
        return evaluate_code_quality_gate(
            baseline_score,
            candidate_score,
            min_quality_retention_ratio=min_quality_retention_ratio,
        )
    if metric != "multiple_choice_accuracy":
        raise TaskQualityEvaluationError(f"unsupported task-quality metric: {metric!r}")
    values = (baseline_score, candidate_score, min_quality_retention_ratio)
    if any(not math.isfinite(value) for value in values):
        raise TaskQualityEvaluationError("task-quality gate values must be finite")
    if not 0.0 <= baseline_score <= 1.0 or not 0.0 <= candidate_score <= 1.0:
        raise TaskQualityEvaluationError("task-quality scores must be between zero and one")
    if not 0.0 <= min_quality_retention_ratio <= 1.0:
        raise TaskQualityEvaluationError("min_quality_retention_ratio must be between zero and one")
    if baseline_score <= 0.0:
        return {
            "metric": metric,
            "measurement_authority": "physical_model_evaluation",
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": candidate_score - baseline_score,
            "quality_retention_ratio": None,
            "min_quality_retention_ratio": min_quality_retention_ratio,
            "accepted": False,
            "inconclusive": True,
            "rejection_reasons": [
                "task-quality retention cannot be evaluated from a zero baseline score"
            ],
        }
    retention_ratio = candidate_score / baseline_score
    passed = retention_ratio >= min_quality_retention_ratio
    return {
        "metric": metric,
        "measurement_authority": "physical_model_evaluation",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - baseline_score,
        "quality_retention_ratio": retention_ratio,
        "min_quality_retention_ratio": min_quality_retention_ratio,
        "accepted": passed,
        "rejection_reasons": [] if passed else ["minimum task-quality retention ratio violated"],
    }


__all__ = [
    "MULTIPLE_CHOICE_EVALUATOR_VERSION",
    "TASK_QUALITY_EVALUATOR_VERSION",
    "CodeExactMatchDataset",
    "CodeExactMatchExample",
    "CodeExactMatchMeasurement",
    "MultipleChoiceDataset",
    "MultipleChoiceExample",
    "MultipleChoiceMeasurement",
    "TaskQualityEvaluationError",
    "evaluate_code_exact_match",
    "evaluate_code_exact_match_with_generator",
    "evaluate_code_quality_gate",
    "evaluate_multiple_choice_accuracy",
    "evaluate_task_quality_gate",
    "validate_task_quality_generation_budget",
]
