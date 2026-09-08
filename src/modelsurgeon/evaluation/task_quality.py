"""Pinned, user-supplied task-quality evaluation for first-party runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

TASK_QUALITY_EVALUATOR_VERSION = "code_exact_match_v1"


class TaskQualityEvaluationError(ValueError):
    """Raised when task-quality evidence cannot be measured safely."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskQualityEvaluationError(f"{label} must be a non-empty string")
    return value


def _normalise_code(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


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
        try:
            payload = resolved.read_bytes()
            text = payload.decode("utf-8")
        except OSError as error:
            raise TaskQualityEvaluationError(
                f"cannot read task-quality dataset: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise TaskQualityEvaluationError("task-quality dataset must be UTF-8") from error
        examples: list[CodeExactMatchExample] = []
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
            examples.append(
                CodeExactMatchExample(
                    example_id,
                    _text(raw.get("prompt"), f"task-quality dataset line {line_number}.prompt"),
                    _text(
                        raw.get("reference"),
                        f"task-quality dataset line {line_number}.reference",
                    ),
                )
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise TaskQualityEvaluationError("task-quality max_samples must be positive")
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
                {"id": example_id, "digest": digest}
                for example_id, digest in self.output_digests
            ],
        }


def evaluate_code_exact_match(
    model: Any,
    tokenizer: Any,
    torch: Any,
    dataset: CodeExactMatchDataset,
    *,
    max_new_tokens: int,
) -> CodeExactMatchMeasurement:
    """Run deterministic greedy generation and compare normalized completions."""

    if max_new_tokens <= 0:
        raise TaskQualityEvaluationError("task-quality max_new_tokens must be positive")
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
        raise TaskQualityEvaluationError(
            "min_quality_retention_ratio must be between zero and one"
        )
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


__all__ = [
    "TASK_QUALITY_EVALUATOR_VERSION",
    "CodeExactMatchDataset",
    "CodeExactMatchExample",
    "CodeExactMatchMeasurement",
    "TaskQualityEvaluationError",
    "evaluate_code_exact_match",
    "evaluate_code_quality_gate",
]
