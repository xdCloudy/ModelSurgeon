"""Streaming Hugging Face text calibration adapter with bounded tokenization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from modelsurgeon.datasets.calibration import CalibrationContract, CalibrationSample


class HuggingFaceCalibrationError(RuntimeError):
    """Raised when optional dependencies or streamed records violate the contract."""


@dataclass(frozen=True, slots=True)
class HuggingFaceCalibrationRequest:
    contract: CalibrationContract
    text_field: str = "text"
    batch_size: int = 8
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.text_field or self.batch_size <= 0 or self.max_tokens <= 0:
            raise ValueError(
                "text field, positive batch size, and positive token limit are required"
            )


@dataclass(frozen=True, slots=True)
class TokenizedCalibrationSample:
    identity: CalibrationSample
    input_ids: tuple[int, ...]

    def to_record(self) -> dict[str, object]:
        return {**self.identity.to_record(), "input_ids": list(self.input_ids)}


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    contract_record: dict[str, object]
    samples: tuple[TokenizedCalibrationSample, ...]

    def to_record(self) -> dict[str, object]:
        return {
            **self.contract_record,
            "tokenized_samples": [sample.to_record() for sample in self.samples],
        }

    def write(self, path: str | Path) -> None:
        """Atomically cache canonical JSON after creating only the target parent."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":")) + "\n"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)


def _batches(
    values: Iterable[tuple[CalibrationSample, str]], size: int
) -> Iterator[list[tuple[CalibrationSample, str]]]:
    batch: list[tuple[CalibrationSample, str]] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def stream_huggingface_calibration(
    request: HuggingFaceCalibrationRequest,
) -> CalibrationManifest:
    """Stream, deterministically select, and tokenize without materializing the dataset."""
    try:
        datasets: Any = import_module("datasets")
        transformers: Any = import_module("transformers")
    except ImportError as error:
        raise HuggingFaceCalibrationError(
            "Hugging Face calibration requires the datasets and transformers packages"
        ) from error
    dataset = datasets.load_dataset(
        request.contract.dataset.dataset,
        split=request.contract.dataset.split,
        revision=request.contract.dataset.revision,
        streaming=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        request.contract.tokenizer.tokenizer,
        revision=request.contract.tokenizer.revision,
        trust_remote_code=False,
    )
    retained: list[tuple[bytes, bytes, CalibrationSample, str]] = []
    retain_count = request.contract.selection.sample_count
    for index, row in enumerate(dataset):
        if not isinstance(row, Mapping):
            raise HuggingFaceCalibrationError(f"streamed row {index} is not a mapping")
        text = row.get(request.text_field)
        if not isinstance(text, str):
            raise HuggingFaceCalibrationError(
                f"streamed row {index} field {request.text_field!r} is not text"
            )
        sample_id = str(row.get("id", index))
        digest = hashlib.sha256(text.encode()).hexdigest()
        identity = CalibrationSample(sample_id, digest)
        candidate = (
            request.contract.selection_rank(sample_id),
            sample_id.encode(),
            identity,
            text,
        )
        duplicate = next(
            (item for item in retained if item[2].sample_id == sample_id), None
        )
        if duplicate is not None:
            if duplicate[2].content_sha256 != digest:
                raise HuggingFaceCalibrationError(
                    f"streamed sample ID {sample_id!r} has conflicting content"
                )
            continue
        if len(retained) < retain_count:
            retained.append(candidate)
        else:
            worst = max(range(len(retained)), key=lambda item: retained[item][:2])
            if candidate[:2] < retained[worst][:2]:
                retained[worst] = candidate
    retained.sort(key=lambda item: item[:2])
    selected_pairs = tuple((item[2], item[3]) for item in retained)
    selected = request.contract.select(tuple(identity for identity, _ in selected_pairs))
    text_by_id = {identity.sample_id: text for identity, text in selected_pairs}
    tokenized: list[TokenizedCalibrationSample] = []
    selected_text = ((sample, text_by_id[sample.sample_id]) for sample in selected)
    for batch in _batches(selected_text, request.batch_size):
        encoded = tokenizer(
            [text for _, text in batch],
            truncation=True,
            max_length=request.max_tokens,
            add_special_tokens=True,
        )
        rows = encoded.get("input_ids")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise HuggingFaceCalibrationError("tokenizer returned an invalid input_ids batch")
        for (identity, _), input_ids in zip(batch, rows, strict=True):
            if not isinstance(input_ids, list) or any(
                not isinstance(item, int) for item in input_ids
            ):
                raise HuggingFaceCalibrationError("tokenizer input IDs must be integer lists")
            tokenized.append(TokenizedCalibrationSample(identity, tuple(input_ids)))
    return CalibrationManifest(request.contract.to_record(selected), tuple(tokenized))
