"""Tests for bounded streaming Hugging Face calibration ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelsurgeon.datasets import (
    CalibrationContract,
    CalibrationSample,
    DatasetIdentity,
    DatasetTrust,
    HuggingFaceCalibrationError,
    HuggingFaceCalibrationRequest,
    PreprocessingIdentity,
    SelectionConfig,
    TokenizerIdentity,
    huggingface,
    stream_huggingface_calibration,
)


def _contract(count: int = 5) -> CalibrationContract:
    return CalibrationContract(
        DatasetIdentity("org/data", "d" * 40, "train", "mit", DatasetTrust.TRUSTED, "reviewed"),
        PreprocessingIdentity("plain-text", "1", "a" * 64),
        TokenizerIdentity("org/tokenizer", "t" * 40, "b" * 64),
        SelectionConfig(19, count),
    )


class FakeTokenizer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, texts: list[str], **options: object) -> dict[str, object]:
        self.batch_sizes.append(len(texts))
        assert options == {"truncation": True, "max_length": 4, "add_special_tokens": True}
        return {"input_ids": [[len(text), index] for index, text in enumerate(texts)]}


def test_streams_with_pinned_revisions_and_tokenizes_only_bounded_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple({"id": f"row-{index}", "text": f"text {index}"} for index in range(100))
    calls: list[dict[str, object]] = []
    tokenizer = FakeTokenizer()

    def load_dataset(name: str, **options: object) -> object:
        calls.append({"name": name, **options})
        return (row for row in rows)

    datasets = SimpleNamespace(load_dataset=load_dataset)
    tokenizers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    )
    monkeypatch.setattr(
        huggingface,
        "import_module",
        lambda name: datasets if name == "datasets" else tokenizers,
    )
    contract = _contract()

    manifest = stream_huggingface_calibration(
        HuggingFaceCalibrationRequest(contract, batch_size=2, max_tokens=4)
    )

    all_samples = tuple(
        CalibrationSample(row["id"], hashlib.sha256(row["text"].encode()).hexdigest())
        for row in rows
    )
    assert [sample.identity for sample in manifest.samples] == list(contract.select(all_samples))
    assert calls == [
        {"name": "org/data", "split": "train", "revision": "d" * 40, "streaming": True}
    ]
    assert tokenizer.batch_sizes == [2, 2, 1]
    assert len(manifest.samples) == 5


def test_manifest_cache_is_canonical_and_replaceable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer = FakeTokenizer()
    datasets = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: iter(({"id": "x", "text": "one"},))
    )
    tokenizers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    )
    monkeypatch.setattr(
        huggingface,
        "import_module",
        lambda name: datasets if name == "datasets" else tokenizers,
    )
    manifest = stream_huggingface_calibration(
        HuggingFaceCalibrationRequest(_contract(1), max_tokens=4)
    )
    target = tmp_path / "cache" / "manifest.json"

    manifest.write(target)
    first = target.read_bytes()
    manifest.write(target)

    assert target.read_bytes() == first
    assert json.loads(first)["selection"]["sample_count"] == 1
    assert not target.with_name(".manifest.json.tmp").exists()


def test_missing_dependencies_and_malformed_rows_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(huggingface, "import_module", missing)
    with pytest.raises(HuggingFaceCalibrationError, match="datasets and transformers"):
        stream_huggingface_calibration(HuggingFaceCalibrationRequest(_contract(1)))

    datasets = SimpleNamespace(load_dataset=lambda *args, **kwargs: iter(({"id": "x"},)))
    tokenizers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        huggingface,
        "import_module",
        lambda name: datasets if name == "datasets" else tokenizers,
    )
    with pytest.raises(HuggingFaceCalibrationError, match="is not text"):
        stream_huggingface_calibration(HuggingFaceCalibrationRequest(_contract(1)))
