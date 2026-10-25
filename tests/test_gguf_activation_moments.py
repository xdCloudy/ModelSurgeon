from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import load_gguf_activation_moments

_SOURCE_SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_sha256": _SOURCE_SHA256,
        "calibration": {
            "dataset": "example/dataset",
            "dataset_revision": "revision",
            "split": "train",
            "text_sha256": "b" * 64,
            "tokenizer": "example/model",
            "tokenizer_revision": "revision",
        },
        "tensors": {
            "blk.0.ffn_down.weight": {
                "input_axis": 0,
                "sample_count": 4,
                "second_moments": [0.25, 1.0],
            }
        },
    }


def _write(path: Path, payload: str | dict[str, object]) -> None:
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


def test_loads_source_bound_activation_moments(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    record = _manifest()
    _write(path, record)

    tensors, calibration = load_gguf_activation_moments(
        path, expected_source_sha256=_SOURCE_SHA256
    )

    assert tensors["blk.0.ffn_down.weight"] == {
        "input_axis": 0,
        "sample_count": 4,
        "second_moments": [0.25, 1.0],
    }
    assert calibration["split"] == "train"


def test_loads_legacy_flat_collector_provenance(tmp_path: Path) -> None:
    path = tmp_path / "legacy-moments.json"
    record = _manifest()
    nested = record.pop("calibration")
    record.update(
        calibration_dataset=nested["dataset"],
        calibration_split=nested["split"],
        calibration_sha256=nested["text_sha256"],
        tokenizer=nested["tokenizer"],
        tokenizer_revision=nested["tokenizer_revision"],
        tokenizer_vocab_size=128,
        bos_token_id=1,
        added_bos=False,
        calibration_chunks=8,
        context_size=512,
        calibration_token_count=4096,
    )
    _write(path, record)

    tensors, calibration = load_gguf_activation_moments(
        path, expected_source_sha256=_SOURCE_SHA256
    )

    assert tensors["blk.0.ffn_down.weight"]["sample_count"] == 4
    assert calibration["dataset_revision"] == f"sha256:{nested['text_sha256']}"
    assert calibration["calibration_token_count"] == 4096


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item.update(source_sha256="c" * 64), "different source"),
        (lambda item: item.update(schema_version=2), "unsupported"),
        (lambda item: item["calibration"].pop("split"), "needs split"),
        (lambda item: item["calibration"].update(text_sha256="bad"), "SHA-256"),
        (lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(input_axis=True), "axis"),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(sample_count=False),
            "sample count",
        ),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(
                second_moments=[float("nan")]
            ),
            "finite",
        ),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(second_moments=[-1]),
            "non-negative",
        ),
        (lambda item: item.update(tensors={}), "non-empty"),
    ],
)
def test_rejects_invalid_activation_moment_records(
    tmp_path: Path, mutate: object, message: str
) -> None:
    record = _manifest()
    mutate(record)  # type: ignore[operator]
    path = tmp_path / "moments.json"
    _write(path, record)

    with pytest.raises(ValueError, match=message):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, '{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate key"):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_nonstandard_nan_json_values(tmp_path: Path) -> None:
    record = _manifest()
    record["tensors"]["blk.0.ffn_down.weight"]["second_moments"] = [float("nan")]
    path = tmp_path / "moments.json"
    _write(path, record)

    with pytest.raises(ValueError, match="finite"):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_manifest_over_configured_limit(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, _manifest())

    with pytest.raises(ValueError, match="file limit"):
        load_gguf_activation_moments(
            path, expected_source_sha256=_SOURCE_SHA256, max_file_bytes=8
        )
