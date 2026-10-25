from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.benchmark_gguf_quantization_budget import _load_activation_moments

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

    tensors, calibration = _load_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)

    assert tensors["blk.0.ffn_down.weight"] == {
        "input_axis": 0,
        "sample_count": 4,
        "second_moments": [0.25, 1.0],
    }
    assert calibration["split"] == "train"


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
        _load_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, '{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate key"):
        _load_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_nonstandard_nan_json_values(tmp_path: Path) -> None:
    record = _manifest()
    record["tensors"]["blk.0.ffn_down.weight"]["second_moments"] = [float("nan")]
    path = tmp_path / "moments.json"
    _write(path, record)

    with pytest.raises(ValueError, match="finite"):
        _load_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_manifest_over_configured_limit(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, _manifest())

    with pytest.raises(ValueError, match="file limit"):
        _load_activation_moments(
            path, expected_source_sha256=_SOURCE_SHA256, max_file_bytes=8
        )
