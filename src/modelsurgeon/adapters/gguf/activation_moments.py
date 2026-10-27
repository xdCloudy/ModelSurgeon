"""Read bounded, source-bound GGUF activation moment manifests."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"activation moment manifest has duplicate key {key!r}")
        result[key] = value
    return result


def load_gguf_activation_moments(
    path: str | Path,
    *,
    expected_source_sha256: str,
    max_file_bytes: int = 16 * 1024 * 1024,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load validated calibration statistics bound to one exact GGUF digest."""
    if max_file_bytes <= 0:
        raise ValueError("activation moment file limit must be positive")
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"activation moment manifest is not a file: {resolved}")
    with resolved.open("rb") as stream:
        payload = stream.read(max_file_bytes + 1)
    if len(payload) > max_file_bytes:
        raise ValueError("activation moment manifest exceeds its configured file limit")
    record = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(record, dict):
        raise ValueError("unsupported activation moment manifest schema")
    schema_version = record.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("unsupported activation moment manifest schema")
    if record.get("source_sha256") != expected_source_sha256.removeprefix("sha256:"):
        raise ValueError("activation moments were calibrated for a different source GGUF")
    calibration = record.get("calibration")
    if calibration is None:
        legacy_text_sha256 = record.get("calibration_sha256")
        if not isinstance(legacy_text_sha256, str):
            raise ValueError("activation moment calibration provenance is missing")
        calibration = {
            "dataset": record.get("calibration_dataset"),
            "dataset_revision": f"sha256:{legacy_text_sha256}",
            "split": record.get("calibration_split"),
            "text_sha256": legacy_text_sha256,
            "tokenizer": record.get("tokenizer"),
            "tokenizer_revision": record.get("tokenizer_revision"),
            "tokenizer_vocab_size": record.get("tokenizer_vocab_size"),
            "bos_token_id": record.get("bos_token_id"),
            "added_bos": record.get("added_bos"),
            "calibration_chunks": record.get("calibration_chunks"),
            "context_size": record.get("context_size"),
            "calibration_token_count": record.get("calibration_token_count"),
        }
    if not isinstance(calibration, dict):
        raise ValueError("activation moment calibration provenance is missing")
    for key in (
        "dataset",
        "dataset_revision",
        "split",
        "text_sha256",
        "tokenizer",
        "tokenizer_revision",
    ):
        value = calibration.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"activation moment calibration provenance needs {key}")
    if re.fullmatch(r"[0-9a-f]{64}", calibration["text_sha256"]) is None:
        raise ValueError("activation calibration text digest must be a lowercase SHA-256")
    tensors = record.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("activation moment tensor table must be a non-empty object")
    validated: dict[str, dict[str, Any]] = {}
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("activation moment entries need tensor names and objects")
        axis = entry.get("input_axis")
        sample_count = entry.get("sample_count")
        values = entry.get("second_moments")
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1):
            raise ValueError(f"activation moments for {name!r} need input axis 0 or 1")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(f"activation moments for {name!r} need a positive sample count")
        if not isinstance(values, list) or not values:
            raise ValueError(f"activation moments for {name!r} need a non-empty vector")
        normalized_values: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"activation moments for {name!r} must be finite and non-negative"
                )
            try:
                normalized = float(value)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"activation moments for {name!r} must be finite and non-negative"
                ) from error
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(
                    f"activation moments for {name!r} must be finite and non-negative"
                )
            normalized_values.append(normalized)
        validated[name] = {
            "input_axis": axis,
            "sample_count": sample_count,
            "second_moments": normalized_values,
        }
    return validated, calibration
