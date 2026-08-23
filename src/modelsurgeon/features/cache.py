"""Atomic disk-backed feature partitions with exact revision/version invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from modelsurgeon.features.schema import (
    FEATURE_SCHEMA_VERSION,
    ErrorProvenance,
    FeatureKind,
    FeatureRecord,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId

FEATURE_CACHE_SCHEMA_VERSION: Final[int] = 1


class FeatureCacheError(RuntimeError):
    """Raised when a published cache partition is corrupt or inconsistent."""


@dataclass(frozen=True, slots=True)
class FeaturePartitionKey:
    model_revision: str
    input_revision: str
    component_id: ComponentId
    extractor: str
    extractor_version: str

    def __post_init__(self) -> None:
        required = (
            self.model_revision,
            self.input_revision,
            self.extractor,
            self.extractor_version,
        )
        if any(not value for value in required):
            raise FeatureCacheError("feature partition key fields cannot be empty")

    def to_record(self) -> dict[str, str]:
        return {
            "model_revision": self.model_revision,
            "input_revision": self.input_revision,
            "component_id": str(self.component_id),
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
        }

    @property
    def digest(self) -> str:
        encoded = _canonical_json_bytes(self.to_record())
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FeaturePartition:
    key: FeaturePartitionKey
    records: tuple[FeatureRecord, ...]
    records_sha256: str

    def __post_init__(self) -> None:
        if not self.records:
            raise FeatureCacheError("feature partitions cannot be empty")
        if len(self.records_sha256) != 64:
            raise FeatureCacheError("feature partition checksum must be SHA-256 hex")
        for record in self.records:
            if record.component_id != self.key.component_id:
                raise FeatureCacheError("partition record component does not match partition key")
            if record.extractor != self.key.extractor:
                raise FeatureCacheError("partition record extractor does not match partition key")
            if record.extractor_version != self.key.extractor_version:
                raise FeatureCacheError(
                    "partition record extractor version does not match partition key"
                )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _records_bytes(records: tuple[FeatureRecord, ...]) -> bytes:
    return _canonical_json_bytes([record.to_record() for record in records])


def _metadata(raw: object) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    if not isinstance(raw, dict):
        raise FeatureCacheError("feature metadata must be an object")
    output: list[tuple[str, str | int | float | bool | None]] = []
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None))):
            raise FeatureCacheError("feature metadata contains unsupported values")
        output.append((key, value))
    return tuple(sorted(output))


def _error_provenance(raw: object) -> ErrorProvenance | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise FeatureCacheError("precision error provenance must be an object")
    try:
        return ErrorProvenance(
            absolute_error=float(raw["absolute_error"]),
            relative_error=float(raw["relative_error"]),
            reference_dtype=str(raw["reference_dtype"]),
            method=str(raw["method"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureCacheError("invalid precision error provenance") from error


def _precision(raw: object) -> PrecisionProvenance:
    if not isinstance(raw, dict):
        raise FeatureCacheError("feature precision must be an object")
    try:
        source = PrecisionSource(str(raw["source"]))
        quantization_raw = raw.get("quantization")
        codec_raw = raw.get("codec_version")
        return PrecisionProvenance(
            source=source,
            storage_dtype=str(raw["storage_dtype"]),
            compute_dtype=str(raw["compute_dtype"]),
            quantization=None if quantization_raw is None else str(quantization_raw),
            codec_version=None if codec_raw is None else str(codec_raw),
            error=_error_provenance(raw.get("error")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureCacheError("invalid feature precision provenance") from error


def _sample_context(raw: object) -> FeatureSampleContext | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise FeatureCacheError("feature sample context must be an object")
    sample_ids = raw.get("sample_ids")
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise FeatureCacheError("feature sample IDs must be a string list")
    try:
        return FeatureSampleContext(
            dataset=str(raw["dataset"]),
            revision=str(raw["revision"]),
            split=str(raw["split"]),
            sample_ids=tuple(sample_ids),
            preprocessing_version=str(raw["preprocessing_version"]),
            tokenizer=str(raw["tokenizer"]),
            tokenizer_revision=str(raw["tokenizer_revision"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureCacheError("invalid feature sample context") from error


def _feature_record(raw: object) -> FeatureRecord:
    if not isinstance(raw, dict):
        raise FeatureCacheError("cached feature record must be an object")
    if raw.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise FeatureCacheError("cached feature schema version is unsupported")
    try:
        kind = FeatureKind(str(raw["kind"]))
        value_raw = raw["value"]
        if kind is FeatureKind.SCALAR:
            if isinstance(value_raw, bool) or not isinstance(value_raw, (int, float)):
                raise FeatureCacheError("cached scalar feature value is invalid")
            value: float | tuple[float, ...] = float(value_raw)
        else:
            if not isinstance(value_raw, list) or not value_raw:
                raise FeatureCacheError("cached vector feature value is invalid")
            invalid_value = any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value_raw
            )
            if invalid_value:
                raise FeatureCacheError("cached vector feature values are invalid")
            value = tuple(float(item) for item in value_raw)
        return FeatureRecord(
            component_id=ComponentId.parse(str(raw["component_id"])),
            name=str(raw["name"]),
            kind=kind,
            value=value,
            dtype=str(raw["dtype"]),
            extractor=str(raw["extractor"]),
            extractor_version=str(raw["extractor_version"]),
            precision=_precision(raw["precision"]),
            sample_context=_sample_context(raw.get("sample_context")),
            metadata=_metadata(raw.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureCacheError("invalid cached feature record") from error


def _key_from_record(raw: object) -> FeaturePartitionKey:
    if not isinstance(raw, dict):
        raise FeatureCacheError("feature partition key must be an object")
    try:
        return FeaturePartitionKey(
            model_revision=str(raw["model_revision"]),
            input_revision=str(raw["input_revision"]),
            component_id=ComponentId.parse(str(raw["component_id"])),
            extractor=str(raw["extractor"]),
            extractor_version=str(raw["extractor_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureCacheError("invalid feature partition key") from error


class FeaturePartitionCache:
    """Store and load complete feature partitions through atomic file publication."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: FeaturePartitionKey) -> Path:
        return self.root / f"{key.digest}.json"

    def write(
        self,
        key: FeaturePartitionKey,
        records: tuple[FeatureRecord, ...],
    ) -> FeaturePartition:
        partition_records = _records_bytes(records)
        checksum = hashlib.sha256(partition_records).hexdigest()
        partition = FeaturePartition(key, records, checksum)
        payload = {
            "cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "complete": True,
            "key": key.to_record(),
            "record_count": len(records),
            "records_sha256": checksum,
            "records": [record.to_record() for record in records],
        }
        encoded = _canonical_json_bytes(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self.path_for(key)
        temporary = self.root / f".{key.digest}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)
        return partition

    def load(self, key: FeaturePartitionKey) -> FeaturePartition | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FeatureCacheError("published feature partition is unreadable") from error
        if not isinstance(raw, dict):
            raise FeatureCacheError("published feature partition must be an object")
        if raw.get("cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            raise FeatureCacheError("feature cache schema version is unsupported")
        if raw.get("complete") is not True:
            return None
        embedded_key = _key_from_record(raw.get("key"))
        if embedded_key != key:
            raise FeatureCacheError("published feature partition key does not match lookup key")
        records_raw = raw.get("records")
        if not isinstance(records_raw, list):
            raise FeatureCacheError("published feature partition records must be a list")
        if raw.get("record_count") != len(records_raw):
            raise FeatureCacheError("published feature partition record count is inconsistent")
        checksum = hashlib.sha256(_canonical_json_bytes(records_raw)).hexdigest()
        if raw.get("records_sha256") != checksum:
            raise FeatureCacheError("published feature partition checksum mismatch")
        records = tuple(_feature_record(item) for item in records_raw)
        return FeaturePartition(key, records, checksum)
