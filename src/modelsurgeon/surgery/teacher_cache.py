"""Bounded immutable teacher-target caching and deterministic repair selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

TEACHER_CACHE_SCHEMA_VERSION = 1
TEACHER_CACHE_PROTOCOL_REVISION = "teacher-cache-v1"
REPAIR_SELECTION_ALGORITHM = "repair-selection-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TeacherCacheError(ValueError):
    """Raised when teacher targets, cache chunks, or selection inputs are unsafe."""


class TeacherCacheTarget(StrEnum):
    LOGITS = "logits"
    FEATURES = "features"


class TeacherCacheEncoding(StrEnum):
    FP32 = "fp32"
    INT8 = "int8"


class RepairSelectionStrategy(StrEnum):
    RANDOM = "random"
    DOMAIN_BALANCED = "domain_balanced"
    DIVERSITY = "diversity"
    PREDICTED_RECOVERABILITY = "predicted_recoverability"


class SelectionExclusionReason(StrEnum):
    HELDOUT = "heldout"
    NON_TRAINING_PARTITION = "non_training_partition"
    TOKEN_BUDGET = "token_budget"
    EXAMPLE_BUDGET = "example_budget"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TeacherCacheError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise TeacherCacheError(f"{label} must be a lowercase SHA-256")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_bytes(value: object) -> bytes:
    return _canonical(value).encode("utf-8")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherCacheError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TeacherCacheError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TeacherCacheKey:
    """Compatibility identity for one immutable teacher target cache."""

    teacher_model_id: str
    teacher_model_revision: str
    tokenizer_digest: str
    example_revision: str
    target: TeacherCacheTarget
    target_schema: str
    encoding: TeacherCacheEncoding = TeacherCacheEncoding.FP32

    def __post_init__(self) -> None:
        for label, value in (
            ("teacher model ID", self.teacher_model_id),
            ("teacher model revision", self.teacher_model_revision),
            ("example revision", self.example_revision),
            ("target schema", self.target_schema),
        ):
            _text(value, label)
        _digest(self.tokenizer_digest, "teacher tokenizer digest")

    @property
    def cache_id(self) -> str:
        return "teacher_cache_" + hashlib.sha256(_canonical_bytes(self.to_record())).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
            "protocol_revision": TEACHER_CACHE_PROTOCOL_REVISION,
            "teacher_model_id": self.teacher_model_id,
            "teacher_model_revision": self.teacher_model_revision,
            "tokenizer_digest": self.tokenizer_digest,
            "example_revision": self.example_revision,
            "target": self.target.value,
            "target_schema": self.target_schema,
            "encoding": self.encoding.value,
        }


@dataclass(frozen=True, slots=True)
class TeacherCacheChunk:
    """One bounded immutable target chunk."""

    index: int
    token_count: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index < 0:
            raise TeacherCacheError("teacher cache chunk index must be non-negative")
        if isinstance(self.token_count, bool) or self.token_count <= 0:
            raise TeacherCacheError("teacher cache chunk token count must be positive")
        if not self.values or any(not math.isfinite(value) for value in self.values):
            raise TeacherCacheError("teacher cache chunks require finite values")

    def _identity_record(self) -> dict[str, object]:
        return {"index": self.index, "token_count": self.token_count, "values": list(self.values)}

    @property
    def checksum(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._identity_record())).hexdigest()

    @property
    def serialized_bytes(self) -> int:
        return len(_canonical_bytes(self.to_record()))

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["checksum"] = self.checksum
        return record


@dataclass(frozen=True, slots=True)
class TeacherCacheManifest:
    key: TeacherCacheKey
    chunks: tuple[TeacherCacheChunk, ...]
    total_tokens: int
    max_chunk_values: int

    def __post_init__(self) -> None:
        if self.max_chunk_values <= 0:
            raise TeacherCacheError("teacher cache chunk value limit must be positive")
        if not self.chunks:
            raise TeacherCacheError("published teacher caches require chunks")
        if tuple(item.index for item in self.chunks) != tuple(range(len(self.chunks))):
            raise TeacherCacheError("teacher cache chunks must be complete and canonical")
        if any(len(item.values) > self.max_chunk_values for item in self.chunks):
            raise TeacherCacheError("teacher cache chunk exceeds its value limit")
        if self.total_tokens != sum(item.token_count for item in self.chunks):
            raise TeacherCacheError("teacher cache token count does not reconcile chunks")

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_record(include_digest=False))).hexdigest()

    @property
    def serialized_bytes(self) -> int:
        return len(_canonical_bytes(self.to_record()))

    def to_record(self, *, include_digest: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
            "protocol_revision": TEACHER_CACHE_PROTOCOL_REVISION,
            "key": self.key.to_record(),
            "total_tokens": self.total_tokens,
            "max_chunk_values": self.max_chunk_values,
            "chunks": [item.to_record() for item in self.chunks],
        }
        if include_digest:
            record["artifact_sha256"] = self.artifact_sha256
        return record


@dataclass(frozen=True, slots=True)
class TeacherCacheTelemetry:
    """Exact cache hit/miss and byte accounting for one access."""

    hit: bool
    chunk_count: int
    bytes_read: int
    bytes_written: int

    def __post_init__(self) -> None:
        if self.chunk_count < 0 or self.bytes_read < 0 or self.bytes_written < 0:
            raise TeacherCacheError("teacher cache telemetry cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "hit": self.hit,
            "chunk_count": self.chunk_count,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }


@dataclass(frozen=True, slots=True)
class TeacherCacheLimits:
    max_chunk_values: int = 65_536
    max_total_bytes: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.max_chunk_values <= 0 or self.max_total_bytes <= 0:
            raise TeacherCacheError("teacher cache limits must be positive")


def _chunk_from_record(value: object) -> TeacherCacheChunk:
    if not isinstance(value, dict) or set(value) != {"index", "token_count", "values", "checksum"}:
        raise TeacherCacheError("teacher cache chunk record has missing or unknown fields")
    try:
        chunk = TeacherCacheChunk(
            int(value["index"]),
            int(value["token_count"]),
            tuple(float(item) for item in value["values"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherCacheError("teacher cache chunk record is malformed") from error
    if value["checksum"] != chunk.checksum:
        raise TeacherCacheError("teacher cache chunk checksum mismatch")
    return chunk


def _key_from_record(value: object) -> TeacherCacheKey:
    if not isinstance(value, dict):
        raise TeacherCacheError("teacher cache key record must be an object")
    try:
        if value.get("schema_version") != TEACHER_CACHE_SCHEMA_VERSION:
            raise TeacherCacheError("unsupported teacher cache schema version")
        if value.get("protocol_revision") != TEACHER_CACHE_PROTOCOL_REVISION:
            raise TeacherCacheError("unsupported teacher cache protocol revision")
        return TeacherCacheKey(
            str(value["teacher_model_id"]),
            str(value["teacher_model_revision"]),
            str(value["tokenizer_digest"]),
            str(value["example_revision"]),
            TeacherCacheTarget(str(value["target"])),
            str(value["target_schema"]),
            TeacherCacheEncoding(str(value["encoding"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherCacheError("teacher cache key record is malformed") from error


def _manifest_from_record(value: object) -> TeacherCacheManifest:
    if not isinstance(value, dict):
        raise TeacherCacheError("teacher cache manifest must be an object")
    if value.get("schema_version") != TEACHER_CACHE_SCHEMA_VERSION:
        raise TeacherCacheError("unsupported teacher cache manifest schema")
    if value.get("protocol_revision") != TEACHER_CACHE_PROTOCOL_REVISION:
        raise TeacherCacheError("unsupported teacher cache manifest protocol")
    try:
        manifest = TeacherCacheManifest(
            _key_from_record(value["key"]),
            tuple(_chunk_from_record(item) for item in value["chunks"]),
            int(value["total_tokens"]),
            int(value["max_chunk_values"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherCacheError("teacher cache manifest is malformed") from error
    if value.get("artifact_sha256") != manifest.artifact_sha256:
        raise TeacherCacheError("teacher cache manifest checksum mismatch")
    return manifest


class TeacherCacheStore:
    """Filesystem-backed immutable chunk store with resumable publication."""

    def __init__(self, root: str | Path, limits: TeacherCacheLimits | None = None) -> None:
        self.root = Path(root)
        self.limits = limits or TeacherCacheLimits()

    def _directory(self, key: TeacherCacheKey) -> Path:
        return self.root / key.cache_id

    def _chunk_path(self, key: TeacherCacheKey, index: int) -> Path:
        return self._directory(key) / f"{index:08d}.chunk.json"

    def _manifest_path(self, key: TeacherCacheKey) -> Path:
        return self._directory(key) / "manifest.json"

    def _load_chunk(self, path: Path) -> TeacherCacheChunk:
        try:
            return _chunk_from_record(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise TeacherCacheError(f"teacher cache chunk is unreadable: {path.name}") from error

    def _chunks(self, key: TeacherCacheKey) -> tuple[TeacherCacheChunk, ...]:
        directory = self._directory(key)
        paths = sorted(directory.glob("[0-9]*.chunk.json")) if directory.exists() else []
        chunks = tuple(self._load_chunk(path) for path in paths)
        if tuple(item.index for item in chunks) != tuple(range(len(chunks))):
            raise TeacherCacheError("teacher cache chunks are missing or out of order")
        if any(len(item.values) > self.limits.max_chunk_values for item in chunks):
            raise TeacherCacheError("teacher cache chunk exceeds configured limits")
        return chunks

    def load(self, key: TeacherCacheKey) -> TeacherCacheManifest | None:
        path = self._manifest_path(key)
        if not path.exists():
            return None
        try:
            manifest = _manifest_from_record(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise TeacherCacheError("teacher cache manifest is unreadable") from error
        if manifest.key != key:
            raise TeacherCacheError("teacher cache manifest key does not match lookup")
        chunks = self._chunks(key)
        if chunks != manifest.chunks:
            raise TeacherCacheError("teacher cache chunks disagree with its manifest")
        if manifest.serialized_bytes > self.limits.max_total_bytes:
            raise TeacherCacheError("teacher cache exceeds configured byte limit")
        return manifest

    def access(
        self, key: TeacherCacheKey
    ) -> tuple[TeacherCacheManifest | None, TeacherCacheTelemetry]:
        manifest = self.load(key)
        if manifest is None:
            return None, TeacherCacheTelemetry(False, 0, 0, 0)
        return manifest, TeacherCacheTelemetry(
            True,
            len(manifest.chunks),
            manifest.serialized_bytes,
            0,
        )

    def write_chunk(self, key: TeacherCacheKey, chunk: TeacherCacheChunk) -> TeacherCacheTelemetry:
        if len(chunk.values) > self.limits.max_chunk_values:
            raise TeacherCacheError("teacher cache chunk exceeds configured value limit")
        encoded = _canonical_bytes(chunk.to_record())
        if len(encoded) > self.limits.max_total_bytes:
            raise TeacherCacheError("teacher cache chunk exceeds configured byte limit")
        directory = self._directory(key)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._chunk_path(key, chunk.index)
        if path.exists():
            existing = self._load_chunk(path)
            if existing != chunk:
                raise TeacherCacheError("immutable teacher cache chunk already differs")
            return TeacherCacheTelemetry(True, 1, len(encoded), 0)
        temporary = directory / f".{path.name}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                existing = self._load_chunk(path)
                if existing != chunk:
                    raise TeacherCacheError(
                        "concurrent teacher cache publication disagreed"
                    ) from error
        finally:
            temporary.unlink(missing_ok=True)
        return TeacherCacheTelemetry(False, 1, 0, len(encoded))

    def publish(
        self,
        key: TeacherCacheKey,
        expected_chunk_count: int,
        total_tokens: int,
    ) -> TeacherCacheManifest:
        if expected_chunk_count <= 0 or total_tokens <= 0:
            raise TeacherCacheError("teacher cache publication counts must be positive")
        existing = self.load(key)
        chunks = self._chunks(key)
        if len(chunks) != expected_chunk_count:
            raise TeacherCacheError("teacher cache publication is incomplete")
        manifest = TeacherCacheManifest(key, chunks, total_tokens, self.limits.max_chunk_values)
        if manifest.serialized_bytes > self.limits.max_total_bytes:
            raise TeacherCacheError("teacher cache exceeds configured byte limit")
        if existing is not None and existing != manifest:
            raise TeacherCacheError("immutable teacher cache manifest already differs")
        if existing is not None:
            return existing
        directory = self._directory(key)
        path = self._manifest_path(key)
        encoded = _canonical_bytes(manifest.to_record())
        temporary = directory / f".{path.name}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.load(key)
                if existing != manifest:
                    raise TeacherCacheError(
                        "concurrent teacher cache publication disagreed"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)
        return manifest


@dataclass(frozen=True, slots=True)
class RepairSelectionCandidate:
    sample_id: str
    content_digest: str
    domain: str
    token_count: int
    utility: float
    diversity: float
    predicted_recoverability: float
    partition: str = "train"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("repair sample ID", self.sample_id),
            ("repair sample domain", self.domain),
            ("repair sample partition", self.partition),
        ):
            _text(value, label)
        _digest(self.content_digest, "repair sample content digest")
        if isinstance(self.token_count, bool) or self.token_count <= 0:
            raise TeacherCacheError("repair sample token count must be positive")
        for numeric_label, numeric_value in (
            ("repair sample utility", self.utility),
            ("repair sample diversity", self.diversity),
            ("predicted recoverability", self.predicted_recoverability),
        ):
            _finite(numeric_value, numeric_label)
        if self.diversity < 0:
            raise TeacherCacheError("repair sample diversity cannot be negative")
        try:
            _canonical(dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise TeacherCacheError("repair sample metadata must be canonical JSON") from error

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "content_digest": self.content_digest,
            "domain": self.domain,
            "token_count": self.token_count,
            "utility": self.utility,
            "diversity": self.diversity,
            "predicted_recoverability": self.predicted_recoverability,
            "partition": self.partition,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RepairSelectionConfig:
    max_examples: int
    max_tokens: int
    seed: int = 0
    strategy: RepairSelectionStrategy = RepairSelectionStrategy.RANDOM
    forbidden_partitions: tuple[str, ...] = ("test", "validation")

    def __post_init__(self) -> None:
        if self.max_examples <= 0 or self.max_tokens <= 0:
            raise TeacherCacheError("repair selection limits must be positive")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise TeacherCacheError("repair selection seed must be an unsigned 64-bit integer")
        if self.forbidden_partitions != tuple(sorted(set(self.forbidden_partitions))):
            raise TeacherCacheError("forbidden partitions must be unique and canonical")
        if any(not item.strip() for item in self.forbidden_partitions):
            raise TeacherCacheError("forbidden partitions cannot be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "algorithm": REPAIR_SELECTION_ALGORITHM,
            "max_examples": self.max_examples,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "strategy": self.strategy.value,
            "forbidden_partitions": list(self.forbidden_partitions),
        }


@dataclass(frozen=True, slots=True)
class SelectionExclusion:
    sample_id: str
    reason: SelectionExclusionReason

    def to_record(self) -> dict[str, str]:
        return {"sample_id": self.sample_id, "reason": self.reason.value}


@dataclass(frozen=True, slots=True)
class RepairSelectionReport:
    config: RepairSelectionConfig
    selected: tuple[RepairSelectionCandidate, ...]
    exclusions: tuple[SelectionExclusion, ...]
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count != sum(item.token_count for item in self.selected):
            raise TeacherCacheError("repair selection token count does not reconcile")
        if self.token_count > self.config.max_tokens:
            raise TeacherCacheError("repair selection exceeds token budget")

    @property
    def selection_id(self) -> str:
        return "repair_selection_" + hashlib.sha256(
            _canonical_bytes(self._identity_record())
        ).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "algorithm": REPAIR_SELECTION_ALGORITHM,
            "config": self.config.to_record(),
            "selected": [item.to_record() for item in self.selected],
            "exclusions": [item.to_record() for item in self.exclusions],
            "token_count": self.token_count,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["selection_id"] = self.selection_id
        return record


def _selection_rank(candidate: RepairSelectionCandidate, config: RepairSelectionConfig) -> bytes:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": REPAIR_SELECTION_ALGORITHM,
                "seed": config.seed,
                "sample_id": candidate.sample_id,
                "content_digest": candidate.content_digest,
            }
        )
    ).digest()


def _ordered_candidates(
    candidates: tuple[RepairSelectionCandidate, ...], config: RepairSelectionConfig
) -> tuple[RepairSelectionCandidate, ...]:
    ranked = sorted(candidates, key=lambda item: (_selection_rank(item, config), item.sample_id))
    if config.strategy is RepairSelectionStrategy.RANDOM:
        return tuple(ranked)
    if config.strategy is RepairSelectionStrategy.PREDICTED_RECOVERABILITY:
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-item.predicted_recoverability, _selection_rank(item, config)),
            )
        )
    if config.strategy is RepairSelectionStrategy.DIVERSITY:
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.diversity,
                    -item.utility,
                    _selection_rank(item, config),
                ),
            )
        )
    by_domain: dict[str, list[RepairSelectionCandidate]] = {}
    for item in ranked:
        by_domain.setdefault(item.domain, []).append(item)
    ordered: list[RepairSelectionCandidate] = []
    while by_domain:
        for domain in sorted(tuple(by_domain)):
            values = by_domain[domain]
            ordered.append(values.pop(0))
            if not values:
                del by_domain[domain]
    return tuple(ordered)


def select_repair_examples(
    candidates: tuple[RepairSelectionCandidate, ...],
    config: RepairSelectionConfig,
) -> RepairSelectionReport:
    """Select bounded repair examples while excluding validation and test candidates."""

    sample_ids = tuple(item.sample_id for item in candidates)
    if len(sample_ids) != len(set(sample_ids)):
        raise TeacherCacheError("repair selection sample IDs must be unique")
    exclusions: list[SelectionExclusion] = []
    eligible: list[RepairSelectionCandidate] = []
    for candidate in candidates:
        if candidate.partition in config.forbidden_partitions:
            exclusions.append(
                SelectionExclusion(candidate.sample_id, SelectionExclusionReason.HELDOUT)
            )
        elif candidate.partition != "train":
            exclusions.append(
                SelectionExclusion(
                    candidate.sample_id,
                    SelectionExclusionReason.NON_TRAINING_PARTITION,
                )
            )
        else:
            eligible.append(candidate)
    selected: list[RepairSelectionCandidate] = []
    tokens = 0
    for candidate in _ordered_candidates(tuple(eligible), config):
        if len(selected) >= config.max_examples:
            exclusions.append(
                SelectionExclusion(
                    candidate.sample_id,
                    SelectionExclusionReason.EXAMPLE_BUDGET,
                )
            )
        elif tokens + candidate.token_count > config.max_tokens:
            exclusions.append(
                SelectionExclusion(
                    candidate.sample_id,
                    SelectionExclusionReason.TOKEN_BUDGET,
                )
            )
        else:
            selected.append(candidate)
            tokens += candidate.token_count
    return RepairSelectionReport(config, tuple(selected), tuple(exclusions), tokens)
