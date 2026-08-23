"""Immutable revision-keyed cache for baseline logits, loss, and scalar metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_CACHE_VERSION = "1"


class BaselineCacheError(RuntimeError):
    """Raised when baseline cache identity, integrity, or immutability is violated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class BaselineCacheKey:
    model_revision: str
    dataset_revision: str
    tokenizer_revision: str
    evaluator_version: str

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.model_revision,
                self.dataset_revision,
                self.tokenizer_revision,
                self.evaluator_version,
            )
        ):
            raise BaselineCacheError("baseline cache revisions must be non-empty")

    def to_record(self) -> dict[str, str]:
        return {
            "model_revision": self.model_revision,
            "dataset_revision": self.dataset_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "evaluator_version": self.evaluator_version,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_record())).hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    logits: tuple[tuple[float, ...], ...]
    mean_loss: float
    token_count: int
    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.logits or any(not row for row in self.logits):
            raise BaselineCacheError("baseline logits require non-empty rows")
        widths = {len(row) for row in self.logits}
        if len(widths) != 1:
            raise BaselineCacheError("baseline logit rows must have one vocabulary width")
        values = [value for row in self.logits for value in row]
        values.append(self.mean_loss)
        values.extend(value for _, value in self.metrics)
        if any(not math.isfinite(value) for value in values):
            raise BaselineCacheError("baseline artifacts must contain finite values")
        if self.mean_loss < 0 or self.token_count <= 0:
            raise BaselineCacheError("baseline loss/token count is invalid")
        keys = tuple(key for key, _ in self.metrics)
        if keys != tuple(sorted(set(keys))) or any(not key for key in keys):
            raise BaselineCacheError("baseline metric keys must be non-empty and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "logits": [list(row) for row in self.logits],
            "mean_loss": self.mean_loss,
            "token_count": self.token_count,
            "metrics": dict(self.metrics),
        }


def _artifact_from_record(raw: object) -> BaselineArtifact:
    if not isinstance(raw, dict):
        raise BaselineCacheError("baseline artifact must be an object")
    logits_raw = raw.get("logits")
    metrics_raw = raw.get("metrics", {})
    if not isinstance(logits_raw, list) or not isinstance(metrics_raw, dict):
        raise BaselineCacheError("baseline artifact logits or metrics are malformed")
    try:
        logits = tuple(tuple(float(value) for value in row) for row in logits_raw)
        metrics = tuple(sorted((str(key), float(value)) for key, value in metrics_raw.items()))
        return BaselineArtifact(
            logits,
            float(raw["mean_loss"]),
            int(raw["token_count"]),
            metrics,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineCacheError("baseline artifact fields are invalid") from error


def _key_from_record(raw: object) -> BaselineCacheKey:
    if not isinstance(raw, dict):
        raise BaselineCacheError("baseline cache key must be an object")
    try:
        return BaselineCacheKey(
            str(raw["model_revision"]),
            str(raw["dataset_revision"]),
            str(raw["tokenizer_revision"]),
            str(raw["evaluator_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineCacheError("baseline cache key is invalid") from error


class BaselineCache:
    def __init__(self, root: str | Path, *, max_serialized_bytes: int = 256 * 1024 * 1024) -> None:
        if max_serialized_bytes <= 0:
            raise BaselineCacheError("baseline cache byte budget must be positive")
        self.root = Path(root)
        self.max_serialized_bytes = max_serialized_bytes

    def path_for(self, key: BaselineCacheKey) -> Path:
        return self.root / f"{key.digest}.json"

    def load(self, key: BaselineCacheKey) -> BaselineArtifact | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BaselineCacheError("published baseline cache entry is unreadable") from error
        if not isinstance(raw, dict) or raw.get("version") != BASELINE_CACHE_VERSION:
            raise BaselineCacheError("published baseline cache schema is unsupported")
        embedded = _key_from_record(raw.get("key"))
        if embedded != key:
            raise BaselineCacheError("published baseline cache key does not match lookup key")
        artifact_raw = raw.get("artifact")
        checksum = hashlib.sha256(_canonical_bytes(artifact_raw)).hexdigest()
        if raw.get("artifact_sha256") != checksum:
            raise BaselineCacheError("published baseline artifact checksum mismatch")
        return _artifact_from_record(artifact_raw)

    def write(self, key: BaselineCacheKey, artifact: BaselineArtifact) -> BaselineArtifact:
        artifact_record = artifact.to_record()
        payload = {
            "version": BASELINE_CACHE_VERSION,
            "key": key.to_record(),
            "artifact_sha256": hashlib.sha256(_canonical_bytes(artifact_record)).hexdigest(),
            "artifact": artifact_record,
        }
        encoded = _canonical_bytes(payload)
        if len(encoded) > self.max_serialized_bytes:
            raise BaselineCacheError(
                f"baseline artifact needs {len(encoded)} bytes, "
                f"budget is {self.max_serialized_bytes}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self.path_for(key)
        existing = self.load(key)
        if existing is not None:
            if existing != artifact:
                raise BaselineCacheError("immutable baseline cache key already has different data")
            return existing
        temporary = self.root / f".{key.digest}.{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, final_path)
            except FileExistsError as error:
                existing = self.load(key)
                if existing != artifact:
                    raise BaselineCacheError(
                        "concurrent immutable baseline publication disagreed"
                    ) from error
                return artifact
        finally:
            temporary.unlink(missing_ok=True)
        return artifact

    def get_or_compute(
        self,
        key: BaselineCacheKey,
        compute: Callable[[], BaselineArtifact],
    ) -> BaselineArtifact:
        cached = self.load(key)
        if cached is not None:
            return cached
        return self.write(key, compute())
