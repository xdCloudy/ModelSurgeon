"""Immutable evidence records emitted by first-party optimization runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from modelsurgeon.experiments.identity import canonical_identity_json

OPTIMIZATION_EVIDENCE_SCHEMA_VERSION = 1


class OptimizationEvidenceError(RuntimeError):
    """Raised when optimization evidence cannot be published safely."""


class OptimizationEvidenceOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OptimizationEvidenceError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _mapping(value, label)


@dataclass(frozen=True, slots=True)
class OptimizationEvidenceRecord:
    """One real candidate or physical-child observation for future learning."""

    observation_id: str
    run_id: str
    stage: str
    state_id: str
    outcome: OptimizationEvidenceOutcome
    model: Mapping[str, object]
    dataset: Mapping[str, object]
    hardware: Mapping[str, object]
    versions: Mapping[str, object]
    lineage: Mapping[str, object]
    candidate_id: str | None = None
    mutation_id: str | None = None
    features: tuple[Mapping[str, object], ...] = ()
    prediction: Mapping[str, object] | None = None
    measurement: Mapping[str, object] | None = None
    artifact: Mapping[str, object] | None = None
    failure: Mapping[str, object] | None = None
    schema_version: int = OPTIMIZATION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = (
            self.observation_id,
            self.run_id,
            self.stage,
            self.state_id,
        )
        if any(not value.strip() for value in required):
            raise OptimizationEvidenceError("evidence identity fields are required")
        if self.schema_version != OPTIMIZATION_EVIDENCE_SCHEMA_VERSION:
            raise OptimizationEvidenceError("unsupported optimization evidence schema")
        for label, value in (
            ("model", self.model),
            ("dataset", self.dataset),
            ("hardware", self.hardware),
            ("versions", self.versions),
            ("lineage", self.lineage),
        ):
            _mapping(value, label)
        for index, feature in enumerate(self.features):
            _mapping(feature, f"features[{index}]")
        _optional_mapping(self.prediction, "prediction")
        _optional_mapping(self.measurement, "measurement")
        _optional_mapping(self.artifact, "artifact")
        _optional_mapping(self.failure, "failure")
        if self.outcome in {
            OptimizationEvidenceOutcome.ACCEPTED,
            OptimizationEvidenceOutcome.REJECTED,
        } and self.measurement is None:
            raise OptimizationEvidenceError(
                "accepted or rejected evidence requires a real measurement"
            )
        if self.outcome is OptimizationEvidenceOutcome.ROLLED_BACK and self.failure is None:
            raise OptimizationEvidenceError("rolled-back evidence requires failure detail")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "state_id": self.state_id,
            "outcome": self.outcome.value,
            "candidate_id": self.candidate_id,
            "mutation_id": self.mutation_id,
            "model": dict(self.model),
            "dataset": dict(self.dataset),
            "hardware": dict(self.hardware),
            "versions": dict(self.versions),
            "lineage": dict(self.lineage),
            "features": [dict(item) for item in self.features],
            "prediction": None if self.prediction is None else dict(self.prediction),
            "measurement": None if self.measurement is None else dict(self.measurement),
            "artifact": None if self.artifact is None else dict(self.artifact),
            "failure": None if self.failure is None else dict(self.failure),
        }


@dataclass(frozen=True, slots=True)
class PublishedOptimizationEvidence:
    observation_id: str
    path: Path
    digest: str

    def to_record(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "path": str(self.path),
            "sha256": self.digest,
        }


class OptimizationEvidenceStore:
    """Publish immutable, content-verified optimization observations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.observation_root = self.root / "observations"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()

    @staticmethod
    def _encoded(record: OptimizationEvidenceRecord) -> bytes:
        return (
            canonical_identity_json(record.to_record()) + "\n"
        ).encode("utf-8")

    @staticmethod
    def _digest(encoded: bytes) -> str:
        return hashlib.sha256(encoded).hexdigest()

    def _manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise OptimizationEvidenceError("optimization evidence manifest is unsafe")
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OptimizationEvidenceError("optimization evidence manifest is corrupt") from error
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise OptimizationEvidenceError("optimization evidence manifest schema is invalid")
        entries = raw.get("entries")
        if not isinstance(entries, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in entries.items()
        ):
            raise OptimizationEvidenceError("optimization evidence manifest entries are invalid")
        return cast(dict[str, str], entries)

    def _write_manifest(self, entries: Mapping[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".manifest-{uuid.uuid4().hex}.partial"
        payload = {"schema_version": 1, "entries": dict(sorted(entries.items()))}
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_identity_json(payload))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, record: OptimizationEvidenceRecord) -> PublishedOptimizationEvidence:
        encoded = self._encoded(record)
        digest = self._digest(encoded)
        with self._lock:
            self.observation_root.mkdir(parents=True, exist_ok=True)
            path = self.observation_root / f"{record.observation_id}.json"
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                    raise OptimizationEvidenceError(
                        "optimization evidence identity conflicts with existing content"
                    )
            else:
                temporary = self.observation_root / (
                    f".{record.observation_id}.{uuid.uuid4().hex}.partial"
                )
                try:
                    with temporary.open("xb") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                            raise OptimizationEvidenceError(
                                "optimization evidence appeared with conflicting content"
                            ) from None
                finally:
                    temporary.unlink(missing_ok=True)
            entries = self._manifest()
            previous = entries.get(record.observation_id)
            if previous is not None and previous != digest:
                raise OptimizationEvidenceError(
                    "optimization evidence manifest identity conflicts with content"
                )
            entries[record.observation_id] = digest
            self._write_manifest(entries)
        return PublishedOptimizationEvidence(record.observation_id, path, digest)


__all__ = [
    "OPTIMIZATION_EVIDENCE_SCHEMA_VERSION",
    "OptimizationEvidenceError",
    "OptimizationEvidenceOutcome",
    "OptimizationEvidenceRecord",
    "OptimizationEvidenceStore",
    "PublishedOptimizationEvidence",
]
