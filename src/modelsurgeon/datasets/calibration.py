"""Versioned calibration dataset, sample identity, and selection contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

CALIBRATION_SCHEMA_VERSION: Literal[1] = 1
SELECTION_ALGORITHM = "sha256-rank-v1"
type MetadataPrimitive = str | int | float | bool | None


class DatasetTrust(StrEnum):
    TRUSTED = "trusted"
    COMMUNITY = "community"
    UNTRUSTED = "untrusted"


def _metadata(value: tuple[tuple[str, MetadataPrimitive], ...]) -> None:
    keys = [key for key, _ in value]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("metadata keys must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset: str
    revision: str
    split: str
    license: str | None
    trust: DatasetTrust
    trust_reason: str
    metadata: tuple[tuple[str, MetadataPrimitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset or not self.revision or not self.split or not self.trust_reason:
            raise ValueError("dataset, revision, split, and trust reason are required")
        if self.license is not None and not self.license:
            raise ValueError("dataset license cannot be blank")
        _metadata(self.metadata)

    def to_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "trust": self.trust.value,
            "trust_reason": self.trust_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingIdentity:
    name: str
    version: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("preprocessing name and version are required")
        _sha256(self.configuration_sha256, "preprocessing configuration")

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "configuration_sha256": self.configuration_sha256,
        }


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    tokenizer: str
    revision: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not self.tokenizer or not self.revision:
            raise ValueError("tokenizer and revision are required")
        _sha256(self.configuration_sha256, "tokenizer configuration")

    def to_record(self) -> dict[str, str]:
        return {
            "tokenizer": self.tokenizer,
            "revision": self.revision,
            "configuration_sha256": self.configuration_sha256,
        }


def _sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    sample_id: str
    content_sha256: str
    metadata: tuple[tuple[str, MetadataPrimitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("calibration sample ID is required")
        _sha256(self.content_sha256, "sample content")
        _metadata(self.metadata)

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "content_sha256": self.content_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    seed: int
    sample_count: int
    algorithm: str = SELECTION_ALGORITHM

    def __post_init__(self) -> None:
        if self.seed < 0 or self.sample_count <= 0:
            raise ValueError("selection seed must be non-negative and count positive")
        if self.algorithm != SELECTION_ALGORITHM:
            raise ValueError(f"unsupported selection algorithm {self.algorithm!r}")


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    dataset: DatasetIdentity
    preprocessing: PreprocessingIdentity
    tokenizer: TokenizerIdentity
    selection: SelectionConfig
    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION

    def select(self, candidates: tuple[CalibrationSample, ...]) -> tuple[CalibrationSample, ...]:
        if len({sample.sample_id for sample in candidates}) != len(candidates):
            raise ValueError("candidate sample IDs must be unique")
        if self.selection.sample_count > len(candidates):
            raise ValueError("selection count exceeds available candidate samples")
        identity = json.dumps(
            {
                "dataset": self.dataset.to_record(),
                "preprocessing": self.preprocessing.to_record(),
                "tokenizer": self.tokenizer.to_record(),
                "seed": self.selection.seed,
                "algorithm": self.selection.algorithm,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        ranked = sorted(
            candidates,
            key=lambda sample: (
                hashlib.sha256(f"{identity}\0{sample.sample_id}".encode()).digest(),
                sample.sample_id.encode(),
            ),
        )
        return tuple(ranked[: self.selection.sample_count])

    def to_record(self, selected: tuple[CalibrationSample, ...]) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset.to_record(),
            "preprocessing": self.preprocessing.to_record(),
            "tokenizer": self.tokenizer.to_record(),
            "selection": {
                "seed": self.selection.seed,
                "sample_count": self.selection.sample_count,
                "algorithm": self.selection.algorithm,
            },
            "samples": [sample.to_record() for sample in selected],
        }
