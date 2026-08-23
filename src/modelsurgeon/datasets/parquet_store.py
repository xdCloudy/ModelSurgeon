"""Partitioned Parquet feature/example storage with atomic visibility manifests."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import MutationExampleRecord
from modelsurgeon.features.schema import FeatureRecord

PARQUET_MANIFEST_SCHEMA_VERSION = 1


class ParquetStoreError(RuntimeError):
    """Raised when a partition or Parquet visibility contract is invalid."""


class ParquetDependencyError(ParquetStoreError):
    """Raised when the optional PyArrow Parquet backend is requested but unavailable."""


class PartitionKind(StrEnum):
    FEATURES = "features"
    EXAMPLES = "examples"


@dataclass(frozen=True, slots=True)
class ParquetPartition:
    kind: PartitionKind
    schema_version: int
    model_identifier: str
    model_revision: str
    campaign_id: str

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ParquetStoreError("partition schema version must be positive")
        if any(
            not value for value in (self.model_identifier, self.model_revision, self.campaign_id)
        ):
            raise ParquetStoreError("partition model and campaign identities are required")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "campaign_id": self.campaign_id,
        }


@dataclass(frozen=True, slots=True)
class PartitionPredicate:
    kind: PartitionKind | None = None
    schema_version: int | None = None
    model_identifier: str | None = None
    model_revision: str | None = None
    campaign_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version is not None and self.schema_version <= 0:
            raise ParquetStoreError("predicate schema version must be positive")
        for value in (self.model_identifier, self.model_revision, self.campaign_id):
            if value is not None and not value:
                raise ParquetStoreError("partition predicate strings cannot be blank")

    def matches(self, partition: ParquetPartition) -> bool:
        return all(
            (
                self.kind is None or partition.kind is self.kind,
                self.schema_version is None or partition.schema_version == self.schema_version,
                self.model_identifier is None
                or partition.model_identifier == self.model_identifier,
                self.model_revision is None or partition.model_revision == self.model_revision,
                self.campaign_id is None or partition.campaign_id == self.campaign_id,
            )
        )


@dataclass(frozen=True, slots=True)
class ParquetManifestEntry:
    partition: ParquetPartition
    relative_path: str
    row_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ParquetStoreError("visible Parquet partitions must contain rows")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ParquetStoreError("Parquet partition digest must be lowercase SHA-256")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
            raise ParquetStoreError("Parquet manifest paths must be relative .parquet paths")

    def to_record(self) -> dict[str, object]:
        return {
            "partition": self.partition.to_record(),
            "relative_path": self.relative_path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ParquetManifest:
    entries: tuple[ParquetManifestEntry, ...]
    schema_version: int = PARQUET_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARQUET_MANIFEST_SCHEMA_VERSION:
            raise ParquetStoreError(
                f"unsupported Parquet manifest schema version {self.schema_version}"
            )
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ParquetStoreError("Parquet manifest paths must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_record() for entry in self.entries],
        }


class ParquetBackend(Protocol):
    def write_rows(self, path: Path, rows: Sequence[Mapping[str, object]]) -> None: ...

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]: ...


class PyArrowParquetBackend:
    """Real Parquet backend loaded lazily so core installs do not import PyArrow."""

    @staticmethod
    def _modules() -> tuple[Any, Any]:
        try:
            arrow = import_module("pyarrow")
            parquet = import_module("pyarrow.parquet")
        except ImportError as error:
            raise ParquetDependencyError(
                "Parquet storage requires the optional 'pyarrow' package"
            ) from error
        return arrow, parquet

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        arrow, parquet = self._modules()
        table = arrow.Table.from_pylist([dict(row) for row in rows])
        parquet.write_table(table, path)

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]:
        _, parquet = self._modules()
        table = parquet.read_table(path)
        rows = table.to_pylist()
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ParquetStoreError("PyArrow returned malformed Parquet rows")
        return tuple(cast(dict[str, object], row) for row in rows)


def _partition_from_record(value: object) -> ParquetPartition:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "schema_version",
        "model_identifier",
        "model_revision",
        "campaign_id",
    }:
        raise ParquetStoreError("manifest partition has missing or unknown fields")
    kind = value["kind"]
    schema_version = value["schema_version"]
    strings = (
        value["model_identifier"],
        value["model_revision"],
        value["campaign_id"],
    )
    if not isinstance(kind, str):
        raise ParquetStoreError("manifest partition kind is invalid")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ParquetStoreError("manifest partition schema version is invalid")
    if not all(isinstance(item, str) for item in strings):
        raise ParquetStoreError("manifest partition identities are invalid")
    try:
        resolved_kind = PartitionKind(kind)
    except ValueError as error:
        raise ParquetStoreError("manifest partition kind is unknown") from error
    return ParquetPartition(
        resolved_kind,
        schema_version,
        cast(str, strings[0]),
        cast(str, strings[1]),
        cast(str, strings[2]),
    )


def _entry_from_record(value: object) -> ParquetManifestEntry:
    if not isinstance(value, dict) or set(value) != {
        "partition",
        "relative_path",
        "row_count",
        "sha256",
    }:
        raise ParquetStoreError("Parquet manifest entry has missing or unknown fields")
    relative_path = value["relative_path"]
    row_count = value["row_count"]
    sha256 = value["sha256"]
    if not isinstance(relative_path, str) or not isinstance(sha256, str):
        raise ParquetStoreError("Parquet manifest entry path or digest is invalid")
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        raise ParquetStoreError("Parquet manifest row count is invalid")
    return ParquetManifestEntry(
        _partition_from_record(value["partition"]),
        relative_path,
        row_count,
        sha256,
    )


def _manifest_from_record(value: object) -> ParquetManifest:
    if not isinstance(value, dict) or set(value) != {"schema_version", "entries"}:
        raise ParquetStoreError("Parquet manifest has missing or unknown fields")
    schema_version = value["schema_version"]
    entries = value["entries"]
    if schema_version != PARQUET_MANIFEST_SCHEMA_VERSION or not isinstance(entries, list):
        raise ParquetStoreError("Parquet manifest schema version or entries are invalid")
    return ParquetManifest(tuple(_entry_from_record(entry) for entry in entries))


def _segment(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _feature_row(record: FeatureRecord) -> dict[str, object]:
    payload = record.to_record()
    return {
        "record_type": "feature",
        "schema_version": record.schema_version,
        "component_id": str(record.component_id),
        "feature_name": record.name,
        "feature_kind": record.kind.value,
        "dtype": record.dtype,
        "extractor": record.extractor,
        "extractor_version": record.extractor_version,
        "payload_json": canonical_identity_json(payload),
    }


def _example_row(record: MutationExampleRecord) -> dict[str, object]:
    return {
        "record_type": "mutation_example",
        "schema_version": record.schema_version,
        "example_id": record.example_id,
        "experiment_id": record.experiment_id,
        "mutation_id": record.mutation_id,
        "outcome": record.outcome.kind.value,
        "dataset_manifest_id": record.dataset.manifest_id,
        "payload_json": canonical_identity_json(record.to_record()),
    }


class PartitionedParquetStore:
    """Publish immutable Parquet files and expose only manifest-committed partitions."""

    def __init__(self, root: str | Path, backend: ParquetBackend | None = None) -> None:
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or PyArrowParquetBackend()
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()

    def load_manifest(self) -> ParquetManifest:
        if not self.manifest_path.exists():
            return ParquetManifest(())
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise ParquetStoreError("Parquet manifest must be a regular file")
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ParquetStoreError("Parquet manifest is unreadable or corrupt") from error
        return _manifest_from_record(value)

    def _relative_directory(self, partition: ParquetPartition) -> PurePosixPath:
        return PurePosixPath(
            f"schema={partition.schema_version}",
            f"model={_segment(partition.model_identifier + '@' + partition.model_revision)}",
            f"campaign={_segment(partition.campaign_id)}",
            f"kind={partition.kind.value}",
        )

    def _publish_manifest(self, manifest: ParquetManifest) -> None:
        temporary = self.root / f".manifest-{uuid.uuid4().hex}.partial"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_identity_json(manifest.to_record()))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _write_partition(
        self,
        partition: ParquetPartition,
        rows: Sequence[Mapping[str, object]],
    ) -> ParquetManifestEntry:
        if not rows:
            raise ParquetStoreError("Parquet partitions require at least one row")
        with self._lock:
            manifest = self.load_manifest()
            relative_directory = self._relative_directory(partition)
            directory = self.root / Path(*relative_directory.parts)
            directory.mkdir(parents=True, exist_ok=True)
            staging = directory / f".part-{uuid.uuid4().hex}.partial"
            try:
                self.backend.write_rows(staging, rows)
                if not staging.is_file() or staging.is_symlink():
                    raise ParquetStoreError(
                        "Parquet backend did not produce a regular staging file"
                    )
                digest = self._file_sha256(staging)
                final_name = f"part-{digest}.parquet"
                final_path = directory / final_name
                if final_path.exists() or final_path.is_symlink():
                    if not final_path.is_file() or self._file_sha256(final_path) != digest:
                        raise ParquetStoreError(
                            "existing Parquet partition conflicts with its digest"
                        )
                else:
                    try:
                        os.link(staging, final_path)
                    except FileExistsError:
                        if not final_path.is_file() or self._file_sha256(final_path) != digest:
                            raise ParquetStoreError(
                                "Parquet partition appeared with conflicting content"
                            ) from None
                    except OSError as error:
                        raise ParquetStoreError(
                            "Parquet partition could not be published without overwrite"
                        ) from error
                relative_path = (relative_directory / final_name).as_posix()
                entry = ParquetManifestEntry(partition, relative_path, len(rows), digest)
                existing = {item.relative_path: item for item in manifest.entries}
                previous = existing.get(relative_path)
                if previous is not None and previous != entry:
                    raise ParquetStoreError(
                        "Parquet manifest identity conflicts with existing entry"
                    )
                existing[relative_path] = entry
                sorted_entries = tuple(
                    sorted(existing.values(), key=lambda item: item.relative_path)
                )
                self._publish_manifest(ParquetManifest(sorted_entries))
                return entry
            finally:
                staging.unlink(missing_ok=True)

    def write_features(
        self,
        partition: ParquetPartition,
        records: Sequence[FeatureRecord],
    ) -> ParquetManifestEntry:
        if partition.kind is not PartitionKind.FEATURES:
            raise ParquetStoreError("feature records require a features partition")
        if any(record.schema_version != partition.schema_version for record in records):
            raise ParquetStoreError("feature schema versions do not match the partition")
        return self._write_partition(partition, tuple(_feature_row(record) for record in records))

    def write_examples(
        self,
        partition: ParquetPartition,
        records: Sequence[MutationExampleRecord],
    ) -> ParquetManifestEntry:
        if partition.kind is not PartitionKind.EXAMPLES:
            raise ParquetStoreError("mutation examples require an examples partition")
        for record in records:
            if record.schema_version != partition.schema_version:
                raise ParquetStoreError("example schema versions do not match the partition")
            if (
                record.model.identifier != partition.model_identifier
                or record.model.revision != partition.model_revision
            ):
                raise ParquetStoreError("example model identity does not match the partition")
        return self._write_partition(partition, tuple(_example_row(record) for record in records))

    def read_rows(
        self,
        predicate: PartitionPredicate | None = None,
    ) -> tuple[dict[str, object], ...]:
        resolved = predicate or PartitionPredicate()
        manifest = self.load_manifest()
        selected = tuple(
            entry for entry in manifest.entries if resolved.matches(entry.partition)
        )
        output: list[dict[str, object]] = []
        for entry in selected:
            path = self.root / Path(*PurePosixPath(entry.relative_path).parts)
            if not path.is_file() or path.is_symlink():
                raise ParquetStoreError("manifest references a missing or unsafe Parquet file")
            if self._file_sha256(path) != entry.sha256:
                raise ParquetStoreError("manifest Parquet digest does not match file content")
            rows = self.backend.read_rows(path)
            if len(rows) != entry.row_count:
                raise ParquetStoreError("manifest Parquet row count does not match file content")
            output.extend(rows)
        return tuple(output)
