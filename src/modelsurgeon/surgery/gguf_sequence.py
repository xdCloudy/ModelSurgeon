"""Cumulative, bounded native GGUF surgery sequences."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from modelsurgeon.surgery.artifact_outcome import (
    ArchitectureState,
    ArtifactFormat,
    ArtifactState,
    DeploymentEvidence,
    FormatArtifactEvidence,
    PhysicalArtifactOutcome,
    PhysicalOutcomeSequence,
    ReconciliationReport,
    TensorOutcome,
)

GGUF_CUMULATIVE_SCHEMA_VERSION = 1


class GGUFCumulativeError(RuntimeError):
    """Raised for malformed GGUF lineage or an unsafe cumulative child."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_file(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise GGUFCumulativeError("native GGUF executor did not return a regular output file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise GGUFCumulativeError("native GGUF executor returned an empty output")
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class GGUFResourceLimits:
    """Consumer limits applied independently to every cumulative stage."""

    max_working_memory_bytes: int
    max_scratch_disk_bytes: int

    def __post_init__(self) -> None:
        if self.max_working_memory_bytes <= 0 or self.max_scratch_disk_bytes <= 0:
            raise GGUFCumulativeError("GGUF resource limits must be positive")


@dataclass(frozen=True, slots=True)
class GGUFSourceState:
    """Reconciled metadata and encoded payload hashes for the current child."""

    parameter_count: int
    storage_bytes: int
    architecture_digest: str
    tensor_payload_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.parameter_count <= 0 or self.storage_bytes <= 0:
            raise GGUFCumulativeError("GGUF source state counts must be positive")
        names = tuple(name for name, _ in self.tensor_payload_sha256)
        if not names or names != tuple(sorted(set(names))):
            raise GGUFCumulativeError(
                "GGUF payload hashes must be uniquely and canonically ordered"
            )
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for _, digest in self.tensor_payload_sha256
        ):
            raise GGUFCumulativeError("GGUF payload hashes must be lowercase SHA-256 values")


@dataclass(frozen=True, slots=True)
class GGUFStageMeasurement:
    """Executor result required before a GGUF child can enter lineage."""

    parameter_count: int
    storage_bytes: int
    architecture_digest: str
    tensor_payload_sha256: tuple[tuple[str, str], ...]
    changed_tensor_names: tuple[str, ...]
    peak_working_memory_bytes: int
    scratch_disk_bytes: int
    reloadable: bool
    generation_smoke: bool

    def __post_init__(self) -> None:
        GGUFSourceState(
            self.parameter_count,
            self.storage_bytes,
            self.architecture_digest,
            self.tensor_payload_sha256,
        )
        if tuple(self.changed_tensor_names) != tuple(sorted(set(self.changed_tensor_names))):
            raise GGUFCumulativeError("changed GGUF tensor names must be canonical")
        if self.peak_working_memory_bytes < 0 or self.scratch_disk_bytes < 0:
            raise GGUFCumulativeError("GGUF resource measurements cannot be negative")
        if not self.reloadable or not self.generation_smoke:
            raise GGUFCumulativeError("GGUF child requires reload and generation evidence")


class GGUFEditExecutor(Protocol):
    """Execute one native edit from a live source child to a new child."""

    def __call__(self, source: Path, destination: Path) -> GGUFStageMeasurement: ...


@dataclass(frozen=True, slots=True)
class GGUFEdit:
    """Stable identity for one native executor and its edit order."""

    mutation_id: str
    operation: str
    execute: GGUFEditExecutor

    def __post_init__(self) -> None:
        if not self.mutation_id.strip() or not self.operation.strip():
            raise GGUFCumulativeError("GGUF edit identity and operation are required")

    def identity_record(self, index: int) -> dict[str, object]:
        return {"index": index, "mutation_id": self.mutation_id, "operation": self.operation}


@dataclass(frozen=True, slots=True)
class GGUFStageEvidence:
    index: int
    mutation_id: str
    operation: str
    artifact: Path
    outcome: PhysicalArtifactOutcome
    unchanged_tensor_names: tuple[str, ...]
    peak_working_memory_bytes: int
    scratch_disk_bytes: int

    def to_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "mutation_id": self.mutation_id,
            "operation": self.operation,
            "artifact": str(self.artifact),
            "unchanged_tensor_names": list(self.unchanged_tensor_names),
            "peak_working_memory_bytes": self.peak_working_memory_bytes,
            "scratch_disk_bytes": self.scratch_disk_bytes,
            "outcome": self.outcome.to_record(),
        }


@dataclass(frozen=True, slots=True)
class GGUFCumulativeRun:
    sequence_id: str
    stages: tuple[GGUFStageEvidence, ...]
    reconciliation: ReconciliationReport
    final_artifact: Path
    failed_index: int | None
    failure_reason: str | None
    schema_version: int = GGUF_CUMULATIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GGUF_CUMULATIVE_SCHEMA_VERSION:
            raise GGUFCumulativeError("unknown GGUF cumulative schema version")
        if self.failed_index is None and self.failure_reason is not None:
            raise GGUFCumulativeError("failure reason requires a failed stage")
        if self.failed_index is not None and not self.failure_reason:
            raise GGUFCumulativeError("failed stage requires a failure reason")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "stages": [stage.to_record() for stage in self.stages],
            "reconciliation": self.reconciliation.to_record(),
            "final_artifact": str(self.final_artifact),
            "failed_index": self.failed_index,
            "failure_reason": self.failure_reason,
        }


def _sequence_id(edits: tuple[GGUFEdit, ...]) -> str:
    identity = [edit.identity_record(index) for index, edit in enumerate(edits)]
    return "gguf_sequence_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _remove_child(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _outcome(
    source_outcome_id: str,
    parent_outcome_id: str | None,
    sequence_id: str,
    index: int,
    edit: GGUFEdit,
    before: GGUFSourceState,
    after: GGUFStageMeasurement,
    artifact: Path,
    digest: str,
    size_bytes: int,
    cumulative_parameter_delta: int,
    cumulative_storage_delta: int,
    edit_order: tuple[str, ...],
) -> PhysicalArtifactOutcome:
    component_digest = hashlib.sha256(_canonical(edit.identity_record(index)).encode()).hexdigest()
    tensor = TensorOutcome(
        f"sequence.{index}.gguf_payload",
        (before.storage_bytes,),
        (after.storage_bytes,),
        before.storage_bytes,
        after.storage_bytes,
        component_digest,
    )
    return PhysicalArtifactOutcome(
        source_outcome_id,
        parent_outcome_id,
        edit.mutation_id,
        ArchitectureState(
            before.parameter_count,
            after.parameter_count,
            len(after.tensor_payload_sha256),
            after.architecture_digest,
        ),
        (tensor,),
        FormatArtifactEvidence(
            ArtifactFormat.GGUF,
            ArtifactState.COMPLETE,
            digest,
            size_bytes,
            1,
            True,
            True,
            {
                "sequence_id": sequence_id,
                "stage_index": index,
                "operation": edit.operation,
                "edit_order": list(edit_order),
                "peak_working_memory_bytes": after.peak_working_memory_bytes,
                "scratch_disk_bytes": after.scratch_disk_bytes,
            },
        ),
        DeploymentEvidence("llama.cpp-reload-generation-smoke", None, None, None, None),
        after.parameter_count - before.parameter_count,
        after.storage_bytes - before.storage_bytes,
        cumulative_parameter_delta,
        cumulative_storage_delta,
        {
            "sequence_id": sequence_id,
            "edit_order": list(edit_order),
            "changed_tensor_names": list(after.changed_tensor_names),
        },
    )


def run_gguf_cumulative_sequence(
    source_artifact: str | Path,
    source_state: GGUFSourceState,
    edits: tuple[GGUFEdit, ...],
    *,
    output_root: str | Path,
    source_outcome_id: str,
    limits: GGUFResourceLimits,
) -> GGUFCumulativeRun:
    """Execute mixed native edits with reopen, payload, and resource gates."""

    source = Path(source_artifact)
    if not source.is_file() or not edits or not source_outcome_id.strip():
        raise GGUFCumulativeError("GGUF source, edit sequence, and source outcome are required")
    sequence_id = _sequence_id(edits)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    current_path = source
    current = source_state
    stages: list[GGUFStageEvidence] = []
    parent_id: str | None = None
    edit_order: list[str] = []
    cumulative_parameter_delta = 0
    cumulative_storage_delta = 0
    failed_index: int | None = None
    failure_reason: str | None = None

    for index, edit in enumerate(edits):
        child = root / f"{sequence_id}.stage-{index:02d}-{edit.mutation_id}.gguf"
        try:
            measurement = edit.execute(current_path, child)
            if measurement.peak_working_memory_bytes > limits.max_working_memory_bytes:
                raise GGUFCumulativeError("GGUF working-memory limit exceeded")
            if measurement.scratch_disk_bytes > limits.max_scratch_disk_bytes:
                raise GGUFCumulativeError("GGUF scratch-disk limit exceeded")
            previous_payloads = dict(current.tensor_payload_sha256)
            next_payloads = dict(measurement.tensor_payload_sha256)
            changed = set(measurement.changed_tensor_names)
            untouched = tuple(sorted(set(previous_payloads) - changed))
            if any(next_payloads.get(name) != previous_payloads[name] for name in untouched):
                raise GGUFCumulativeError("an untouched GGUF tensor payload changed")
            if set(next_payloads) - set(previous_payloads) - changed:
                raise GGUFCumulativeError("new GGUF tensor payload was not declared changed")
            digest, size_bytes = _digest_file(child)
            edit_order.append(edit.mutation_id)
            cumulative_parameter_delta += measurement.parameter_count - current.parameter_count
            cumulative_storage_delta += measurement.storage_bytes - current.storage_bytes
            outcome = _outcome(
                source_outcome_id,
                parent_id,
                sequence_id,
                index,
                edit,
                current,
                measurement,
                child,
                digest,
                size_bytes,
                cumulative_parameter_delta,
                cumulative_storage_delta,
                tuple(edit_order),
            )
        except Exception as error:
            _remove_child(child)
            failed_index = index
            failure_reason = str(error) or error.__class__.__name__
            break
        stages.append(
            GGUFStageEvidence(
                index,
                edit.mutation_id,
                edit.operation,
                child,
                outcome,
                untouched,
                measurement.peak_working_memory_bytes,
                measurement.scratch_disk_bytes,
            )
        )
        current_path = child
        current = GGUFSourceState(
            measurement.parameter_count,
            measurement.storage_bytes,
            measurement.architecture_digest,
            measurement.tensor_payload_sha256,
        )
        parent_id = outcome.outcome_id

    if not stages:
        raise GGUFCumulativeError(f"first cumulative edit failed: {failure_reason}")
    sequence = PhysicalOutcomeSequence(
        source_state.parameter_count,
        source_state.storage_bytes,
        tuple(stage.outcome for stage in stages),
    )
    return GGUFCumulativeRun(
        sequence_id,
        tuple(stages),
        sequence.reconcile(),
        current_path,
        failed_index,
        failure_reason,
    )


def render_gguf_sequence_report(run: GGUFCumulativeRun) -> str:
    """Render a stable JSON report for retained native evidence."""

    return json.dumps(run.to_record(), sort_keys=True, separators=(",", ":")) + "\n"
