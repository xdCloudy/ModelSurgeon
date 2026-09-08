"""Cumulative, reload-bound Hugging Face physical surgery sequences."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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

HUGGINGFACE_CUMULATIVE_SCHEMA_VERSION = 1


class HuggingFaceCumulativeError(RuntimeError):
    """Raised for malformed sequences or unsafe cumulative execution."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class HuggingFaceArtifactPublisher(Protocol):
    """Atomically save one candidate model to the requested child path."""

    def __call__(self, model: Any, destination: Path) -> Path: ...


class HuggingFaceArtifactReloader(Protocol):
    """Reload a published child into the model representation used by the next step."""

    def __call__(self, artifact: Path) -> Any: ...


class HuggingFaceGenerationSmoke(Protocol):
    """Run a bounded deterministic generation check and return its success."""

    def __call__(self, model: Any) -> bool: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceEdit:
    """One ordered physical edit callback with a stable mutation identity."""

    mutation_id: str
    operation: str
    apply: Callable[[Any], object]

    def __post_init__(self) -> None:
        if not self.mutation_id.strip() or not self.operation.strip():
            raise HuggingFaceCumulativeError("edit identity and operation are required")

    def identity_record(self, index: int) -> dict[str, object]:
        return {"index": index, "mutation_id": self.mutation_id, "operation": self.operation}


@dataclass(frozen=True, slots=True)
class HuggingFaceStageEvidence:
    """Published evidence for one accepted cumulative stage."""

    index: int
    mutation_id: str
    operation: str
    artifact: Path
    outcome: PhysicalArtifactOutcome
    reloadable: bool
    generation_smoke: bool
    evaluation: Mapping[str, object] | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "mutation_id": self.mutation_id,
            "operation": self.operation,
            "artifact": str(self.artifact),
            "reloadable": self.reloadable,
            "generation_smoke": self.generation_smoke,
            "evaluation": None if self.evaluation is None else dict(self.evaluation),
            "outcome": self.outcome.to_record(),
        }


@dataclass(frozen=True, slots=True)
class HuggingFaceCumulativeRun:
    """Accepted stages, exact reconciliation, and any safely rolled-back failure."""

    sequence_id: str
    stages: tuple[HuggingFaceStageEvidence, ...]
    reconciliation: ReconciliationReport
    failed_index: int | None
    failure_reason: str | None
    final_model: Any
    failed_evaluation: Mapping[str, object] | None = None
    schema_version: int = HUGGINGFACE_CUMULATIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HUGGINGFACE_CUMULATIVE_SCHEMA_VERSION:
            raise HuggingFaceCumulativeError("unknown Hugging Face cumulative schema version")
        if self.failed_index is None and self.failure_reason is not None:
            raise HuggingFaceCumulativeError("failure reason requires a failed stage")
        if self.failed_index is not None and not self.failure_reason:
            raise HuggingFaceCumulativeError("failed stage requires a failure reason")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "stages": [item.to_record() for item in self.stages],
            "reconciliation": self.reconciliation.to_record(),
            "failed_index": self.failed_index,
            "failure_reason": self.failure_reason,
            "failed_evaluation": (
                None if self.failed_evaluation is None else dict(self.failed_evaluation)
            ),
        }


@dataclass(frozen=True, slots=True)
class _ModelSnapshot:
    parameters: int
    storage_bytes: int
    tensor_count: int
    architecture_digest: str


def _snapshot(model: Any) -> _ModelSnapshot:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise HuggingFaceCumulativeError("model must expose state_dict()")
    records: list[dict[str, object]] = []
    parameters = 0
    storage_bytes = 0
    for name, tensor in sorted(state_dict().items()):
        shape = tuple(int(value) for value in tensor.shape)
        numel = int(tensor.numel())
        byte_size = numel * int(tensor.element_size())
        parameters += numel
        storage_bytes += byte_size
        records.append({"name": name, "shape": list(shape), "dtype": str(tensor.dtype)})
    if not records or parameters <= 0 or storage_bytes <= 0:
        raise HuggingFaceCumulativeError("model state must contain non-empty tensors")
    digest = hashlib.sha256(_canonical(records).encode()).hexdigest()
    return _ModelSnapshot(parameters, storage_bytes, len(records), digest)


def _file_identity(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise HuggingFaceCumulativeError("publisher did not return a regular artifact file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise HuggingFaceCumulativeError("publisher returned an empty artifact")
    return digest.hexdigest(), size


def _remove_child(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _sequence_id(edits: tuple[HuggingFaceEdit, ...]) -> str:
    identity = [edit.identity_record(index) for index, edit in enumerate(edits)]
    return "hf_sequence_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _stage_outcome(
    *,
    source_outcome_id: str,
    parent_outcome_id: str | None,
    sequence_id: str,
    index: int,
    edit: HuggingFaceEdit,
    before: _ModelSnapshot,
    after: _ModelSnapshot,
    artifact: Path,
    digest: str,
    size_bytes: int,
    cumulative_parameter_delta: int,
    cumulative_storage_delta: int,
    edit_order: tuple[str, ...],
) -> PhysicalArtifactOutcome:
    component_digest = hashlib.sha256(
        _canonical(edit.identity_record(index)).encode()
    ).hexdigest()
    tensor = TensorOutcome(
        f"sequence.{index}.model_state",
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
            before.parameters,
            after.parameters,
            after.tensor_count,
            after.architecture_digest,
        ),
        (tensor,),
        FormatArtifactEvidence(
            ArtifactFormat.HUGGINGFACE,
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
                "artifact": str(artifact),
            },
        ),
        DeploymentEvidence("huggingface-reload-generation-smoke", None, None, None, None),
        after.parameters - before.parameters,
        after.storage_bytes - before.storage_bytes,
        cumulative_parameter_delta,
        cumulative_storage_delta,
        {
            "sequence_id": sequence_id,
            "edit_order": list(edit_order),
            "operation": edit.operation,
        },
    )


def run_huggingface_cumulative_sequence(
    source_model: Any,
    edits: tuple[HuggingFaceEdit, ...],
    *,
    output_root: str | Path,
    source_outcome_id: str,
    publish: HuggingFaceArtifactPublisher,
    reload: HuggingFaceArtifactReloader,
    generate: HuggingFaceGenerationSmoke,
    evaluate: Callable[[Any], Mapping[str, object]] | None = None,
) -> HuggingFaceCumulativeRun:
    """Apply mixed edits with save/reload/generate gates after every accepted stage.

    The publisher is responsible for using the atomic safetensors checkpoint
    writer. A failed edit only ever touches its uniquely named child path; the
    last reloaded model and all accepted children remain the committed state.
    """

    if not edits:
        raise HuggingFaceCumulativeError("cumulative sequence requires at least one edit")
    if not source_outcome_id.strip():
        raise HuggingFaceCumulativeError("source outcome identity is required")
    initial = _snapshot(source_model)
    sequence_id = _sequence_id(edits)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    accepted_model = source_model
    stages: list[HuggingFaceStageEvidence] = []
    parent_id: str | None = None
    failed_index: int | None = None
    failure_reason: str | None = None
    failed_evaluation: Mapping[str, object] | None = None
    cumulative_parameter_delta = 0
    cumulative_storage_delta = 0
    edit_order: list[str] = []

    for index, edit in enumerate(edits):
        candidate = copy.deepcopy(accepted_model)
        before = _snapshot(accepted_model)
        child = root / f"{sequence_id}.stage-{index:02d}-{edit.mutation_id}"
        try:
            edit.apply(candidate)
            published = publish(candidate, child)
            digest, size_bytes = _file_identity(published)
            reloaded = reload(published)
            if generate(reloaded) is not True:
                raise HuggingFaceCumulativeError("generation smoke did not return true")
            evaluation: Mapping[str, object] | None = None
            if evaluate is not None:
                evaluation = evaluate(reloaded)
                if not isinstance(evaluation, Mapping):
                    raise HuggingFaceCumulativeError(
                        "cumulative evaluator must return a mapping"
                    )
                if evaluation.get("accepted") is not True:
                    failed_evaluation = dict(evaluation)
                    raise HuggingFaceCumulativeError(
                        "cumulative physical evaluation rejected the child"
                    )
            after = _snapshot(reloaded)
            cumulative_parameter_delta += after.parameters - before.parameters
            cumulative_storage_delta += after.storage_bytes - before.storage_bytes
            edit_order.append(edit.mutation_id)
            outcome = _stage_outcome(
                source_outcome_id=source_outcome_id,
                parent_outcome_id=parent_id,
                sequence_id=sequence_id,
                index=index,
                edit=edit,
                before=before,
                after=after,
                artifact=published,
                digest=digest,
                size_bytes=size_bytes,
                cumulative_parameter_delta=cumulative_parameter_delta,
                cumulative_storage_delta=cumulative_storage_delta,
                edit_order=tuple(edit_order),
            )
        except Exception as error:
            _remove_child(child)
            failed_index = index
            failure_reason = str(error) or error.__class__.__name__
            break
        stages.append(
            HuggingFaceStageEvidence(
                index,
                edit.mutation_id,
                edit.operation,
                published,
                outcome,
                True,
                True,
                evaluation,
            )
        )
        accepted_model = reloaded
        parent_id = outcome.outcome_id

    if not stages:
        raise HuggingFaceCumulativeError(f"first cumulative edit failed: {failure_reason}")
    sequence = PhysicalOutcomeSequence(
        initial.parameters,
        initial.storage_bytes,
        tuple(item.outcome for item in stages),
    )
    return HuggingFaceCumulativeRun(
        sequence_id,
        tuple(stages),
        sequence.reconcile(),
        failed_index,
        failure_reason,
        accepted_model,
        failed_evaluation,
    )


def render_huggingface_sequence_report(run: HuggingFaceCumulativeRun) -> str:
    """Render a stable JSON report without serializing the live model object."""

    return json.dumps(run.to_record(), sort_keys=True, separators=(",", ":")) + "\n"
