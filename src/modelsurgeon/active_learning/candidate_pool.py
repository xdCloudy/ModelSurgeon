"""Bounded resumable JSONL pools of canonical graph-valid mutation candidates."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from modelsurgeon.experiments.candidates import (
    CANDIDATE_ENUMERATOR_VERSION,
    MutationCandidate,
)
from modelsurgeon.experiments.identity import canonical_identity_json

CANDIDATE_POOL_SCHEMA_VERSION: Final[int] = 1
MAX_CANDIDATE_POOL_SIZE: Final[int] = 100_000


class CandidatePoolError(ValueError):
    """Raised when pool publication or resume state is incompatible."""


@dataclass(frozen=True, slots=True)
class CandidatePoolProvenance:
    run_id: str
    graph_digest: str
    model_revision: str
    tool_revision: str
    seed: int
    enumerator_version: str = CANDIDATE_ENUMERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.startswith("run_"):
            raise CandidatePoolError("candidate pool requires a canonical run ID")
        if any(
            not value
            for value in (
                self.graph_digest,
                self.model_revision,
                self.tool_revision,
                self.enumerator_version,
            )
        ):
            raise CandidatePoolError("candidate pool provenance fields cannot be blank")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise CandidatePoolError("candidate pool seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "graph_digest": self.graph_digest,
            "model_revision": self.model_revision,
            "tool_revision": self.tool_revision,
            "seed": self.seed,
            "enumerator_version": self.enumerator_version,
        }


@dataclass(frozen=True, slots=True)
class CheapCandidateFeatures:
    layer_index: int | None
    affected_component_count: int
    constraint_count: int
    request_parameter_count: int
    scope: str
    node_kind: str

    def to_record(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "affected_component_count": self.affected_component_count,
            "constraint_count": self.constraint_count,
            "request_parameter_count": self.request_parameter_count,
            "scope": self.scope,
            "node_kind": self.node_kind,
        }


@dataclass(frozen=True, slots=True)
class CandidatePoolManifest:
    path: str
    candidate_count: int
    requested_count: int
    complete: bool
    content_sha256: str
    provenance: CandidatePoolProvenance
    schema_version: int = CANDIDATE_POOL_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "candidate_count": self.candidate_count,
            "requested_count": self.requested_count,
            "complete": self.complete,
            "content_sha256": self.content_sha256,
            "provenance": self.provenance.to_record(),
        }


def cheap_candidate_features(candidate: MutationCandidate) -> CheapCandidateFeatures:
    """Extract mutation-free graph/request features suitable for pool scoring."""

    return CheapCandidateFeatures(
        layer_index=candidate.layer_index,
        affected_component_count=len(candidate.affected_components),
        constraint_count=len(candidate.constraint_ids),
        request_parameter_count=len(candidate.request.parameters),
        scope=candidate.scope.value,
        node_kind=candidate.node_kind,
    )


def write_candidate_pool(
    candidates: Sequence[MutationCandidate],
    output: Path,
    provenance: CandidatePoolProvenance,
    *,
    max_candidates: int = MAX_CANDIDATE_POOL_SIZE,
    max_new_candidates: int | None = None,
) -> CandidatePoolManifest:
    """Append one deterministic bounded segment and return a resumable manifest."""

    if not 1 <= max_candidates <= MAX_CANDIDATE_POOL_SIZE:
        raise CandidatePoolError("candidate pool maximum must be within 1..100000")
    requested = min(len(candidates), max_candidates)
    if max_new_candidates is not None and max_new_candidates <= 0:
        raise CandidatePoolError("candidate pool invocation limit must be positive")
    selected = candidates[:requested]
    ids = tuple(item.candidate_id for item in selected)
    if len(ids) != len(set(ids)):
        raise CandidatePoolError("candidate pool input IDs must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(output.suffix + ".manifest.json")
    existing_count, digest = _validate_resume(output, checkpoint, selected, provenance)
    remaining = requested - existing_count
    to_write = remaining if max_new_candidates is None else min(remaining, max_new_candidates)
    mode = "ab" if existing_count else "wb"
    with output.open(mode) as stream:
        for candidate in selected[existing_count : existing_count + to_write]:
            payload = _candidate_record(candidate, provenance)
            encoded = (canonical_identity_json(payload) + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    count = existing_count + to_write
    manifest = CandidatePoolManifest(
        str(output), count, requested, count == requested, digest.hexdigest(), provenance
    )
    _publish_manifest(checkpoint, manifest)
    return manifest


def _candidate_record(
    candidate: MutationCandidate, provenance: CandidatePoolProvenance
) -> dict[str, object]:
    return {
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "candidate": candidate.to_record(),
        "cheap_features": cheap_candidate_features(candidate).to_record(),
        "provenance": provenance.to_record(),
    }


def _validate_resume(
    output: Path,
    checkpoint: Path,
    candidates: Sequence[MutationCandidate],
    provenance: CandidatePoolProvenance,
) -> tuple[int, Any]:
    digest = hashlib.sha256()
    if not output.exists() and not checkpoint.exists():
        return 0, digest
    if output.exists() != checkpoint.exists():
        raise CandidatePoolError("candidate pool data and resume manifest must both exist")
    try:
        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidatePoolError("candidate pool resume manifest is unreadable") from error
    if not isinstance(raw, Mapping):
        raise CandidatePoolError("candidate pool resume manifest must be an object")
    if raw.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
        raise CandidatePoolError("candidate pool resume schema is incompatible")
    if raw.get("provenance") != provenance.to_record():
        raise CandidatePoolError("candidate pool resume provenance changed")
    expected_count = raw.get("candidate_count")
    expected_digest = raw.get("content_sha256")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise CandidatePoolError("candidate pool resume count is invalid")
    if not isinstance(expected_digest, str) or expected_count > len(candidates):
        raise CandidatePoolError("candidate pool resume bounds are invalid")
    count = 0
    try:
        with output.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if count >= expected_count:
                    raise CandidatePoolError("candidate pool contains uncheckpointed records")
                record = json.loads(line)
                candidate_raw = record.get("candidate") if isinstance(record, Mapping) else None
                if not isinstance(candidate_raw, Mapping):
                    raise CandidatePoolError("candidate pool record is malformed")
                if candidate_raw.get("candidate_id") != candidates[count].candidate_id:
                    raise CandidatePoolError("candidate pool deterministic prefix changed")
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidatePoolError("candidate pool data is unreadable") from error
    if count != expected_count or digest.hexdigest() != expected_digest:
        raise CandidatePoolError("candidate pool resume digest/count mismatch")
    return count, digest


def _publish_manifest(path: Path, manifest: CandidatePoolManifest) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(manifest.to_record(), indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
