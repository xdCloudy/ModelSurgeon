"""Canonical mutation equivalence and active-work exclusion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from modelsurgeon.experiments.candidates import MutationCandidate
from modelsurgeon.experiments.identity import canonical_identity_json

CANDIDATE_EQUIVALENCE_SCHEMA_VERSION: Final[int] = 1


class CandidateDeduplicationError(ValueError):
    """Raised when candidate equivalence declarations are ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class AdapterEquivalenceDeclarations:
    adapter: str
    version: str
    by_mutation_id: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.adapter or not self.version:
            raise CandidateDeduplicationError("adapter equivalence identity/version is required")
        if any(not mutation_id or not key for mutation_id, key in self.by_mutation_id.items()):
            raise CandidateDeduplicationError("adapter equivalence declarations cannot be blank")


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    candidate_id: str
    equivalence_key: str
    reason: str

    def to_record(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "equivalence_key": self.equivalence_key,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateDeduplicationReport:
    candidates: tuple[MutationCandidate, ...]
    exclusions: tuple[CandidateExclusion, ...]
    equivalence_keys: tuple[tuple[str, str], ...]
    schema_version: int = CANDIDATE_EQUIVALENCE_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_count": len(self.candidates),
            "candidates": [item.to_record() for item in self.candidates],
            "equivalence_keys": dict(self.equivalence_keys),
            "exclusions": [item.to_record() for item in self.exclusions],
        }


def candidate_equivalence_key(
    candidate: MutationCandidate,
    declarations: AdapterEquivalenceDeclarations | None = None,
) -> str:
    """Hash canonical semantics, or a namespaced adapter-declared equivalence class."""

    declared = None
    if declarations is not None:
        declared = declarations.by_mutation_id.get(candidate.mutation_id)
    if declared is not None and declarations is not None:
        payload: dict[str, object] = {
            "schema_version": CANDIDATE_EQUIVALENCE_SCHEMA_VERSION,
            "source": "adapter",
            "adapter": declarations.adapter,
            "adapter_version": declarations.version,
            "declared_key": declared,
        }
    else:
        payload = {
            "schema_version": CANDIDATE_EQUIVALENCE_SCHEMA_VERSION,
            "source": "canonical-request",
            "mutation_kind": candidate.request.kind.value,
            "affected_components": sorted(str(item) for item in candidate.affected_components),
            "parameters": [[name, value] for name, value in candidate.request.parameters],
        }
    digest = hashlib.sha256(canonical_identity_json(payload).encode("utf-8")).hexdigest()
    return f"eq_{digest}"


def deduplicate_candidates(
    candidates: Sequence[MutationCandidate],
    *,
    completed: Sequence[MutationCandidate] = (),
    in_flight: Sequence[MutationCandidate] = (),
    declarations: AdapterEquivalenceDeclarations | None = None,
) -> CandidateDeduplicationReport:
    """Exclude completed, in-flight, and repeated equivalence classes deterministically."""

    completed_keys = {candidate_equivalence_key(item, declarations) for item in completed}
    in_flight_keys = {candidate_equivalence_key(item, declarations) for item in in_flight}
    if completed_keys & in_flight_keys:
        raise CandidateDeduplicationError(
            "the same mutation equivalence class cannot be completed and in-flight"
        )
    seen: set[str] = set()
    accepted: list[MutationCandidate] = []
    exclusions: list[CandidateExclusion] = []
    keys: list[tuple[str, str]] = []
    for candidate in candidates:
        key = candidate_equivalence_key(candidate, declarations)
        if key in completed_keys:
            exclusions.append(CandidateExclusion(candidate.candidate_id, key, "already-completed"))
        elif key in in_flight_keys:
            exclusions.append(CandidateExclusion(candidate.candidate_id, key, "already-in-flight"))
        elif key in seen:
            exclusions.append(
                CandidateExclusion(candidate.candidate_id, key, "duplicate-equivalence")
            )
        else:
            seen.add(key)
            accepted.append(candidate)
            keys.append((candidate.candidate_id, key))
    return CandidateDeduplicationReport(tuple(accepted), tuple(exclusions), tuple(keys))
