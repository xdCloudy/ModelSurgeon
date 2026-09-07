"""Bounded, deterministic multi-axis architecture candidate spaces."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.deployable_state import ArchitectureAxis

CANDIDATE_SPACE_SCHEMA_VERSION: Final[int] = 1
MAX_CANDIDATES: Final[int] = 100_000


class CandidateSpaceError(ValueError):
    """Raised when a candidate space is invalid or cannot resume safely."""


class CandidateOutcome(StrEnum):
    EMITTED = "emitted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateSpaceError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateSpaceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateSpaceError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ArchitectureChoice:
    """One legal value with bounded analytic cost estimates."""

    value: object
    parameter_count: int
    storage_bytes: int
    source_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.parameter_count < 0 or self.storage_bytes < 0:
            raise CandidateSpaceError("choice costs cannot be negative")
        distance = _finite(self.source_distance, "source distance")
        if distance < 0:
            raise CandidateSpaceError("source distance cannot be negative")
        _canonical(self.value)

    def to_record(self) -> dict[str, object]:
        return {
            "value": self.value,
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "source_distance": self.source_distance,
        }


@dataclass(frozen=True, slots=True)
class AxisDomain:
    axis: ArchitectureAxis
    choices: tuple[ArchitectureChoice, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise CandidateSpaceError(f"axis {self.axis.value} requires at least one choice")
        keys = tuple(_canonical(item.value) for item in self.choices)
        if len(keys) != len(set(keys)):
            raise CandidateSpaceError(
                f"axis {self.axis.value} choices must be unique and canonical"
            )
        object.__setattr__(
            self,
            "choices",
            tuple(sorted(self.choices, key=lambda item: _canonical(item.value))),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "choices": [item.to_record() for item in self.choices],
        }


@dataclass(frozen=True, slots=True)
class HardwareRule:
    profile_id: str
    max_parameter_count: int | None = None
    max_storage_bytes: int | None = None

    def __post_init__(self) -> None:
        _text(self.profile_id, "hardware profile ID")
        if self.max_parameter_count is not None and self.max_parameter_count <= 0:
            raise CandidateSpaceError("hardware parameter ceiling must be positive")
        if self.max_storage_bytes is not None and self.max_storage_bytes <= 0:
            raise CandidateSpaceError("hardware storage ceiling must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "max_parameter_count": self.max_parameter_count,
            "max_storage_bytes": self.max_storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class CandidateSpaceRules:
    max_parameter_count: int | None = None
    max_storage_bytes: int | None = None
    max_source_distance: float | None = None
    divisible_axes: tuple[tuple[ArchitectureAxis, ArchitectureAxis], ...] = ()

    def __post_init__(self) -> None:
        if self.max_parameter_count is not None and self.max_parameter_count <= 0:
            raise CandidateSpaceError("parameter ceiling must be positive")
        if self.max_storage_bytes is not None and self.max_storage_bytes <= 0:
            raise CandidateSpaceError("storage ceiling must be positive")
        if self.max_source_distance is not None and _finite(
            self.max_source_distance, "source distance ceiling"
        ) < 0:
            raise CandidateSpaceError("source distance ceiling cannot be negative")
        pairs = tuple(
            sorted(set(self.divisible_axes), key=lambda item: (item[0].value, item[1].value))
        )
        if len(pairs) != len(self.divisible_axes):
            raise CandidateSpaceError("divisibility pairs must be unique and canonical")
        object.__setattr__(self, "divisible_axes", pairs)
        if any(left is right for left, right in pairs):
            raise CandidateSpaceError("divisibility pairs require distinct axes")

    def to_record(self) -> dict[str, object]:
        return {
            "max_parameter_count": self.max_parameter_count,
            "max_storage_bytes": self.max_storage_bytes,
            "max_source_distance": self.max_source_distance,
            "divisible_axes": [[left.value, right.value] for left, right in self.divisible_axes],
        }


@dataclass(frozen=True, slots=True)
class CandidateSpaceConfig:
    seed: int
    max_candidates: int = MAX_CANDIDATES
    page_size: int = 256

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise CandidateSpaceError("candidate space seed must be unsigned 64-bit")
        if not 0 < self.max_candidates <= MAX_CANDIDATES:
            raise CandidateSpaceError("candidate space maximum must be within 1..100000")
        if not 0 < self.page_size <= self.max_candidates:
            raise CandidateSpaceError("candidate space page size must be within the candidate cap")

    def to_record(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "max_candidates": self.max_candidates,
            "page_size": self.page_size,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    base_state_id: str
    assignments: tuple[tuple[ArchitectureAxis, object], ...]
    hardware_profile_id: str
    parameter_count: int
    storage_bytes: int
    source_distance: float
    outcome: CandidateOutcome = CandidateOutcome.EMITTED

    def __post_init__(self) -> None:
        if not self.base_state_id.startswith("state_"):
            raise CandidateSpaceError("architecture candidates require a canonical base state")
        _text(self.hardware_profile_id, "candidate hardware profile ID")
        axes = tuple(axis for axis, _ in self.assignments)
        if axes != tuple(sorted(set(axes), key=lambda item: item.value)):
            raise CandidateSpaceError("candidate assignments must be unique and canonical")
        if self.parameter_count < 0 or self.storage_bytes < 0:
            raise CandidateSpaceError("candidate costs cannot be negative")
        if _finite(self.source_distance, "candidate source distance") < 0:
            raise CandidateSpaceError("candidate source distance cannot be negative")

    @property
    def candidate_id(self) -> str:
        payload = {
            "schema_version": CANDIDATE_SPACE_SCHEMA_VERSION,
            "base_state_id": self.base_state_id,
            "assignments": [[axis.value, value] for axis, value in self.assignments],
            "hardware_profile_id": self.hardware_profile_id,
        }
        return f"architecture_candidate_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CANDIDATE_SPACE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "base_state_id": self.base_state_id,
            "assignments": [
                {"axis": axis.value, "value": value} for axis, value in self.assignments
            ],
            "hardware_profile_id": self.hardware_profile_id,
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "source_distance": self.source_distance,
            "outcome": self.outcome.value,
        }


@dataclass(frozen=True, slots=True)
class CandidateGenerationPage:
    space_id: str
    candidates: tuple[ArchitectureCandidate, ...]
    next_cursor: int
    complete: bool
    rejection_counts: tuple[tuple[str, int], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CANDIDATE_SPACE_SCHEMA_VERSION,
            "space_id": self.space_id,
            "candidates": [item.to_record() for item in self.candidates],
            "next_cursor": self.next_cursor,
            "complete": self.complete,
            "rejection_counts": [
                {"reason": reason, "count": count} for reason, count in self.rejection_counts
            ],
        }


class ArchitectureCandidateSpace:
    """Lazy Cartesian space with bounded ranked retention and resumable pages."""

    def __init__(
        self,
        base_state_id: str,
        domains: tuple[AxisDomain, ...],
        hardware_rules: tuple[HardwareRule, ...],
        rules: CandidateSpaceRules,
        config: CandidateSpaceConfig,
    ) -> None:
        if not base_state_id.startswith("state_"):
            raise CandidateSpaceError("candidate spaces require a canonical base state")
        if not domains:
            raise CandidateSpaceError("candidate spaces require at least one axis domain")
        axes = tuple(domain.axis for domain in domains)
        if len(axes) != len(set(axes)):
            raise CandidateSpaceError("axis domains must be unique and canonical")
        profiles = tuple(rule.profile_id for rule in hardware_rules)
        if len(profiles) != len(set(profiles)):
            raise CandidateSpaceError("hardware rules must be unique and canonical")
        if not hardware_rules:
            raise CandidateSpaceError("candidate spaces require at least one hardware rule")
        domain_axes = set(axes)
        if any(
            left not in domain_axes or right not in domain_axes
            for left, right in rules.divisible_axes
        ):
            raise CandidateSpaceError(
                "divisibility rules must reference declared axis domains"
            )
        self.base_state_id = base_state_id
        self.domains = tuple(sorted(domains, key=lambda item: item.axis.value))
        self.hardware_rules = tuple(sorted(hardware_rules, key=lambda item: item.profile_id))
        self.rules = rules
        self.config = config

    @property
    def space_id(self) -> str:
        payload = {
            "schema_version": CANDIDATE_SPACE_SCHEMA_VERSION,
            "base_state_id": self.base_state_id,
            "domains": [item.to_record() for item in self.domains],
            "hardware_rules": [item.to_record() for item in self.hardware_rules],
            "rules": self.rules.to_record(),
            "config": self.config.to_record(),
        }
        return f"space_{hashlib.sha256(canonical_identity_json(payload).encode()).hexdigest()}"

    def _rank(self, candidate: ArchitectureCandidate) -> tuple[int, str]:
        digest = hashlib.sha256(
            f"{self.config.seed}:{candidate.candidate_id}".encode("ascii")
        ).digest()
        return int.from_bytes(digest, "big"), candidate.candidate_id

    def _legal(
        self,
        assignments: dict[ArchitectureAxis, ArchitectureChoice],
        placement: HardwareRule,
    ) -> str | None:
        parameters = sum(item.parameter_count for item in assignments.values())
        storage = sum(item.storage_bytes for item in assignments.values())
        distance = math.fsum(item.source_distance for item in assignments.values())
        if (
            self.rules.max_parameter_count is not None
            and parameters > self.rules.max_parameter_count
        ):
            return "parameter_ceiling"
        if self.rules.max_storage_bytes is not None and storage > self.rules.max_storage_bytes:
            return "storage_ceiling"
        if self.rules.max_source_distance is not None and distance > self.rules.max_source_distance:
            return "source_distance_ceiling"
        if (
            placement.max_parameter_count is not None
            and parameters > placement.max_parameter_count
        ):
            return f"hardware_parameter_ceiling:{placement.profile_id}"
        if placement.max_storage_bytes is not None and storage > placement.max_storage_bytes:
            return f"hardware_storage_ceiling:{placement.profile_id}"
        for left, right in self.rules.divisible_axes:
            left_value = assignments[left].value
            right_value = assignments[right].value
            if not isinstance(left_value, int) or isinstance(left_value, bool):
                return f"non_integer_divisor:{left.value}"
            if not isinstance(right_value, int) or isinstance(right_value, bool):
                return f"non_integer_dividend:{right.value}"
            if left_value <= 0 or right_value <= 0 or right_value % left_value:
                return f"illegal_divisibility:{left.value}:{right.value}"
        return None

    def _iter_candidates(
        self,
    ) -> tuple[tuple[ArchitectureCandidate, ...], tuple[tuple[str, int], ...]]:
        choices = tuple(domain.choices for domain in self.domains)
        retained: list[tuple[tuple[int, str], ArchitectureCandidate]] = []
        rejections: Counter[str] = Counter()
        for placement in self.hardware_rules:
            for combination in itertools.product(*choices):
                assignment_map = {
                    domain.axis: choice
                    for domain, choice in zip(self.domains, combination, strict=True)
                }
                reason = self._legal(assignment_map, placement)
                if reason is not None:
                    rejections[reason] += 1
                    continue
                candidate = ArchitectureCandidate(
                    self.base_state_id,
                    tuple(
                        (axis, assignment_map[axis].value)
                        for axis in sorted(assignment_map, key=lambda item: item.value)
                    ),
                    placement.profile_id,
                    sum(item.parameter_count for item in combination),
                    sum(item.storage_bytes for item in combination),
                    math.fsum(item.source_distance for item in combination),
                )
                ranked = (self._rank(candidate), candidate)
                if len(retained) < self.config.max_candidates:
                    heapq.heappush(retained, ranked)
                elif ranked > retained[0]:
                    heapq.heapreplace(retained, ranked)
        selected = sorted(retained, key=lambda item: item[0])
        return (
            tuple(item[1] for item in selected),
            tuple(sorted(rejections.items())),
        )

    def generate(self, *, cursor: int = 0) -> CandidateGenerationPage:
        if cursor < 0:
            raise CandidateSpaceError("candidate space cursor cannot be negative")
        candidates, rejections = self._iter_candidates()
        if cursor > len(candidates):
            raise CandidateSpaceError("candidate space cursor exceeds retained candidates")
        page = candidates[cursor : cursor + self.config.page_size]
        next_cursor = cursor + len(page)
        return CandidateGenerationPage(
            self.space_id,
            page,
            next_cursor,
            next_cursor == len(candidates),
            rejections,
        )
