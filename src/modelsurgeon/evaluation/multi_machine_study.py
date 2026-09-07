"""Bounded multi-machine evidence for hardware-aware candidate selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.config import ObjectiveDirection

MULTI_MACHINE_STUDY_SCHEMA_VERSION: Final[int] = 1
MIN_REPETITIONS: Final[int] = 3


class MultiMachineStudyError(ValueError):
    """Raised when multi-machine evidence cannot support a comparison."""


class MachineClass(StrEnum):
    CPU_ONLY = "cpu_only"
    LOW_VRAM = "low_vram"
    FULL_GPU = "full_gpu"


class StudyArm(StrEnum):
    HARDWARE_AWARE = "hardware_aware"
    HARDWARE_BLIND = "hardware_blind"


class ObservationStatus(StrEnum):
    MEASURED = "measured"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ComparisonStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class StudyClaim(StrEnum):
    POSITIVE = "positive"
    NEGATIVE_RESULT = "negative_result"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiMachineStudyError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiMachineStudyError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiMachineStudyError(f"{label} must be finite")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise MultiMachineStudyError(f"{label} must be non-negative")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class MachineProfile:
    """Immutable host/runtime identity retained with every study result."""

    profile_id: str
    machine_class: MachineClass
    runtime: str
    cpu_threads: int
    vram_gb: float
    hardware_identity: str
    environment_digest: str

    def __post_init__(self) -> None:
        _text(self.profile_id, "profile ID")
        _text(self.runtime, "runtime")
        _text(self.hardware_identity, "hardware identity")
        _text(self.environment_digest, "environment digest")
        if self.cpu_threads <= 0:
            raise MultiMachineStudyError("CPU threads must be positive")
        _nonnegative(self.vram_gb, "VRAM")
        if self.machine_class is MachineClass.CPU_ONLY and self.vram_gb != 0:
            raise MultiMachineStudyError("CPU-only profiles must report zero VRAM")
        if self.machine_class is MachineClass.LOW_VRAM and self.vram_gb <= 0:
            raise MultiMachineStudyError("low-VRAM profiles must report positive VRAM")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "machine_class": self.machine_class.value,
            "runtime": self.runtime,
            "cpu_threads": self.cpu_threads,
            "vram_gb": self.vram_gb,
            "hardware_identity": self.hardware_identity,
            "environment_digest": self.environment_digest,
        }


@dataclass(frozen=True, slots=True)
class StudyObjective:
    """Requested deployment objective; lower or higher values can be preferred."""

    name: str
    direction: ObjectiveDirection
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        _text(self.name, "objective name")
        _nonnegative(self.tolerance, "objective tolerance")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class StudyObservation:
    """One bounded host/run result, including terminal non-measured outcomes."""

    profile_id: str
    candidate_id: str
    arm: StudyArm
    objective: str
    repetition: int
    status: ObservationStatus
    value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.profile_id, "observation profile ID")
        _text(self.candidate_id, "observation candidate ID")
        _text(self.objective, "observation objective")
        if self.repetition < 0:
            raise MultiMachineStudyError("observation repetition cannot be negative")
        if self.status is ObservationStatus.MEASURED:
            if self.value is None:
                raise MultiMachineStudyError("measured observations require a value")
            _nonnegative(self.value, "measured objective value")
            if self.reason is not None:
                raise MultiMachineStudyError("measured observations cannot carry a failure reason")
        elif self.value is not None or not self.reason:
            raise MultiMachineStudyError(
                "non-measured observations require a reason and no objective value"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "candidate_id": self.candidate_id,
            "arm": self.arm.value,
            "objective": self.objective,
            "repetition": self.repetition,
            "status": self.status.value,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepeatedMetric:
    """Deterministic host-level mean and 95% normal-approximation interval."""

    samples: tuple[float, ...]
    mean: float
    confidence_low: float
    confidence_high: float

    def __post_init__(self) -> None:
        if len(self.samples) < MIN_REPETITIONS:
            raise MultiMachineStudyError("host comparisons require at least three measurements")
        if any(not math.isfinite(value) or value < 0 for value in self.samples):
            raise MultiMachineStudyError("metric samples must be finite and non-negative")
        if not self.confidence_low <= self.mean <= self.confidence_high:
            raise MultiMachineStudyError("confidence interval must contain the mean")

    def to_record(self) -> dict[str, object]:
        return {
            "samples": list(self.samples),
            "mean": self.mean,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


def _summarize(values: tuple[float, ...]) -> RepeatedMetric:
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        standard_error = 0.0
    else:
        variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_error = math.sqrt(variance / len(values))
    margin = 1.96 * standard_error
    return RepeatedMetric(values, mean, max(0.0, mean - margin), mean + margin)


@dataclass(frozen=True, slots=True)
class SelectionComparison:
    profile_id: str
    objective: StudyObjective
    hardware_aware_candidate_id: str | None
    hardware_blind_candidate_id: str | None
    aware_metric: RepeatedMetric | None
    blind_metric: RepeatedMetric | None
    status: ComparisonStatus
    aware_beats_or_ties: bool | None
    reason: str

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "objective": self.objective.to_record(),
            "hardware_aware_candidate_id": self.hardware_aware_candidate_id,
            "hardware_blind_candidate_id": self.hardware_blind_candidate_id,
            "aware_metric": None if self.aware_metric is None else self.aware_metric.to_record(),
            "blind_metric": None if self.blind_metric is None else self.blind_metric.to_record(),
            "status": self.status.value,
            "aware_beats_or_ties": self.aware_beats_or_ties,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MultiMachineStudy:
    """Complete, bounded cross-profile comparison with an explicit claim boundary."""

    study_id: str
    profiles: tuple[MachineProfile, ...]
    candidate_ids: tuple[str, ...]
    objectives: tuple[StudyObjective, ...]
    observations: tuple[StudyObservation, ...]
    source_artifact_digest: str
    corpus_identity: str
    optimization_budget_id: str
    comparisons: tuple[SelectionComparison, ...]
    claim: StudyClaim
    schema_version: int = MULTI_MACHINE_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.study_id.startswith("study_"):
            raise MultiMachineStudyError("study IDs must be canonical")
        if self.schema_version != MULTI_MACHINE_STUDY_SCHEMA_VERSION:
            raise MultiMachineStudyError("unsupported multi-machine study schema")
        if len(self.profiles) < 3:
            raise MultiMachineStudyError("multi-machine studies require at least three profiles")
        classes = {profile.machine_class for profile in self.profiles}
        if not classes & {MachineClass.CPU_ONLY, MachineClass.LOW_VRAM}:
            raise MultiMachineStudyError("study requires a CPU-only or low-VRAM profile")
        if len(self.candidate_ids) < 2 or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise MultiMachineStudyError("study requires at least two unique candidates")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise MultiMachineStudyError("profile IDs must be unique")
        _text(self.source_artifact_digest, "source artifact digest")
        _text(self.corpus_identity, "corpus identity")
        _text(self.optimization_budget_id, "optimization budget ID")
        self._validate_observations()

    def _validate_observations(self) -> None:
        profile_ids = {profile.profile_id for profile in self.profiles}
        candidate_ids = set(self.candidate_ids)
        objective_names = {objective.name for objective in self.objectives}
        keys: set[tuple[str, str, StudyArm, str, int]] = set()
        for observation in self.observations:
            key = (
                observation.profile_id,
                observation.candidate_id,
                observation.arm,
                observation.objective,
                observation.repetition,
            )
            if key in keys:
                raise MultiMachineStudyError("study observations must be unique")
            keys.add(key)
            if (
                observation.profile_id not in profile_ids
                or observation.candidate_id not in candidate_ids
            ):
                raise MultiMachineStudyError(
                    "observation references an unknown profile or candidate"
                )
            if observation.objective not in objective_names:
                raise MultiMachineStudyError("observation references an unknown objective")
        expected = {
            (profile.profile_id, candidate_id, arm, objective.name)
            for profile in self.profiles
            for candidate_id in self.candidate_ids
            for arm in StudyArm
            for objective in self.objectives
        }
        for profile_id, candidate_id, arm, objective in expected:
            records = [
                item
                for item in self.observations
                if (item.profile_id, item.candidate_id, item.arm, item.objective)
                == (profile_id, candidate_id, arm, objective)
            ]
            if len(records) < MIN_REPETITIONS:
                raise MultiMachineStudyError(
                    "each profile/candidate/arm requires three repetitions"
                )

    @property
    def computed_study_id(self) -> str:
        payload = {
            "profiles": [item.to_record() for item in self.profiles],
            "candidate_ids": list(self.candidate_ids),
            "objectives": [item.to_record() for item in self.objectives],
            "source_artifact_digest": self.source_artifact_digest,
            "corpus_identity": self.corpus_identity,
            "optimization_budget_id": self.optimization_budget_id,
        }
        return f"study_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    @property
    def hardware_specific_choices(self) -> bool:
        selected = {
            item.hardware_aware_candidate_id
            for item in self.comparisons
            if item.status is ComparisonStatus.MEASURED
            and item.hardware_aware_candidate_id is not None
        }
        return len(selected) > 1

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "computed_study_id": self.computed_study_id,
            "profiles": [item.to_record() for item in self.profiles],
            "candidate_ids": list(self.candidate_ids),
            "objectives": [item.to_record() for item in self.objectives],
            "observations": [item.to_record() for item in self.observations],
            "source_artifact_digest": self.source_artifact_digest,
            "corpus_identity": self.corpus_identity,
            "optimization_budget_id": self.optimization_budget_id,
            "comparisons": [item.to_record() for item in self.comparisons],
            "hardware_specific_choices": self.hardware_specific_choices,
            "claim": self.claim.value,
        }


def _metric_for(
    observations: tuple[StudyObservation, ...],
    profile_id: str,
    candidate_id: str,
    arm: StudyArm,
    objective: str,
) -> RepeatedMetric | None:
    values = tuple(
        item.value
        for item in observations
        if item.profile_id == profile_id
        and item.candidate_id == candidate_id
        and item.arm is arm
        and item.objective == objective
        and item.status is ObservationStatus.MEASURED
        and item.value is not None
    )
    return _summarize(values) if len(values) >= MIN_REPETITIONS else None


def _select(
    candidates: tuple[str, ...], metrics: dict[str, RepeatedMetric], objective: StudyObjective
) -> str:
    return sorted(
        candidates,
        key=lambda candidate: (
            metrics[candidate].mean
            if objective.direction is ObjectiveDirection.MINIMIZE
            else -metrics[candidate].mean,
            candidate,
        ),
    )[0]


def build_multi_machine_study(
    *,
    profiles: tuple[MachineProfile, ...],
    candidate_ids: tuple[str, ...],
    objectives: tuple[StudyObjective, ...],
    observations: tuple[StudyObservation, ...],
    source_artifact_digest: str,
    corpus_identity: str,
    optimization_budget_id: str,
) -> MultiMachineStudy:
    """Build deterministic per-profile aware-vs-blind comparisons."""

    if not objectives or len({item.name for item in objectives}) != len(objectives):
        raise MultiMachineStudyError("study objectives must be non-empty and unique")
    comparisons: list[SelectionComparison] = []
    for profile in profiles:
        for objective in objectives:
            aware_metrics = {
                candidate: metric
                for candidate in candidate_ids
                if (metric := _metric_for(
                    observations,
                    profile.profile_id,
                    candidate,
                    StudyArm.HARDWARE_AWARE,
                    objective.name,
                )) is not None
            }
            blind_metrics = {
                candidate: metric
                for candidate in candidate_ids
                if (metric := _metric_for(
                    observations,
                    profile.profile_id,
                    candidate,
                    StudyArm.HARDWARE_BLIND,
                    objective.name,
                )) is not None
            }
            pooled_blind: dict[str, RepeatedMetric] = {}
            for candidate in candidate_ids:
                values = tuple(
                    item.value
                    for item in observations
                    if item.candidate_id == candidate
                    and item.arm is StudyArm.HARDWARE_BLIND
                    and item.objective == objective.name
                    and item.status is ObservationStatus.MEASURED
                    and item.value is not None
                )
                if len(values) >= MIN_REPETITIONS:
                    pooled_blind[candidate] = _summarize(values)
            if (
                len(aware_metrics) != len(candidate_ids)
                or len(blind_metrics) != len(candidate_ids)
                or len(pooled_blind) != len(candidate_ids)
            ):
                comparisons.append(
                    SelectionComparison(
                        profile.profile_id,
                        objective,
                        None,
                        None,
                        None,
                        None,
                        ComparisonStatus.UNKNOWN,
                        None,
                        "insufficient_measured_evidence",
                    )
                )
                continue
            aware_candidate = _select(tuple(aware_metrics), aware_metrics, objective)
            blind_candidate = _select(tuple(pooled_blind), pooled_blind, objective)
            aware_metric = aware_metrics[aware_candidate]
            blind_metric = blind_metrics.get(blind_candidate)
            if blind_metric is None:
                comparisons.append(
                    SelectionComparison(
                        profile.profile_id,
                        objective,
                        aware_candidate,
                        blind_candidate,
                        aware_metric,
                        None,
                        ComparisonStatus.UNKNOWN,
                        None,
                        "blind_choice_unmeasured_on_profile",
                    )
                )
                continue
            if objective.direction is ObjectiveDirection.MINIMIZE:
                beats = aware_metric.mean <= blind_metric.mean + objective.tolerance
            else:
                beats = aware_metric.mean + objective.tolerance >= blind_metric.mean
            comparisons.append(
                SelectionComparison(
                    profile.profile_id,
                    objective,
                    aware_candidate,
                    blind_candidate,
                    aware_metric,
                    blind_metric,
                    ComparisonStatus.MEASURED,
                    beats,
                    "measured_profile_comparison",
                )
            )
    measured = [item for item in comparisons if item.status is ComparisonStatus.MEASURED]
    if not measured:
        claim = StudyClaim.UNSUPPORTED
    elif len(measured) != len(comparisons):
        claim = StudyClaim.INCONCLUSIVE
    elif all(item.aware_beats_or_ties for item in measured):
        claim = (
            StudyClaim.POSITIVE
            if any(
                item.hardware_aware_candidate_id != item.hardware_blind_candidate_id
                for item in measured
            )
            else StudyClaim.NEGATIVE_RESULT
        )
    else:
        claim = StudyClaim.NEGATIVE_RESULT
    payload = {
        "profiles": [item.to_record() for item in profiles],
        "candidate_ids": list(candidate_ids),
        "objectives": [item.to_record() for item in objectives],
        "source_artifact_digest": source_artifact_digest,
        "corpus_identity": corpus_identity,
        "optimization_budget_id": optimization_budget_id,
    }
    study_id = f"study_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"
    return MultiMachineStudy(
        study_id,
        profiles,
        candidate_ids,
        objectives,
        observations,
        source_artifact_digest,
        corpus_identity,
        optimization_budget_id,
        tuple(comparisons),
        claim,
    )
