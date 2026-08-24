"""Validated comparative records for iterative surgery and repair studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

ITERATIVE_SEARCH_STUDY_SCHEMA_VERSION = 1


class IterativeSearchStudyError(ValueError):
    """Raised when a comparative study would overstate incomplete evidence."""


class StudyBackend(StrEnum):
    HUGGING_FACE = "hugging_face"
    NATIVE_GGUF = "native_gguf"


class StudyArm(StrEnum):
    NO_REPAIR = "no_repair"
    REPAIR = "repair"
    ONE_SHOT = "one_shot"


@dataclass(frozen=True, slots=True)
class SearchGoals:
    max_quality_increase: float
    min_latency_gain_ratio: float
    min_size_gain_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.max_quality_increase,
            self.min_latency_gain_ratio,
            self.min_size_gain_ratio,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise IterativeSearchStudyError("search goals must be finite and non-negative")

    def to_record(self) -> dict[str, float]:
        return {
            "max_quality_increase": self.max_quality_increase,
            "min_latency_gain_ratio": self.min_latency_gain_ratio,
            "min_size_gain_ratio": self.min_size_gain_ratio,
        }


@dataclass(frozen=True, slots=True)
class StudyMeasurement:
    measurement_id: str
    backend: StudyBackend
    removed_channels: tuple[int, ...]
    quality_value: float
    median_latency_seconds: float
    size_bytes: int
    parameter_count: int
    wall_seconds: float
    gpu_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    artifact_sha256: str | None

    def __post_init__(self) -> None:
        if not self.measurement_id.startswith("measurement_"):
            raise IterativeSearchStudyError("measurement IDs must be canonical")
        if self.removed_channels != tuple(sorted(set(self.removed_channels))) or any(
            channel < 0 for channel in self.removed_channels
        ):
            raise IterativeSearchStudyError("removed channels must be canonical")
        finite = (
            self.quality_value,
            self.median_latency_seconds,
            self.wall_seconds,
            self.gpu_seconds,
            self.cpu_seconds,
        )
        if (
            any(not math.isfinite(value) or value < 0 for value in finite)
            or self.median_latency_seconds <= 0
            or self.size_bytes <= 0
            or self.parameter_count <= 0
        ):
            raise IterativeSearchStudyError("measurement values and costs are invalid")
        if self.artifact_sha256 is not None:
            if len(self.artifact_sha256) != 64:
                raise IterativeSearchStudyError("artifact digest must be SHA-256")
            try:
                int(self.artifact_sha256, 16)
            except ValueError as error:
                raise IterativeSearchStudyError("artifact digest must be hexadecimal") from error
        for value in (self.peak_rss_bytes, self.peak_vram_bytes):
            if value is not None and value < 0:
                raise IterativeSearchStudyError("memory peaks cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "measurement_id": self.measurement_id,
            "backend": self.backend.value,
            "removed_channels": list(self.removed_channels),
            "quality_value": self.quality_value,
            "median_latency_seconds": self.median_latency_seconds,
            "size_bytes": self.size_bytes,
            "parameter_count": self.parameter_count,
            "cost": {
                "wall_seconds": self.wall_seconds,
                "gpu_seconds": self.gpu_seconds,
                "cpu_seconds": self.cpu_seconds,
                "peak_rss_bytes": self.peak_rss_bytes,
                "peak_vram_bytes": self.peak_vram_bytes,
            },
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class SearchGeneration:
    generation: int
    parent_measurement_id: str
    candidate_measurement_ids: tuple[str, ...]
    selected_measurement_id: str

    def __post_init__(self) -> None:
        if self.generation <= 0 or not self.parent_measurement_id.startswith("measurement_"):
            raise IterativeSearchStudyError("search generation identity is invalid")
        if (
            not self.candidate_measurement_ids
            or self.candidate_measurement_ids != tuple(sorted(set(self.candidate_measurement_ids)))
            or self.selected_measurement_id not in self.candidate_measurement_ids
        ):
            raise IterativeSearchStudyError("generation candidates/selection are invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "parent_measurement_id": self.parent_measurement_id,
            "candidate_measurement_ids": list(self.candidate_measurement_ids),
            "selected_measurement_id": self.selected_measurement_id,
        }


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    arm: StudyArm
    measurement_id: str
    quality_increase: float
    latency_gain_ratio: float
    size_gain_ratio: float
    quality_goal_met: bool
    latency_goal_met: bool
    size_goal_met: bool

    def __post_init__(self) -> None:
        if not self.measurement_id.startswith("measurement_") or any(
            not math.isfinite(value)
            for value in (
                self.quality_increase,
                self.latency_gain_ratio,
                self.size_gain_ratio,
            )
        ):
            raise IterativeSearchStudyError("study-arm outcome is invalid")

    @property
    def all_goals_met(self) -> bool:
        return self.quality_goal_met and self.latency_goal_met and self.size_goal_met

    def to_record(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "measurement_id": self.measurement_id,
            "quality_increase": self.quality_increase,
            "latency_gain_ratio": self.latency_gain_ratio,
            "size_gain_ratio": self.size_gain_ratio,
            "quality_goal_met": self.quality_goal_met,
            "latency_goal_met": self.latency_goal_met,
            "size_goal_met": self.size_goal_met,
            "all_goals_met": self.all_goals_met,
        }


@dataclass(frozen=True, slots=True)
class BackendStudy:
    backend: StudyBackend
    quality_metric: str
    goals: SearchGoals
    baseline_measurement_id: str
    generations: tuple[SearchGeneration, ...]
    outcomes: tuple[ArmOutcome, ...]
    measurements: tuple[StudyMeasurement, ...]

    def __post_init__(self) -> None:
        if not self.quality_metric:
            raise IterativeSearchStudyError("backend quality metric is required")
        by_id = {item.measurement_id: item for item in self.measurements}
        if len(by_id) != len(self.measurements) or self.baseline_measurement_id not in by_id:
            raise IterativeSearchStudyError("backend measurements must be unique with a baseline")
        if any(item.backend is not self.backend for item in self.measurements):
            raise IterativeSearchStudyError("measurement backend does not match study")
        baseline = by_id[self.baseline_measurement_id]
        if baseline.removed_channels:
            raise IterativeSearchStudyError("backend baseline cannot remove channels")
        if tuple(item.generation for item in self.generations) != tuple(
            range(1, len(self.generations) + 1)
        ):
            raise IterativeSearchStudyError("search generations must be contiguous")
        for generation in self.generations:
            referenced = {
                generation.parent_measurement_id,
                *generation.candidate_measurement_ids,
            }
            if not referenced <= set(by_id):
                raise IterativeSearchStudyError("generation references unknown measurements")
        expected_parents = (
            self.baseline_measurement_id,
            *(item.selected_measurement_id for item in self.generations[:-1]),
        )
        if tuple(item.parent_measurement_id for item in self.generations) != expected_parents:
            raise IterativeSearchStudyError("generation lineage must follow prior selections")
        if {item.arm for item in self.outcomes} != set(StudyArm) or len(self.outcomes) != 3:
            raise IterativeSearchStudyError("backend must compare exactly three study arms")
        if any(item.measurement_id not in by_id for item in self.outcomes):
            raise IterativeSearchStudyError("study arm references an unknown measurement")
        for outcome in self.outcomes:
            expected = compare_arm(outcome.arm, baseline, by_id[outcome.measurement_id], self.goals)
            if outcome != expected:
                raise IterativeSearchStudyError("study-arm outcome disagrees with measurements")
        no_repair = next(item for item in self.outcomes if item.arm is StudyArm.NO_REPAIR)
        if (
            not self.generations
            or no_repair.measurement_id
            != self.generations[-1].selected_measurement_id
        ):
            raise IterativeSearchStudyError("no-repair arm must be the final iterative selection")

    @property
    def total_wall_seconds(self) -> float:
        return math.fsum(item.wall_seconds for item in self.measurements)

    @property
    def total_gpu_seconds(self) -> float:
        return math.fsum(item.gpu_seconds for item in self.measurements)

    @property
    def total_cpu_seconds(self) -> float:
        return math.fsum(item.cpu_seconds for item in self.measurements)

    def to_record(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "quality_metric": self.quality_metric,
            "goals": self.goals.to_record(),
            "baseline_measurement_id": self.baseline_measurement_id,
            "generations": [item.to_record() for item in self.generations],
            "outcomes": [item.to_record() for item in self.outcomes],
            "measurements": [item.to_record() for item in self.measurements],
            "cost": {
                "wall_seconds": self.total_wall_seconds,
                "gpu_seconds": self.total_gpu_seconds,
                "cpu_seconds": self.total_cpu_seconds,
            },
        }


@dataclass(frozen=True, slots=True)
class IterativeSearchStudy:
    seed: int
    backends: tuple[BackendStudy, ...]
    schema_version: int = ITERATIVE_SEARCH_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.seed < 0 or self.schema_version != ITERATIVE_SEARCH_STUDY_SCHEMA_VERSION:
            raise IterativeSearchStudyError("study seed or schema version is invalid")
        if {item.backend for item in self.backends} != set(StudyBackend) or len(self.backends) != 2:
            raise IterativeSearchStudyError("study requires HF and native-GGUF backends")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "backends": [item.to_record() for item in self.backends],
            "total_experiment_cost": {
                "wall_seconds": math.fsum(item.total_wall_seconds for item in self.backends),
                "gpu_seconds": math.fsum(item.total_gpu_seconds for item in self.backends),
                "cpu_seconds": math.fsum(item.total_cpu_seconds for item in self.backends),
            },
        }


def compare_arm(
    arm: StudyArm,
    baseline: StudyMeasurement,
    candidate: StudyMeasurement,
    goals: SearchGoals,
) -> ArmOutcome:
    """Compare one measured final arm with its same-backend immutable baseline."""

    if baseline.backend is not candidate.backend:
        raise IterativeSearchStudyError("arm comparison backends must match")
    quality_increase = candidate.quality_value - baseline.quality_value
    latency_gain = 1.0 - candidate.median_latency_seconds / baseline.median_latency_seconds
    size_gain = 1.0 - candidate.size_bytes / baseline.size_bytes
    return ArmOutcome(
        arm,
        candidate.measurement_id,
        quality_increase,
        latency_gain,
        size_gain,
        quality_increase <= goals.max_quality_increase,
        latency_gain >= goals.min_latency_gain_ratio,
        size_gain >= goals.min_size_gain_ratio,
    )
