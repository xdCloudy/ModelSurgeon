"""Memory-bounded calibration batch planning with exact sample-boundary resume."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.resource_budget import (
    ResourceBudgetExceeded,
    ResourceKind,
    StageResourceBudget,
    StageResourceEstimate,
    preflight_resource_budget,
)

CALIBRATION_BATCH_PLANNER_VERSION = "1"


class CalibrationBatchPlanningError(ValueError):
    """Raised when calibration batching cannot preserve its bounded-resume contract."""


class CalibrationSampleIdentityLike(Protocol):
    sample_id: str
    content_sha256: str


class TokenizedCalibrationSampleLike(Protocol):
    identity: CalibrationSampleIdentityLike
    input_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalibrationBatchMemoryModel:
    fixed_ram_bytes: int = 0
    ram_bytes_per_sample: int = 0
    ram_bytes_per_token: int = 0
    fixed_vram_bytes: int = 0
    vram_bytes_per_sample: int = 0
    vram_bytes_per_token: int = 0

    def __post_init__(self) -> None:
        values = (
            self.fixed_ram_bytes,
            self.ram_bytes_per_sample,
            self.ram_bytes_per_token,
            self.fixed_vram_bytes,
            self.vram_bytes_per_sample,
            self.vram_bytes_per_token,
        )
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise CalibrationBatchPlanningError("batch memory model values must be non-negative")

    def estimate(self, sample_count: int, token_count: int) -> StageResourceEstimate:
        if sample_count <= 0 or token_count <= 0:
            raise CalibrationBatchPlanningError(
                "batch memory estimates require positive sample and token counts"
            )
        return StageResourceEstimate(
            ram_bytes=(
                self.fixed_ram_bytes
                + sample_count * self.ram_bytes_per_sample
                + token_count * self.ram_bytes_per_token
            ),
            vram_bytes=(
                self.fixed_vram_bytes
                + sample_count * self.vram_bytes_per_sample
                + token_count * self.vram_bytes_per_token
            ),
        )

    def to_record(self) -> dict[str, int]:
        return {
            "fixed_ram_bytes": self.fixed_ram_bytes,
            "ram_bytes_per_sample": self.ram_bytes_per_sample,
            "ram_bytes_per_token": self.ram_bytes_per_token,
            "fixed_vram_bytes": self.fixed_vram_bytes,
            "vram_bytes_per_sample": self.vram_bytes_per_sample,
            "vram_bytes_per_token": self.vram_bytes_per_token,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBatchPlannerConfig:
    max_batch_size: int
    max_batch_tokens: int
    resource_budget: StageResourceBudget
    memory_model: CalibrationBatchMemoryModel = CalibrationBatchMemoryModel()
    min_batch_size: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("max_batch_size", self.max_batch_size),
            ("max_batch_tokens", self.max_batch_tokens),
            ("min_batch_size", self.min_batch_size),
        ):
            if isinstance(value, bool) or value <= 0:
                raise CalibrationBatchPlanningError(f"{name} must be a positive integer")
        if self.min_batch_size > self.max_batch_size:
            raise CalibrationBatchPlanningError("minimum batch size cannot exceed maximum")


@dataclass(frozen=True, slots=True)
class CalibrationBatchCursor:
    manifest_digest: str
    next_sample_index: int

    def __post_init__(self) -> None:
        if len(self.manifest_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.manifest_digest
        ):
            raise CalibrationBatchPlanningError("batch cursor requires a SHA-256 manifest digest")
        if isinstance(self.next_sample_index, bool) or self.next_sample_index < 0:
            raise CalibrationBatchPlanningError("batch cursor sample index cannot be negative")

    def to_record(self) -> dict[str, str | int]:
        return {
            "manifest_digest": self.manifest_digest,
            "next_sample_index": self.next_sample_index,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBatchObservation:
    manifest_digest: str
    start_index: int
    end_index: int
    token_count: int
    ram_baseline_bytes: int | None = None
    ram_peak_bytes: int | None = None
    vram_baseline_bytes: int | None = None
    vram_peak_bytes: int | None = None
    exhausted_resource: ResourceKind | None = None

    def __post_init__(self) -> None:
        CalibrationBatchCursor(self.manifest_digest, self.start_index)
        if self.end_index <= self.start_index:
            raise CalibrationBatchPlanningError(
                "batch observation requires a non-empty sample range"
            )
        if isinstance(self.token_count, bool) or self.token_count <= 0:
            raise CalibrationBatchPlanningError("batch observation requires a positive token count")
        for label, baseline, peak in (
            ("RAM", self.ram_baseline_bytes, self.ram_peak_bytes),
            ("VRAM", self.vram_baseline_bytes, self.vram_peak_bytes),
        ):
            if (baseline is None) != (peak is None):
                raise CalibrationBatchPlanningError(
                    f"{label} telemetry requires both baseline and peak bytes"
                )
            if baseline is not None and (baseline < 0 or peak is None or peak < baseline):
                raise CalibrationBatchPlanningError(
                    f"{label} telemetry peak must be at least its non-negative baseline"
                )
        if self.exhausted_resource not in {None, ResourceKind.RAM, ResourceKind.VRAM}:
            raise CalibrationBatchPlanningError(
                "batch adaptation only accepts RAM or VRAM exhaustion observations"
            )

    @property
    def sample_count(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    start_index: int
    end_index: int
    sample_ids: tuple[str, ...]
    token_count: int
    estimated_ram_bytes: int
    estimated_vram_bytes: int

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise CalibrationBatchPlanningError("calibration batch requires a non-empty range")
        if self.end_index - self.start_index != len(self.sample_ids):
            raise CalibrationBatchPlanningError("calibration batch range and sample IDs disagree")
        if len(self.sample_ids) != len(set(self.sample_ids)) or any(
            not sample_id for sample_id in self.sample_ids
        ):
            raise CalibrationBatchPlanningError("calibration batch sample IDs must be unique")
        if self.token_count <= 0:
            raise CalibrationBatchPlanningError("calibration batch token count must be positive")
        if self.estimated_ram_bytes < 0 or self.estimated_vram_bytes < 0:
            raise CalibrationBatchPlanningError(
                "calibration batch memory estimates cannot be negative"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sample_ids": list(self.sample_ids),
            "token_count": self.token_count,
            "estimated_ram_bytes": self.estimated_ram_bytes,
            "estimated_vram_bytes": self.estimated_vram_bytes,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBatchPlan:
    manifest_digest: str
    cursor: CalibrationBatchCursor
    next_cursor: CalibrationBatchCursor
    batch: CalibrationBatch | None
    effective_max_batch_size: int
    memory_model: CalibrationBatchMemoryModel
    version: str = CALIBRATION_BATCH_PLANNER_VERSION

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_BATCH_PLANNER_VERSION:
            raise CalibrationBatchPlanningError(
                f"unsupported calibration batch planner version {self.version}"
            )
        if self.cursor.manifest_digest != self.manifest_digest:
            raise CalibrationBatchPlanningError("batch plan cursor does not match manifest")
        if self.next_cursor.manifest_digest != self.manifest_digest:
            raise CalibrationBatchPlanningError("batch plan next cursor does not match manifest")
        if self.effective_max_batch_size <= 0:
            raise CalibrationBatchPlanningError("effective batch size must be positive")
        expected_next = (
            self.cursor.next_sample_index if self.batch is None else self.batch.end_index
        )
        if self.next_cursor.next_sample_index != expected_next:
            raise CalibrationBatchPlanningError("batch plan next cursor is not an exact boundary")
        if self.batch is not None and self.batch.start_index != self.cursor.next_sample_index:
            raise CalibrationBatchPlanningError("batch does not start at the resume cursor")

    @property
    def complete(self) -> bool:
        return self.batch is None

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "manifest_digest": self.manifest_digest,
            "cursor": self.cursor.to_record(),
            "next_cursor": self.next_cursor.to_record(),
            "complete": self.complete,
            "effective_max_batch_size": self.effective_max_batch_size,
            "memory_model": self.memory_model.to_record(),
            "batch": None if self.batch is None else self.batch.to_record(),
        }


def calibration_manifest_digest(samples: Sequence[TokenizedCalibrationSampleLike]) -> str:
    """Hash the exact ordered selected samples and tokens used for batching/resume."""

    payload: list[dict[str, object]] = []
    seen: set[str] = set()
    for sample in samples:
        sample_id = sample.identity.sample_id
        if not sample_id or sample_id in seen:
            raise CalibrationBatchPlanningError(
                "calibration batching requires unique non-empty sample IDs"
            )
        seen.add(sample_id)
        if not sample.identity.content_sha256:
            raise CalibrationBatchPlanningError("calibration sample content digest is required")
        if not sample.input_ids:
            raise CalibrationBatchPlanningError(
                f"calibration sample {sample_id!r} has no tokens and cannot be batched"
            )
        payload.append(
            {
                "sample_id": sample_id,
                "content_sha256": sample.identity.content_sha256,
                "input_ids": list(sample.input_ids),
            }
        )
    encoded = canonical_identity_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measured_per_token(
    baseline: int | None,
    peak: int | None,
    *,
    fixed_bytes: int,
    per_sample_bytes: int,
    sample_count: int,
    token_count: int,
) -> int | None:
    if baseline is None or peak is None:
        return None
    variable = max(0, peak - baseline - fixed_bytes - per_sample_bytes * sample_count)
    return math.ceil(variable / token_count)


def adapt_calibration_memory_model(
    model: CalibrationBatchMemoryModel,
    observations: Sequence[CalibrationBatchObservation],
    manifest_digest: str,
) -> CalibrationBatchMemoryModel:
    """Raise per-token estimates to the maximum measured stage-local memory usage."""

    ram_per_token = model.ram_bytes_per_token
    vram_per_token = model.vram_bytes_per_token
    for observation in observations:
        if observation.manifest_digest != manifest_digest:
            raise CalibrationBatchPlanningError(
                "batch telemetry belongs to a different calibration manifest"
            )
        measured_ram = _measured_per_token(
            observation.ram_baseline_bytes,
            observation.ram_peak_bytes,
            fixed_bytes=model.fixed_ram_bytes,
            per_sample_bytes=model.ram_bytes_per_sample,
            sample_count=observation.sample_count,
            token_count=observation.token_count,
        )
        measured_vram = _measured_per_token(
            observation.vram_baseline_bytes,
            observation.vram_peak_bytes,
            fixed_bytes=model.fixed_vram_bytes,
            per_sample_bytes=model.vram_bytes_per_sample,
            sample_count=observation.sample_count,
            token_count=observation.token_count,
        )
        if measured_ram is not None:
            ram_per_token = max(ram_per_token, measured_ram)
        if measured_vram is not None:
            vram_per_token = max(vram_per_token, measured_vram)
    return CalibrationBatchMemoryModel(
        model.fixed_ram_bytes,
        model.ram_bytes_per_sample,
        ram_per_token,
        model.fixed_vram_bytes,
        model.vram_bytes_per_sample,
        vram_per_token,
    )


def _effective_batch_size(
    config: CalibrationBatchPlannerConfig,
    observations: Sequence[CalibrationBatchObservation],
) -> int:
    maximum = config.max_batch_size
    if observations and observations[-1].exhausted_resource is not None:
        failed_size = observations[-1].sample_count
        if failed_size <= config.min_batch_size:
            raise CalibrationBatchPlanningError(
                "memory exhaustion occurred at the configured minimum batch size"
            )
        maximum = min(maximum, max(config.min_batch_size, failed_size // 2))
    return max(config.min_batch_size, maximum)


def _fits_resources(
    config: CalibrationBatchPlannerConfig,
    hardware: HardwareInventory,
    estimate: StageResourceEstimate,
) -> bool:
    try:
        preflight_resource_budget(
            "calibration-batch",
            config.resource_budget,
            hardware,
            estimate,
        )
    except ResourceBudgetExceeded:
        return False
    return True


def _validate_observations(
    samples: Sequence[TokenizedCalibrationSampleLike],
    cursor: CalibrationBatchCursor,
    observations: Sequence[CalibrationBatchObservation],
    manifest_digest: str,
) -> None:
    for index, observation in enumerate(observations):
        if observation.manifest_digest != manifest_digest:
            raise CalibrationBatchPlanningError(
                "batch telemetry belongs to a different calibration manifest"
            )
        if observation.end_index > len(samples):
            raise CalibrationBatchPlanningError(
                "batch telemetry sample range exceeds the calibration manifest"
            )
        expected_tokens = sum(
            len(sample.input_ids)
            for sample in samples[observation.start_index : observation.end_index]
        )
        if observation.token_count != expected_tokens:
            raise CalibrationBatchPlanningError(
                "batch telemetry token count does not match its manifest sample range"
            )
        already_completed = observation.end_index <= cursor.next_sample_index
        latest_failed_here = (
            index == len(observations) - 1
            and observation.exhausted_resource is not None
            and observation.start_index == cursor.next_sample_index
        )
        if not already_completed and not latest_failed_here:
            raise CalibrationBatchPlanningError(
                "batch telemetry is ahead of the resume cursor without a current failed batch"
            )


class CalibrationBatchPlanner:
    """Plan one contiguous batch so callers can adapt again after measured telemetry."""

    def __init__(
        self,
        config: CalibrationBatchPlannerConfig,
        hardware: HardwareInventory,
    ) -> None:
        self.config = config
        self.hardware = hardware

    def plan_next(
        self,
        samples: Sequence[TokenizedCalibrationSampleLike],
        *,
        cursor: CalibrationBatchCursor | None = None,
        observations: Sequence[CalibrationBatchObservation] = (),
    ) -> CalibrationBatchPlan:
        manifest_digest = calibration_manifest_digest(samples)
        resolved_cursor = cursor or CalibrationBatchCursor(manifest_digest, 0)
        if resolved_cursor.manifest_digest != manifest_digest:
            raise CalibrationBatchPlanningError(
                "resume cursor belongs to a different calibration manifest"
            )
        if resolved_cursor.next_sample_index > len(samples):
            raise CalibrationBatchPlanningError("resume cursor is beyond the calibration manifest")
        _validate_observations(samples, resolved_cursor, observations, manifest_digest)
        adapted_model = adapt_calibration_memory_model(
            self.config.memory_model,
            observations,
            manifest_digest,
        )
        effective_max = _effective_batch_size(self.config, observations)
        start = resolved_cursor.next_sample_index
        if start == len(samples):
            return CalibrationBatchPlan(
                manifest_digest,
                resolved_cursor,
                resolved_cursor,
                None,
                effective_max,
                adapted_model,
            )

        sample_ids: list[str] = []
        token_count = 0
        final_estimate: StageResourceEstimate | None = None
        end = start
        while end < len(samples) and len(sample_ids) < effective_max:
            sample = samples[end]
            sample_tokens = len(sample.input_ids)
            proposed_tokens = token_count + sample_tokens
            if proposed_tokens > self.config.max_batch_tokens:
                if not sample_ids:
                    raise CalibrationBatchPlanningError(
                        f"sample {sample.identity.sample_id!r} exceeds max_batch_tokens without "
                        "token-splitting; reduce per-sample tokenization length"
                    )
                break
            proposed_count = len(sample_ids) + 1
            estimate = adapted_model.estimate(proposed_count, proposed_tokens)
            if not _fits_resources(self.config, self.hardware, estimate):
                if not sample_ids:
                    preflight_resource_budget(
                        "calibration-batch",
                        self.config.resource_budget,
                        self.hardware,
                        estimate,
                    )
                    raise CalibrationBatchPlanningError(
                        "unreachable: resource preflight accepted a batch rejected by planning"
                    )
                break
            sample_ids.append(sample.identity.sample_id)
            token_count = proposed_tokens
            final_estimate = estimate
            end += 1

        if not sample_ids or final_estimate is None:
            raise CalibrationBatchPlanningError("batch planner could not select a bounded sample")
        batch = CalibrationBatch(
            start,
            end,
            tuple(sample_ids),
            token_count,
            final_estimate.ram_bytes,
            final_estimate.vram_bytes,
        )
        next_cursor = CalibrationBatchCursor(manifest_digest, end)
        return CalibrationBatchPlan(
            manifest_digest,
            resolved_cursor,
            next_cursor,
            batch,
            effective_max,
            adapted_model,
        )
