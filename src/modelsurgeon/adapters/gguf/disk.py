"""Conservative GGUF output and scratch-disk preflight checks."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class GGUFDiskSpaceError(OSError):
    """Raised before writing when a target filesystem cannot satisfy the estimate."""


def _bytes(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} byte count")


@dataclass(frozen=True, slots=True)
class GGUFDiskEstimate:
    """Worst-case physical allocation estimate before sparse/copy optimizations."""

    output_bytes: int
    scratch_bytes: int
    alignment_bytes: int = 32
    safety_margin_bytes: int = 0

    def __post_init__(self) -> None:
        _bytes("output", self.output_bytes)
        _bytes("scratch", self.scratch_bytes)
        _bytes("alignment", self.alignment_bytes, positive=True)
        _bytes("safety margin", self.safety_margin_bytes)
        if self.alignment_bytes & (self.alignment_bytes - 1):
            raise ValueError("alignment must be a power of two")

    @property
    def aligned_output_bytes(self) -> int:
        mask = self.alignment_bytes - 1
        return (self.output_bytes + mask) & ~mask

    @property
    def alignment_padding_bytes(self) -> int:
        return self.aligned_output_bytes - self.output_bytes


@dataclass(frozen=True, slots=True)
class DiskProbe:
    """Available bytes and stable identity for one filesystem."""

    device_id: str
    free_bytes: int

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("disk device identity cannot be empty")
        _bytes("free disk", self.free_bytes)


DiskProbeFunction = Callable[[Path], DiskProbe]


@dataclass(frozen=True, slots=True)
class GGUFFilesystemRequirement:
    device_id: str
    probe_path: Path
    required_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class GGUFDiskPlan:
    """Successful preflight record suitable for operation provenance."""

    output_path: Path
    scratch_path: Path
    estimate: GGUFDiskEstimate
    filesystems: tuple[GGUFFilesystemRequirement, ...]


def _existing_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise GGUFDiskSpaceError(f"no existing parent for disk target {path}")
        candidate = parent
    return candidate


def _system_probe(path: Path) -> DiskProbe:
    existing = _existing_path(path)
    return DiskProbe(str(os.stat(existing).st_dev), shutil.disk_usage(existing).free)


def _requirements(
    output_path: Path,
    scratch_path: Path,
    output_bytes: int,
    scratch_bytes: int,
    safety_margin_bytes: int,
    probe: DiskProbeFunction,
) -> tuple[GGUFFilesystemRequirement, ...]:
    targets = (
        ("output", output_path.parent, output_bytes),
        ("scratch", scratch_path, scratch_bytes),
    )
    grouped: dict[str, tuple[Path, int, int]] = {}
    for _, path, required in targets:
        disk = probe(_existing_path(path))
        if disk.device_id in grouped:
            first_path, existing_required, existing_free = grouped[disk.device_id]
            grouped[disk.device_id] = (
                first_path,
                existing_required + required,
                min(existing_free, disk.free_bytes),
            )
        else:
            grouped[disk.device_id] = (path, required, disk.free_bytes)
    return tuple(
        GGUFFilesystemRequirement(
            device_id,
            path,
            required + safety_margin_bytes,
            free,
        )
        for device_id, (path, required, free) in sorted(grouped.items())
    )


def _assert_capacity(requirements: tuple[GGUFFilesystemRequirement, ...]) -> None:
    failures = tuple(item for item in requirements if item.required_bytes > item.free_bytes)
    if failures:
        details = "; ".join(
            f"{item.device_id} requires {item.required_bytes} bytes but has "
            f"{item.free_bytes} bytes free"
            for item in failures
        )
        raise GGUFDiskSpaceError(f"insufficient GGUF output/scratch space: {details}")


def preflight_gguf_disk(
    output_path: str | Path,
    scratch_path: str | Path,
    estimate: GGUFDiskEstimate,
    *,
    probe: DiskProbeFunction = _system_probe,
) -> GGUFDiskPlan:
    """Validate conservative output and scratch allocations before any write."""

    output = Path(output_path).resolve()
    scratch = Path(scratch_path).resolve()
    requirements = _requirements(
        output,
        scratch,
        estimate.aligned_output_bytes,
        estimate.scratch_bytes,
        estimate.safety_margin_bytes,
        probe,
    )
    _assert_capacity(requirements)
    return GGUFDiskPlan(output, scratch, estimate, requirements)


def monitor_gguf_disk(
    plan: GGUFDiskPlan,
    *,
    output_remaining_bytes: int,
    scratch_remaining_bytes: int,
    probe: DiskProbeFunction = _system_probe,
) -> tuple[GGUFFilesystemRequirement, ...]:
    """Recheck remaining worst-case allocations while output construction proceeds."""

    _bytes("remaining output", output_remaining_bytes)
    _bytes("remaining scratch", scratch_remaining_bytes)
    requirements = _requirements(
        plan.output_path,
        plan.scratch_path,
        output_remaining_bytes,
        scratch_remaining_bytes,
        plan.estimate.safety_margin_bytes,
        probe,
    )
    _assert_capacity(requirements)
    return requirements
