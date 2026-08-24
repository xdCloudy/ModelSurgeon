"""Consumer-hardware scale-study contracts and bounded defaults."""

from __future__ import annotations

from dataclasses import dataclass


class ScaleStudyError(ValueError):
    """Raised when scale-study measurements or hardware inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ScaleDefault:
    execution_device: str
    memory_mode: str
    calibration_tokens: int
    requires_quantization: bool
    rationale: str

    def to_record(self) -> dict[str, object]:
        return {
            "execution_device": self.execution_device,
            "memory_mode": self.memory_mode,
            "calibration_tokens": self.calibration_tokens,
            "requires_quantization": self.requires_quantization,
            "rationale": self.rationale,
        }


def choose_scale_default(
    parameter_count: int,
    *,
    accelerator_memory_bytes: int | None,
) -> ScaleDefault:
    """Return a conservative reference default for one model scale.

    Thresholds intentionally describe a 12 GiB consumer accelerator rather than
    pretending that nominal weight bytes are the complete runtime footprint.
    """

    if parameter_count <= 0:
        raise ScaleStudyError("parameter count must be positive")
    if accelerator_memory_bytes is not None and accelerator_memory_bytes <= 0:
        raise ScaleStudyError("accelerator memory must be positive when present")
    # Consumer cards sold as 12 GB can report slightly less than 12 GiB after
    # firmware reservations, so classify from an 11 GiB usable-memory floor.
    has_12_gib = accelerator_memory_bytes is not None and accelerator_memory_bytes >= 11 << 30
    if has_12_gib and parameter_count <= 500_000_000:
        return ScaleDefault(
            "accelerator",
            "full_tensor",
            64,
            False,
            "FP16 weights and short calibration batches retain ample 12 GiB headroom",
        )
    if has_12_gib and parameter_count <= 1_600_000_000:
        return ScaleDefault(
            "accelerator",
            "tensor",
            32,
            False,
            "process one affected tensor at a time to bound activation and mutation peaks",
        )
    if has_12_gib and parameter_count <= 4_100_000_000:
        return ScaleDefault(
            "accelerator",
            "streaming",
            16,
            True,
            "quantized weights and streaming leave headroom unavailable to full FP16",
        )
    return ScaleDefault(
        "cpu",
        "streaming",
        8,
        parameter_count > 1_600_000_000,
        "CPU streaming is the safe fallback; use quantized accelerator evaluation when available",
    )
