"""Immutable empirical hardware and runtime capability profiles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.hardware import HardwareInventory, collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.runtime_telemetry import HardwareNormalizationContext

HARDWARE_PROFILE_SCHEMA_VERSION = 1
MIN_PROBE_REPETITIONS = 7
MIN_PROBE_WARMUPS = 2


class HardwareProfileError(ValueError):
    """Raised when a profile is incomplete, drifting, or unsafe to compare."""


class HardwareTargetClass(StrEnum):
    CPU_ONLY = "cpu_only"
    LOW_VRAM = "low_vram"
    LAPTOP = "laptop"
    WORKSTATION = "workstation"


class ProbeKind(StrEnum):
    BANDWIDTH = "bandwidth"
    TRANSFER = "transfer"
    GEMM = "gemm"
    MODEL_LOAD = "model_load"
    LLAMA_CPP_OFFLOAD = "llama_cpp_offload"


class ProbeOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareProfileError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareProfileError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HardwareProfileError(f"{label} must be finite")
    return result


def _optional_nonnegative(value: float | None, label: str) -> None:
    if value is not None and (_finite(value, label) < 0):
        raise HardwareProfileError(f"{label} must be non-negative when available")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class RuntimeRevision:
    """Version evidence for a runtime, including explicit unavailable values."""

    name: str
    version: str | None
    revision: str | None
    source: str

    def __post_init__(self) -> None:
        _text(self.name, "runtime name")
        _text(self.source, "runtime version source")
        if self.version is None and self.revision is None and self.source != "unavailable":
            raise HardwareProfileError("unavailable runtimes must use source='unavailable'")

    def to_record(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "version": self.version,
            "revision": self.revision,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class HardwareProfileSettings:
    """Settings that can materially change probe results; unknowns remain null."""

    cpu_threads: int
    llama_cpp_threads: int
    gpu_device_index: int | None = None
    gpu_layers: int | None = None
    power_limit_watts: float | None = None
    thermal_policy: str | None = None

    def __post_init__(self) -> None:
        if self.cpu_threads <= 0 or self.llama_cpp_threads <= 0:
            raise HardwareProfileError("CPU and llama.cpp thread settings must be positive")
        for value, label in (
            (self.gpu_device_index, "GPU device index"),
            (self.gpu_layers, "GPU layers"),
        ):
            if value is not None and value < 0:
                raise HardwareProfileError(f"{label} must be non-negative when available")
        _optional_nonnegative(self.power_limit_watts, "power limit")
        if self.thermal_policy is not None:
            _text(self.thermal_policy, "thermal policy")

    def to_record(self) -> dict[str, object]:
        return {
            "cpu_threads": self.cpu_threads,
            "llama_cpp_threads": self.llama_cpp_threads,
            "gpu_device_index": self.gpu_device_index,
            "gpu_layers": self.gpu_layers,
            "power_limit_watts": self.power_limit_watts,
            "thermal_policy": self.thermal_policy,
        }


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """One bounded probe definition in the preregistered protocol."""

    kind: ProbeKind
    name: str
    unit: str
    max_seconds: float

    def __post_init__(self) -> None:
        _text(self.name, "probe name")
        _text(self.unit, "probe unit")
        if self.max_seconds <= 0:
            raise HardwareProfileError("probe time budget must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "unit": self.unit,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True, slots=True)
class HardwareProbeProtocol:
    """Warmups, repetitions, and bounded probe definitions."""

    warmups: int
    repetitions: int
    specs: tuple[ProbeSpec, ...]
    protocol_revision: str
    schema_version: int = HARDWARE_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HARDWARE_PROFILE_SCHEMA_VERSION:
            raise HardwareProfileError("unsupported hardware profile schema version")
        if self.warmups < MIN_PROBE_WARMUPS or self.repetitions < MIN_PROBE_REPETITIONS:
            raise HardwareProfileError("hardware probes require two warmups and seven repetitions")
        if not self.specs or tuple(item.kind for item in self.specs) != tuple(
            sorted({item.kind for item in self.specs}, key=lambda item: item.value)
        ):
            raise HardwareProfileError("probe specs must be unique and canonically ordered")
        if {item.kind for item in self.specs} != set(ProbeKind):
            raise HardwareProfileError(
                "protocol must define bandwidth, transfer, GEMM, load, and offload probes"
            )
        _text(self.protocol_revision, "protocol revision")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "specs": [item.to_record() for item in self.specs],
            "protocol_revision": self.protocol_revision,
        }

    @property
    def protocol_id(self) -> str:
        digest = hashlib.sha256(canonical_identity_json(self.to_record()).encode()).hexdigest()
        return f"hardware_protocol_{digest}"


@dataclass(frozen=True, slots=True)
class ProbeSample:
    """One probe observation with optional sensor fields."""

    value: float
    hardware_context_id: str
    environment_fingerprint: str | None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    thermal_celsius: float | None = None
    clock_mhz: float | None = None
    power_watts: float | None = None

    def __post_init__(self) -> None:
        value = _finite(self.value, "probe sample")
        if value < 0:
            raise HardwareProfileError("probe samples must be non-negative")
        _text(self.hardware_context_id, "probe hardware context ID")
        if self.environment_fingerprint is not None:
            _text(self.environment_fingerprint, "environment fingerprint")
        for memory_value, label in (
            (self.peak_ram_bytes, "peak RAM"),
            (self.peak_vram_bytes, "peak VRAM"),
        ):
            if memory_value is not None and memory_value < 0:
                raise HardwareProfileError(f"{label} must be non-negative")
        for sensor_value, label in (
            (self.thermal_celsius, "thermal temperature"),
            (self.clock_mhz, "clock"),
            (self.power_watts, "power"),
        ):
            _optional_nonnegative(sensor_value, label)

    def to_record(self) -> dict[str, object]:
        return {
            "value": self.value,
            "hardware_context_id": self.hardware_context_id,
            "environment_fingerprint": self.environment_fingerprint,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "thermal_celsius": self.thermal_celsius,
            "clock_mhz": self.clock_mhz,
            "power_watts": self.power_watts,
        }


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    median: float
    p95: float
    dispersion: float
    repetitions: int

    def to_record(self) -> dict[str, float | int]:
        return {
            "median": self.median,
            "p95": self.p95,
            "dispersion": self.dispersion,
            "repetitions": self.repetitions,
        }


def summarize_probe_samples(samples: tuple[ProbeSample, ...]) -> ProbeSummary:
    """Compute deterministic median, rank-based p95, and population dispersion."""

    if len(samples) < MIN_PROBE_REPETITIONS:
        raise HardwareProfileError("measured probes require at least seven repetitions")
    values = sorted(sample.value for sample in samples)
    median = values[(len(values) - 1) // 2]
    p95 = values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)]
    mean = sum(values) / len(values)
    dispersion = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return ProbeSummary(median, p95, dispersion, len(values))


@dataclass(frozen=True, slots=True)
class HardwareProbeResult:
    """A measured or explicit terminal outcome for one protocol probe."""

    spec: ProbeSpec
    hardware_context_id: str
    outcome: ProbeOutcome
    samples: tuple[ProbeSample, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.hardware_context_id, "probe hardware context ID")
        if self.outcome is ProbeOutcome.MEASURED:
            if self.reason is not None:
                raise HardwareProfileError("measured probes cannot carry a reason")
            if len(self.samples) < MIN_PROBE_REPETITIONS:
                raise HardwareProfileError("measured probes require at least seven repetitions")
            if any(
                sample.hardware_context_id != self.hardware_context_id for sample in self.samples
            ):
                raise HardwareProfileError(
                    "probe samples changed hardware context during measurement"
                )
            fingerprints = {sample.environment_fingerprint for sample in self.samples}
            if None in fingerprints and len(fingerprints) > 1:
                raise HardwareProfileError(
                    "environment fingerprint availability changed during measurement"
                )
            if len(fingerprints - {None}) > 1:
                raise HardwareProfileError(
                    "materially drifting environment detected during measurement"
                )
        else:
            if self.samples:
                raise HardwareProfileError("non-measured probes cannot carry samples")
            _text(self.reason or "", "probe outcome reason")

    @property
    def summary(self) -> ProbeSummary | None:
        return (
            summarize_probe_samples(self.samples) if self.outcome is ProbeOutcome.MEASURED else None
        )

    @property
    def probe_id(self) -> str:
        identity = {
            "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
            "spec": self.spec.to_record(),
            "hardware_context_id": self.hardware_context_id,
            "outcome": self.outcome.value,
            "samples": [item.to_record() for item in self.samples],
            "reason": self.reason,
        }
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"hardware_probe_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "spec": self.spec.to_record(),
            "hardware_context_id": self.hardware_context_id,
            "outcome": self.outcome.value,
            "samples": [item.to_record() for item in self.samples],
            "summary": None if self.summary is None else self.summary.to_record(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Versioned host profile suitable for grouping empirical measurements."""

    profile_name: str
    target_class: HardwareTargetClass
    inventory: HardwareInventory
    hardware_context: HardwareNormalizationContext
    runtimes: tuple[RuntimeRevision, ...]
    settings: HardwareProfileSettings
    protocol: HardwareProbeProtocol
    probes: tuple[HardwareProbeResult, ...]
    schema_version: int = HARDWARE_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.profile_name, "hardware profile name")
        if self.schema_version != HARDWARE_PROFILE_SCHEMA_VERSION:
            raise HardwareProfileError("unsupported hardware profile schema version")
        if (
            self.hardware_context.context_id
            != HardwareNormalizationContext.from_inventory(self.inventory).context_id
        ):
            raise HardwareProfileError("hardware context does not match inventory")
        if not self.runtimes or len({item.name for item in self.runtimes}) != len(self.runtimes):
            raise HardwareProfileError("profile runtime revisions must be uniquely named")
        if len(self.probes) != len(set(item.spec.kind for item in self.probes)):
            raise HardwareProfileError("profile probes must be unique by kind")
        if {item.spec.kind for item in self.probes} != set(ProbeKind):
            raise HardwareProfileError("profile must retain every protocol probe outcome")
        if any(
            item.hardware_context_id != self.hardware_context.context_id for item in self.probes
        ):
            raise HardwareProfileError("probe context does not match profile hardware context")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "target_class": self.target_class.value,
            "inventory": self.inventory.to_record(),
            "hardware_context": self.hardware_context.to_record(),
            "runtimes": [item.to_record() for item in self.runtimes],
            "settings": self.settings.to_record(),
            "protocol": self.protocol.to_record(),
            "probes": [item.to_record() for item in self.probes],
        }

    @property
    def profile_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"hardware_profile_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["profile_id"] = self.profile_id
        record["protocol_id"] = self.protocol.protocol_id
        return record


DEFAULT_PROBE_SPECS = (
    ProbeSpec(ProbeKind.BANDWIDTH, "memory_bandwidth", "GB/s", 60.0),
    ProbeSpec(ProbeKind.GEMM, "representative_gemm", "TFLOP/s", 120.0),
    ProbeSpec(ProbeKind.LLAMA_CPP_OFFLOAD, "llama_cpp_offload", "tokens/s", 300.0),
    ProbeSpec(ProbeKind.MODEL_LOAD, "model_load", "GB/s", 300.0),
    ProbeSpec(ProbeKind.TRANSFER, "host_device_transfer", "GB/s", 60.0),
)
DEFAULT_HARDWARE_PROBE_PROTOCOL = HardwareProbeProtocol(
    MIN_PROBE_WARMUPS,
    MIN_PROBE_REPETITIONS,
    DEFAULT_PROBE_SPECS,
    "modelsurgeon-v1.3-hardware-profile-v1",
)


def build_hardware_profile(
    *,
    profile_name: str,
    target_class: HardwareTargetClass,
    inventory: HardwareInventory,
    runtimes: tuple[RuntimeRevision, ...],
    settings: HardwareProfileSettings,
    probes: tuple[HardwareProbeResult, ...],
    protocol: HardwareProbeProtocol = DEFAULT_HARDWARE_PROBE_PROTOCOL,
) -> HardwareProfile:
    """Build a profile after deriving the canonical context from inventory."""

    return HardwareProfile(
        profile_name,
        target_class,
        inventory,
        HardwareNormalizationContext.from_inventory(inventory),
        runtimes,
        settings,
        protocol,
        probes,
    )


def build_default_hardware_profile(disk_path: str = ".") -> HardwareProfile:
    """Collect a CPU-first profile with explicit unknown probe outcomes."""

    inventory = collect_hardware_inventory(disk_path)
    context_id = HardwareNormalizationContext.from_inventory(inventory).context_id
    target_class = (
        HardwareTargetClass.WORKSTATION if inventory.cuda.devices else HardwareTargetClass.CPU_ONLY
    )
    logical_cores = inventory.cpu.logical_cores or 1
    runtimes = (
        RuntimeRevision(
            "modelsurgeon",
            inventory.software.modelsurgeon_version,
            None,
            "package" if inventory.software.modelsurgeon_version else "unavailable",
        ),
        RuntimeRevision(
            "pytorch",
            inventory.software.pytorch_version,
            inventory.cuda.compiled_version,
            "package" if inventory.software.pytorch_version else "unavailable",
        ),
        RuntimeRevision("llama.cpp", None, None, "unavailable"),
    )
    probes = tuple(
        HardwareProbeResult(
            spec,
            context_id,
            ProbeOutcome.UNKNOWN,
            reason="probe runner was not invoked; no measurement is inferred",
        )
        for spec in DEFAULT_PROBE_SPECS
    )
    return build_hardware_profile(
        profile_name="local-default",
        target_class=target_class,
        inventory=inventory,
        runtimes=runtimes,
        settings=HardwareProfileSettings(logical_cores, logical_cores),
        probes=probes,
    )


def render_hardware_profile_protocol(*, format: str = "json") -> str:
    """Render the preregistered probe protocol deterministically."""

    payload: Mapping[str, object] = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "minimum_warmups": MIN_PROBE_WARMUPS,
        "minimum_repetitions": MIN_PROBE_REPETITIONS,
        "protocol": DEFAULT_HARDWARE_PROBE_PROTOCOL.to_record(),
        "outcomes": [item.value for item in ProbeOutcome],
    }
    if format == "json":
        return canonical_identity_json(payload) + "\n"
    if format == "markdown":
        lines = [
            "# Empirical hardware profile protocol",
            "",
            "Minimum warmups: 2",
            "",
            "Minimum repetitions: 7",
            "",
            "| Probe | Unit | Budget (s) |",
            "| --- | --- | ---: |",
        ]
        lines.extend(
            f"| `{item.name}` | `{item.unit}` | {item.max_seconds:g} |"
            for item in DEFAULT_PROBE_SPECS
        )
        return "\n".join(lines) + "\n"
    raise HardwareProfileError("hardware profile protocol format must be json or markdown")
