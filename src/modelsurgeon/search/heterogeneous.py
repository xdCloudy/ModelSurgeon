"""Capability-gated heterogeneous architecture compilation and materialization."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.deployable_state import ArtifactContainerFormat

HETEROGENEOUS_SCHEMA_VERSION: Final[int] = 1


class HeterogeneousStateError(ValueError):
    """Raised when a heterogeneous state or evidence record is unsafe."""


class HeterogeneousOutcome(StrEnum):
    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MaterializationEvidenceStatus(StrEnum):
    COMPLETE = "complete"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HeterogeneousStateError(f"{label} is required")
    return value


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HeterogeneousStateError(f"{label} must be a positive integer")
    return value


def _fraction(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeterogeneousStateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise HeterogeneousStateError(f"{label} must be finite and within [0, 1]")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise HeterogeneousStateError(f"{label} must be a lowercase SHA-256")
    return result


_CODEC_BYTES: dict[str, float] = {
    "f32": 4.0,
    "float32": 4.0,
    "f16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "q8": 1.0,
    "q8_0": 1.0,
    "q4": 0.5,
    "q4_k": 0.5625,
    "q4_k_m": 0.5625,
    "q3": 0.4375,
    "q2": 0.3125,
}


@dataclass(frozen=True, slots=True)
class HeterogeneousLayer:
    """Effective choices for one layer, including all mixed axes."""

    layer_index: int
    attention_width: int
    mlp_width: int
    low_rank_factors: tuple[tuple[str, int], ...] = ()
    sparsity: tuple[tuple[str, float], ...] = ()
    quantization_codec: str = "f16"

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or self.layer_index < 0:
            raise HeterogeneousStateError("layer index must be non-negative")
        _positive(self.attention_width, "attention width")
        _positive(self.mlp_width, "MLP width")
        _text(self.quantization_codec, "layer quantization codec")
        rank_names = tuple(name for name, _ in self.low_rank_factors)
        if rank_names != tuple(sorted(set(rank_names))):
            raise HeterogeneousStateError("low-rank components must be unique and canonical")
        for component, rank in self.low_rank_factors:
            _text(component, "low-rank component")
            _positive(rank, "low-rank rank")
        sparsity_names = tuple(name for name, _ in self.sparsity)
        if sparsity_names != tuple(sorted(set(sparsity_names))):
            raise HeterogeneousStateError("sparsity components must be unique and canonical")
        for component, fraction in self.sparsity:
            _text(component, "sparsity component")
            _fraction(fraction, "sparsity fraction")

    @property
    def parameter_count(self) -> int:
        dense = (self.attention_width * self.attention_width) + (4 * self.mlp_width)
        low_rank = sum(
            rank * (self.attention_width + self.mlp_width) for _, rank in self.low_rank_factors
        )
        sparse_factor = math.prod(1.0 - fraction for _, fraction in self.sparsity)
        return max(1, math.ceil((dense + low_rank) * sparse_factor))

    @property
    def storage_bytes(self) -> int:
        bytes_per_parameter = _CODEC_BYTES.get(self.quantization_codec)
        if bytes_per_parameter is None:
            raise HeterogeneousStateError(
                f"no analytic storage estimate for codec {self.quantization_codec!r}"
            )
        return max(1, math.ceil(self.parameter_count * bytes_per_parameter))

    def to_record(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "attention_width": self.attention_width,
            "mlp_width": self.mlp_width,
            "low_rank_factors": [
                {"component": component, "rank": rank} for component, rank in self.low_rank_factors
            ],
            "sparsity": [
                {"component": component, "fraction": fraction}
                for component, fraction in self.sparsity
            ],
            "quantization_codec": self.quantization_codec,
        }


@dataclass(frozen=True, slots=True)
class HeterogeneousArchitectureSpec:
    model_family: str
    model_revision: str
    source_artifact_digest: str
    artifact_format: ArtifactContainerFormat
    runtime: str
    layers: tuple[HeterogeneousLayer, ...]
    source_parameter_count: int
    source_storage_bytes: int

    def __post_init__(self) -> None:
        _text(self.model_family, "model family")
        _text(self.model_revision, "model revision")
        _digest(self.source_artifact_digest, "source artifact digest")
        _text(self.runtime, "runtime")
        _positive(self.source_parameter_count, "source parameter count")
        _positive(self.source_storage_bytes, "source storage bytes")
        if not self.layers:
            raise HeterogeneousStateError("heterogeneous specifications require layers")
        indexes = tuple(layer.layer_index for layer in self.layers)
        if indexes != tuple(range(len(self.layers))):
            raise HeterogeneousStateError("layer choices must cover contiguous canonical indices")

    @property
    def requested_codecs(self) -> tuple[str, ...]:
        return tuple(sorted({layer.quantization_codec for layer in self.layers}))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HETEROGENEOUS_SCHEMA_VERSION,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "source_artifact_digest": self.source_artifact_digest,
            "artifact_format": self.artifact_format.value,
            "runtime": self.runtime,
            "layers": [layer.to_record() for layer in self.layers],
            "source_parameter_count": self.source_parameter_count,
            "source_storage_bytes": self.source_storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class HeterogeneousCapabilities:
    artifact_format: ArtifactContainerFormat
    runtime: str
    supported_codecs: tuple[str, ...]
    supports_layer_widths: bool = True
    supports_low_rank: bool = False
    supports_sparsity: bool = False
    supports_mixed_quantization: bool = False
    max_layers: int | None = None
    evidence_source: str = "declared-runtime-capability"

    def __post_init__(self) -> None:
        _text(self.runtime, "capability runtime")
        codecs = tuple(sorted(set(self.supported_codecs)))
        if not codecs or codecs != self.supported_codecs:
            raise HeterogeneousStateError("supported codecs must be non-empty and canonical")
        if self.max_layers is not None:
            _positive(self.max_layers, "capability layer limit")
        _text(self.evidence_source, "capability evidence source")

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_format": self.artifact_format.value,
            "runtime": self.runtime,
            "supported_codecs": list(self.supported_codecs),
            "supports_layer_widths": self.supports_layer_widths,
            "supports_low_rank": self.supports_low_rank,
            "supports_sparsity": self.supports_sparsity,
            "supports_mixed_quantization": self.supports_mixed_quantization,
            "max_layers": self.max_layers,
            "evidence_source": self.evidence_source,
        }


@dataclass(frozen=True, slots=True)
class CompiledHeterogeneousState:
    spec: HeterogeneousArchitectureSpec
    capabilities: HeterogeneousCapabilities
    state_id: str
    parameter_count: int
    storage_bytes: int
    requested_layers: tuple[HeterogeneousLayer, ...]
    effective_layers: tuple[HeterogeneousLayer, ...]

    def __post_init__(self) -> None:
        if not self.state_id.startswith("heterogeneous_state_"):
            raise HeterogeneousStateError("compiled state requires a canonical ID")
        if self.requested_layers != self.spec.layers or self.effective_layers != self.spec.layers:
            raise HeterogeneousStateError("requested and effective layers must be retained exactly")
        _positive(self.parameter_count, "compiled parameter count")
        _positive(self.storage_bytes, "compiled storage bytes")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HETEROGENEOUS_SCHEMA_VERSION,
            "state_id": self.state_id,
            "spec": self.spec.to_record(),
            "capabilities": self.capabilities.to_record(),
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "requested_layers": [layer.to_record() for layer in self.requested_layers],
            "effective_layers": [layer.to_record() for layer in self.effective_layers],
        }


@dataclass(frozen=True, slots=True)
class HeterogeneousCompilation:
    outcome: HeterogeneousOutcome
    state: CompiledHeterogeneousState | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise HeterogeneousStateError("compilation outcomes require reasons")
        if self.outcome is HeterogeneousOutcome.ACCEPTED and self.state is None:
            raise HeterogeneousStateError("accepted compilation requires a state")
        if self.outcome is not HeterogeneousOutcome.ACCEPTED and self.state is not None:
            raise HeterogeneousStateError("non-accepted compilation cannot publish a state")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HETEROGENEOUS_SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "state": None if self.state is None else self.state.to_record(),
            "reasons": list(self.reasons),
        }


def compile_heterogeneous_state(
    spec: HeterogeneousArchitectureSpec,
    capabilities: HeterogeneousCapabilities,
) -> HeterogeneousCompilation:
    """Compile only states the declared artifact/runtime boundary can represent."""

    reasons: list[str] = []
    if spec.artifact_format is not capabilities.artifact_format:
        reasons.append("artifact format is unsupported by the selected runtime")
    if spec.runtime != capabilities.runtime:
        reasons.append("runtime identity does not match capability evidence")
    if capabilities.max_layers is not None and len(spec.layers) > capabilities.max_layers:
        reasons.append("layer count exceeds runtime capability")
    if not capabilities.supports_layer_widths:
        reasons.append("layer-specific widths are unsupported")
    if any(layer.low_rank_factors for layer in spec.layers) and not capabilities.supports_low_rank:
        reasons.append("layer low-rank replacements are unsupported")
    if any(layer.sparsity for layer in spec.layers) and not capabilities.supports_sparsity:
        reasons.append("layer sparsity is unsupported")
    if not set(spec.requested_codecs) <= set(capabilities.supported_codecs):
        reasons.append("one or more requested codecs are unsupported")
    if len(spec.requested_codecs) > 1 and not capabilities.supports_mixed_quantization:
        reasons.append("mixed quantization is unsupported by the runtime")
    if reasons:
        return HeterogeneousCompilation(HeterogeneousOutcome.UNSUPPORTED, None, tuple(reasons))
    try:
        parameter_count = sum(layer.parameter_count for layer in spec.layers)
        storage_bytes = sum(layer.storage_bytes for layer in spec.layers)
    except HeterogeneousStateError as error:
        return HeterogeneousCompilation(HeterogeneousOutcome.UNKNOWN, None, (str(error),))
    identity = canonical_identity_json(
        {
            "schema_version": HETEROGENEOUS_SCHEMA_VERSION,
            "spec": spec.to_record(),
            "capabilities": capabilities.to_record(),
            "parameter_count": parameter_count,
            "storage_bytes": storage_bytes,
        }
    )
    state = CompiledHeterogeneousState(
        spec,
        capabilities,
        f"heterogeneous_state_{hashlib.sha256(identity.encode()).hexdigest()}",
        parameter_count,
        storage_bytes,
        spec.layers,
        spec.layers,
    )
    return HeterogeneousCompilation(
        HeterogeneousOutcome.ACCEPTED,
        state,
        ("all requested layer choices reconcile with declared runtime capabilities",),
    )


@dataclass(frozen=True, slots=True)
class ReloadEvidence:
    reloaded: bool
    runtime: str
    effective_layers: tuple[HeterogeneousLayer, ...]
    parameter_count: int
    storage_bytes: int
    reason: str

    def __post_init__(self) -> None:
        _text(self.runtime, "reload runtime")
        _positive(self.parameter_count, "reloaded parameter count")
        _positive(self.storage_bytes, "reloaded storage bytes")
        _text(self.reason, "reload reason")


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    metric: str
    value: float
    unit: str
    repetitions: int

    def __post_init__(self) -> None:
        _text(self.metric, "benchmark metric")
        _text(self.unit, "benchmark unit")
        _positive(self.repetitions, "benchmark repetitions")
        if not math.isfinite(self.value) or self.value < 0:
            raise HeterogeneousStateError("benchmark values must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "repetitions": self.repetitions,
        }


@dataclass(frozen=True, slots=True)
class MaterializationEvidence:
    state_id: str
    status: MaterializationEvidenceStatus
    source_artifact_digest: str
    output_artifact_digest: str | None
    output_size_bytes: int | None
    reload: ReloadEvidence | None
    benchmarks: tuple[BenchmarkObservation, ...]
    storage_reconciled: bool
    reason: str

    def __post_init__(self) -> None:
        _text(self.state_id, "materialization state ID")
        _digest(self.source_artifact_digest, "materialization source digest")
        _text(self.reason, "materialization reason")
        if self.output_artifact_digest is not None:
            _digest(self.output_artifact_digest, "materialization output digest")
        if self.output_size_bytes is not None:
            _positive(self.output_size_bytes, "materialization output size")
        if self.status is MaterializationEvidenceStatus.COMPLETE:
            if self.output_artifact_digest is None or self.output_size_bytes is None:
                raise HeterogeneousStateError("complete materialization requires output facts")
            if self.reload is None or not self.reload.reloaded or not self.storage_reconciled:
                raise HeterogeneousStateError("complete materialization requires reconciled reload")
        elif self.output_artifact_digest is not None or self.output_size_bytes is not None:
            raise HeterogeneousStateError("failed materialization cannot publish output facts")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HETEROGENEOUS_SCHEMA_VERSION,
            "state_id": self.state_id,
            "status": self.status.value,
            "source_artifact_digest": self.source_artifact_digest,
            "output_artifact_digest": self.output_artifact_digest,
            "output_size_bytes": self.output_size_bytes,
            "reload": None
            if self.reload is None
            else {
                "reloaded": self.reload.reloaded,
                "runtime": self.reload.runtime,
                "effective_layers": [layer.to_record() for layer in self.reload.effective_layers],
                "parameter_count": self.reload.parameter_count,
                "storage_bytes": self.reload.storage_bytes,
                "reason": self.reload.reason,
            },
            "benchmarks": [item.to_record() for item in self.benchmarks],
            "storage_reconciled": self.storage_reconciled,
            "reason": self.reason,
        }


Writer = Callable[[Path, Path, CompiledHeterogeneousState], None]
Reloader = Callable[[Path], ReloadEvidence]
Benchmarker = Callable[[Path], tuple[BenchmarkObservation, ...]]


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def materialize_heterogeneous_state(
    compiled: CompiledHeterogeneousState,
    source_path: Path,
    output_path: Path,
    writer: Writer,
    reloader: Reloader,
    benchmark: Benchmarker,
) -> MaterializationEvidence:
    """Materialize through a staging file and publish only after reload/reconciliation."""

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        return MaterializationEvidence(
            compiled.state_id,
            MaterializationEvidenceStatus.FAILED,
            compiled.spec.source_artifact_digest,
            None,
            None,
            None,
            (),
            False,
            "source and output paths must be distinct",
        )
    if not source_path.is_file():
        return MaterializationEvidence(
            compiled.state_id,
            MaterializationEvidenceStatus.UNKNOWN,
            compiled.spec.source_artifact_digest,
            None,
            None,
            None,
            (),
            False,
            "source artifact does not exist",
        )
    source_digest, _ = _file_digest(source_path)
    if source_digest != compiled.spec.source_artifact_digest:
        return MaterializationEvidence(
            compiled.state_id,
            MaterializationEvidenceStatus.FAILED,
            compiled.spec.source_artifact_digest,
            None,
            None,
            None,
            (),
            False,
            "source artifact digest changed or does not match the specification",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    try:
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{compiled.state_id}.", suffix=".staging", dir=output_path.parent
        )
        os.close(descriptor)
        staging_path = Path(staging_name)
        writer(source_path, staging_path, compiled)
        if _file_digest(source_path)[0] != source_digest:
            raise HeterogeneousStateError("source artifact changed during materialization")
        reload = reloader(staging_path)
        benchmarks = benchmark(staging_path)
        storage_reconciled = (
            reload.reloaded
            and reload.runtime == compiled.capabilities.runtime
            and reload.effective_layers == compiled.effective_layers
            and reload.parameter_count == compiled.parameter_count
            and reload.storage_bytes == compiled.storage_bytes
        )
        if not storage_reconciled:
            return MaterializationEvidence(
                compiled.state_id,
                MaterializationEvidenceStatus.UNKNOWN,
                source_digest,
                None,
                None,
                reload,
                benchmarks,
                False,
                "reload evidence did not reconcile requested/effective layers and analytic costs",
            )
        os.replace(staging_path, output_path)
        staging_path = None
        output_digest, output_size = _file_digest(output_path)
        return MaterializationEvidence(
            compiled.state_id,
            MaterializationEvidenceStatus.COMPLETE,
            source_digest,
            output_digest,
            output_size,
            reload,
            benchmarks,
            True,
            "materialized, reloaded, benchmarked, and reconciled atomically",
        )
    except (OSError, HeterogeneousStateError, ValueError) as error:
        return MaterializationEvidence(
            compiled.state_id,
            MaterializationEvidenceStatus.FAILED,
            source_digest,
            None,
            None,
            None,
            (),
            False,
            str(error),
        )
    finally:
        if staging_path is not None:
            with suppress(FileNotFoundError):
                staging_path.unlink()
