"""Explicit, fail-closed mixed-precision selection and metric precision binding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.experiments.identity import canonical_identity_json

PRECISION_POLICY_VERSION = "1"
PRECISION_CAPABILITY_SOURCE = "hardware-inventory-v1"


class PrecisionPolicyError(ValueError):
    """Raised when a precision request or observed execution violates the policy."""


class PrecisionDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class PrecisionDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class PrecisionRequestKind(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    AUTOCAST = "autocast"


class PrecisionExecutionMode(StrEnum):
    DIRECT = "direct"
    AUTOCAST = "autocast"


_DTYPE_BY_REQUEST = {
    PrecisionRequestKind.FLOAT32: PrecisionDType.FLOAT32,
    PrecisionRequestKind.FLOAT16: PrecisionDType.FLOAT16,
    PrecisionRequestKind.BFLOAT16: PrecisionDType.BFLOAT16,
}
_DTYPE_ORDER = {
    PrecisionDType.FLOAT32: 0,
    PrecisionDType.FLOAT16: 1,
    PrecisionDType.BFLOAT16: 2,
}


@dataclass(frozen=True, slots=True)
class PrecisionCapabilities:
    device: PrecisionDevice
    direct_dtypes: tuple[PrecisionDType, ...]
    autocast_dtypes: tuple[PrecisionDType, ...]
    accumulation_dtypes: tuple[PrecisionDType, ...]
    source: str = PRECISION_CAPABILITY_SOURCE

    def __post_init__(self) -> None:
        if not self.source:
            raise PrecisionPolicyError("precision capability source is required")
        for label, values in (
            ("direct", self.direct_dtypes),
            ("autocast", self.autocast_dtypes),
            ("accumulation", self.accumulation_dtypes),
        ):
            if len(values) != len(set(values)):
                raise PrecisionPolicyError(f"{label} precision capabilities must be unique")
            if values != tuple(sorted(values, key=_DTYPE_ORDER.__getitem__)):
                raise PrecisionPolicyError(f"{label} precision capabilities must be canonical")
        if PrecisionDType.FLOAT32 not in self.direct_dtypes:
            raise PrecisionPolicyError("precision capabilities must support direct float32")
        if PrecisionDType.FLOAT32 not in self.accumulation_dtypes:
            raise PrecisionPolicyError("precision capabilities must support float32 accumulation")
        if any(dtype not in self.direct_dtypes for dtype in self.autocast_dtypes):
            raise PrecisionPolicyError("autocast dtypes must also be directly supported")
        if PrecisionDType.FLOAT32 in self.autocast_dtypes:
            raise PrecisionPolicyError("float32 is not a mixed-precision autocast target")

    def to_record(self) -> dict[str, object]:
        return {
            "device": self.device.value,
            "direct_dtypes": [dtype.value for dtype in self.direct_dtypes],
            "autocast_dtypes": [dtype.value for dtype in self.autocast_dtypes],
            "accumulation_dtypes": [dtype.value for dtype in self.accumulation_dtypes],
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PrecisionRequest:
    kind: PrecisionRequestKind
    accumulation_dtype: PrecisionDType = PrecisionDType.FLOAT32
    allow_fallback: bool = False
    autocast_preference: tuple[PrecisionDType, ...] = (
        PrecisionDType.BFLOAT16,
        PrecisionDType.FLOAT16,
    )

    def __post_init__(self) -> None:
        if len(self.autocast_preference) != len(set(self.autocast_preference)):
            raise PrecisionPolicyError("autocast preference dtypes must be unique")
        if PrecisionDType.FLOAT32 in self.autocast_preference:
            raise PrecisionPolicyError("autocast preference cannot contain float32")
        if self.kind is PrecisionRequestKind.AUTOCAST and not self.autocast_preference:
            raise PrecisionPolicyError("autocast requests require at least one preferred dtype")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "accumulation_dtype": self.accumulation_dtype.value,
            "allow_fallback": self.allow_fallback,
            "autocast_preference": [dtype.value for dtype in self.autocast_preference],
        }


@dataclass(frozen=True, slots=True)
class PrecisionDecision:
    request: PrecisionRequest
    capabilities: PrecisionCapabilities
    mode: PrecisionExecutionMode
    compute_dtype: PrecisionDType
    accumulation_dtype: PrecisionDType
    fallback_reason: str | None = None
    version: str = PRECISION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != PRECISION_POLICY_VERSION:
            raise PrecisionPolicyError(f"unsupported precision policy version {self.version}")
        supported = (
            self.capabilities.autocast_dtypes
            if self.mode is PrecisionExecutionMode.AUTOCAST
            else self.capabilities.direct_dtypes
        )
        if self.compute_dtype not in supported:
            raise PrecisionPolicyError("precision decision compute dtype is unsupported")
        if self.accumulation_dtype not in self.capabilities.accumulation_dtypes:
            raise PrecisionPolicyError("precision decision accumulation dtype is unsupported")
        if self.fallback_reason is not None and not self.fallback_reason:
            raise PrecisionPolicyError("precision fallback reason cannot be blank")

    @property
    def fallback_applied(self) -> bool:
        return self.fallback_reason is not None

    def _identity_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request": self.request.to_record(),
            "capabilities": self.capabilities.to_record(),
            "mode": self.mode.value,
            "compute_dtype": self.compute_dtype.value,
            "accumulation_dtype": self.accumulation_dtype.value,
            "fallback_reason": self.fallback_reason,
        }

    @property
    def decision_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode("utf-8")
        ).hexdigest()
        return f"precision_{digest}"

    def to_record(self) -> dict[str, object]:
        return {**self._identity_record(), "decision_id": self.decision_id}


@dataclass(frozen=True, slots=True)
class PrecisionExecutionContext:
    compute_dtype: PrecisionDType
    accumulation_dtype: PrecisionDType
    autocast_enabled: bool

    def to_record(self) -> dict[str, str | bool]:
        return {
            "compute_dtype": self.compute_dtype.value,
            "accumulation_dtype": self.accumulation_dtype.value,
            "autocast_enabled": self.autocast_enabled,
        }


@dataclass(frozen=True, slots=True)
class MetricPrecisionRecord:
    metric_name: str
    decision_id: str
    execution: PrecisionExecutionContext

    def __post_init__(self) -> None:
        if not self.metric_name or not self.decision_id:
            raise PrecisionPolicyError("metric precision records require metric and decision IDs")

    def to_record(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "decision_id": self.decision_id,
            "execution": self.execution.to_record(),
        }


def _parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    major_text, separator, minor_text = value.partition(".")
    if not separator or not major_text.isdigit() or not minor_text.isdigit():
        return None
    return int(major_text), int(minor_text)


def precision_capabilities_from_hardware(
    inventory: HardwareInventory,
    device: PrecisionDevice,
) -> PrecisionCapabilities:
    """Resolve conservative hardware-level precision support without framework guessing."""

    if device is PrecisionDevice.CPU:
        return PrecisionCapabilities(
            device,
            (PrecisionDType.FLOAT32,),
            (),
            (PrecisionDType.FLOAT32,),
        )
    if not inventory.cuda.available:
        raise PrecisionPolicyError("CUDA precision requested on CPU-only hardware")
    capabilities = tuple(
        _parse_compute_capability(item.compute_capability) for item in inventory.cuda.devices
    )
    known = bool(capabilities) and all(item is not None for item in capabilities)
    direct = [PrecisionDType.FLOAT32]
    autocast: list[PrecisionDType] = []
    if known:
        minimum = min(item for item in capabilities if item is not None)
        if minimum >= (5, 3):
            direct.append(PrecisionDType.FLOAT16)
            autocast.append(PrecisionDType.FLOAT16)
        if minimum >= (8, 0):
            direct.append(PrecisionDType.BFLOAT16)
            autocast.append(PrecisionDType.BFLOAT16)
    return PrecisionCapabilities(
        device,
        tuple(sorted(direct, key=_DTYPE_ORDER.__getitem__)),
        tuple(sorted(autocast, key=_DTYPE_ORDER.__getitem__)),
        (PrecisionDType.FLOAT32,),
    )


def _resolve_accumulation(
    request: PrecisionRequest,
    capabilities: PrecisionCapabilities,
) -> tuple[PrecisionDType, str | None]:
    if request.accumulation_dtype in capabilities.accumulation_dtypes:
        return request.accumulation_dtype, None
    if request.allow_fallback and PrecisionDType.FLOAT32 in capabilities.accumulation_dtypes:
        return (
            PrecisionDType.FLOAT32,
            f"requested accumulation dtype {request.accumulation_dtype.value} is unsupported; "
            "fell back to float32 accumulation",
        )
    raise PrecisionPolicyError(
        f"requested accumulation dtype {request.accumulation_dtype.value} is unsupported on "
        f"{capabilities.device.value}"
    )


def _join_fallback_reasons(*reasons: str | None) -> str | None:
    values = tuple(reason for reason in reasons if reason is not None)
    return "; ".join(values) if values else None


def select_precision_policy(
    request: PrecisionRequest,
    capabilities: PrecisionCapabilities,
) -> PrecisionDecision:
    """Select an explicit compute/accumulation policy or fail closed."""

    accumulation, accumulation_fallback = _resolve_accumulation(request, capabilities)
    if request.kind is PrecisionRequestKind.AUTOCAST:
        selected = next(
            (
                dtype
                for dtype in request.autocast_preference
                if dtype in capabilities.autocast_dtypes
            ),
            None,
        )
        if selected is not None:
            return PrecisionDecision(
                request,
                capabilities,
                PrecisionExecutionMode.AUTOCAST,
                selected,
                accumulation,
                accumulation_fallback,
            )
        if not request.allow_fallback:
            raise PrecisionPolicyError(
                f"autocast is unsupported for requested dtypes on {capabilities.device.value}"
            )
        return PrecisionDecision(
            request,
            capabilities,
            PrecisionExecutionMode.DIRECT,
            PrecisionDType.FLOAT32,
            accumulation,
            _join_fallback_reasons(
                "autocast is unsupported for requested dtypes; fell back to direct float32",
                accumulation_fallback,
            ),
        )

    requested_dtype = _DTYPE_BY_REQUEST[request.kind]
    if requested_dtype in capabilities.direct_dtypes:
        return PrecisionDecision(
            request,
            capabilities,
            PrecisionExecutionMode.DIRECT,
            requested_dtype,
            accumulation,
            accumulation_fallback,
        )
    if not request.allow_fallback:
        raise PrecisionPolicyError(
            f"requested compute dtype {requested_dtype.value} is unsupported on "
            f"{capabilities.device.value}"
        )
    return PrecisionDecision(
        request,
        capabilities,
        PrecisionExecutionMode.DIRECT,
        PrecisionDType.FLOAT32,
        accumulation,
        _join_fallback_reasons(
            f"requested compute dtype {requested_dtype.value} is unsupported; "
            "fell back to float32",
            accumulation_fallback,
        ),
    )


def expected_execution_context(decision: PrecisionDecision) -> PrecisionExecutionContext:
    return PrecisionExecutionContext(
        decision.compute_dtype,
        decision.accumulation_dtype,
        decision.mode is PrecisionExecutionMode.AUTOCAST,
    )


def bind_metric_precision(
    metric_name: str,
    decision: PrecisionDecision,
    observed: PrecisionExecutionContext,
) -> MetricPrecisionRecord:
    """Bind a metric only when its observed precision exactly matches the selected policy."""

    if not metric_name:
        raise PrecisionPolicyError("metric precision binding requires a metric name")
    expected = expected_execution_context(decision)
    if observed != expected:
        raise PrecisionPolicyError(
            f"metric {metric_name!r} precision changed from selected policy: "
            f"expected {expected.to_record()}, observed {observed.to_record()}"
        )
    return MetricPrecisionRecord(metric_name, decision.decision_id, observed)
