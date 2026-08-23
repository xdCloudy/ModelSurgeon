"""Tier 0 deterministic non-finite checks for parameters and evaluation outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

TIER0_NUMERICS_VERSION = "1"


class NumericSurfaceKind(StrEnum):
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    LOGITS = "logits"
    LOSS = "loss"


class NumericSurface(Protocol):
    kind: NumericSurfaceKind
    identity: str

    def value_count(self) -> int: ...

    def value_at(self, index: int) -> float: ...


@dataclass(frozen=True, slots=True)
class SequenceNumericSurface:
    kind: NumericSurfaceKind
    identity: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.values:
            raise ValueError("numeric surfaces require an identity and values")

    def value_count(self) -> int:
        return len(self.values)

    def value_at(self, index: int) -> float:
        return self.values[index]


@dataclass(frozen=True, slots=True)
class Tier0NumericsConfig:
    max_values_per_surface: int = 4096

    def __post_init__(self) -> None:
        if self.max_values_per_surface <= 0:
            raise ValueError("numerical validation sample budget must be positive")


@dataclass(frozen=True, slots=True)
class Tier0NumericsFailure:
    kind: NumericSurfaceKind
    identity: str
    sampled_index: int
    value_class: str

    def __post_init__(self) -> None:
        if not self.identity or self.sampled_index < 0:
            raise ValueError("numerical failure context is invalid")
        if self.value_class not in {"nan", "+inf", "-inf"}:
            raise ValueError("unsupported numerical failure class")


@dataclass(frozen=True, slots=True)
class Tier0NumericsResult:
    passed: bool
    inspected_surfaces: int
    inspected_values: int
    failure: Tier0NumericsFailure | None
    max_values_per_surface: int
    version: str = TIER0_NUMERICS_VERSION

    def __post_init__(self) -> None:
        if self.inspected_surfaces < 0 or self.inspected_values < 0:
            raise ValueError("numerical inspection counts cannot be negative")
        if self.passed != (self.failure is None):
            raise ValueError("numerical result pass state disagrees with failure state")

    def to_record(self) -> dict[str, object]:
        failure = None
        if self.failure is not None:
            failure = {
                "kind": self.failure.kind.value,
                "identity": self.failure.identity,
                "sampled_index": self.failure.sampled_index,
                "value_class": self.failure.value_class,
            }
        return {
            "version": self.version,
            "passed": self.passed,
            "inspected_surfaces": self.inspected_surfaces,
            "inspected_values": self.inspected_values,
            "max_values_per_surface": self.max_values_per_surface,
            "failure": failure,
        }


def deterministic_sample_indices(count: int, budget: int) -> tuple[int, ...]:
    """Return canonical evenly distributed indices including both endpoints."""

    if count <= 0 or budget <= 0:
        raise ValueError("sampling count and budget must be positive")
    if count <= budget:
        return tuple(range(count))
    if budget == 1:
        return (0,)
    return tuple((position * (count - 1)) // (budget - 1) for position in range(budget))


def _classify_nonfinite(value: float) -> str | None:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return None


def validate_tier0_numerics(
    surfaces: tuple[NumericSurface, ...],
    config: Tier0NumericsConfig | None = None,
) -> Tier0NumericsResult:
    """Inspect surfaces in supplied canonical order and return the first non-finite value."""

    if not surfaces:
        raise ValueError("numerical validation requires at least one surface")
    resolved = config or Tier0NumericsConfig()
    inspected_values = 0
    for surface_index, surface in enumerate(surfaces):
        if not surface.identity:
            raise ValueError("numeric surface identity cannot be empty")
        count = surface.value_count()
        if count <= 0:
            raise ValueError(f"numeric surface {surface.identity!r} is empty")
        for index in deterministic_sample_indices(count, resolved.max_values_per_surface):
            value = float(surface.value_at(index))
            inspected_values += 1
            value_class = _classify_nonfinite(value)
            if value_class is not None:
                return Tier0NumericsResult(
                    False,
                    surface_index + 1,
                    inspected_values,
                    Tier0NumericsFailure(surface.kind, surface.identity, index, value_class),
                    resolved.max_values_per_surface,
                )
    return Tier0NumericsResult(
        True,
        len(surfaces),
        inspected_values,
        None,
        resolved.max_values_per_surface,
    )
