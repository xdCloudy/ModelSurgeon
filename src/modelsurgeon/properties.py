"""Deterministic property-case generation and retained shrinkable failures."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PROPERTY_SUITE_SCHEMA_VERSION: Final[int] = 1
PROPERTY_SUITE_PROTOCOL_REVISION: Final[str] = "mutation-properties-v1"
WRITABLE_NATIVE_GGUF_CODECS: Final[tuple[str, ...]] = (
    "BF16",
    "F16",
    "F32",
    "IQ4_NL",
    "IQ4_XS",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
)
PROPERTY_OPERATIONS: Final[tuple[str, ...]] = (
    "hf_mutation_sequence",
    "native_attention_head_removal",
    "native_layer_removal",
    "native_low_rank_replacement",
    "native_mlp_channel_removal",
    "native_direct_copy",
)
PROPERTY_FAILURE_STAGES: Final[tuple[str, ...]] = (
    "source",
    "stage",
    "checksum",
    "structure",
    "reload",
    "promotion",
    "rollback",
)


class PropertySuiteError(ValueError):
    """Raised when generated property evidence is malformed or not clean."""


class PropertyOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PropertyCase:
    seed: int
    target: str
    operation: str
    failure_stage: str
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.target or not self.operation:
            raise PropertySuiteError("property cases require seed, target, and operation")
        if self.failure_stage not in PROPERTY_FAILURE_STAGES:
            raise PropertySuiteError("unknown property failure stage")
        if not self.dimensions or any(value <= 0 for value in self.dimensions):
            raise PropertySuiteError("property dimensions must be positive")

    @property
    def case_id(self) -> str:
        payload = {
            "seed": self.seed,
            "target": self.target,
            "operation": self.operation,
            "failure_stage": self.failure_stage,
            "dimensions": self.dimensions,
        }
        return (
            "property_"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )

    def shrink(self) -> PropertyCase:
        """Return a deterministic smaller case while retaining its seed."""
        reduced = tuple(max(1, value // 2) for value in self.dimensions)
        return PropertyCase(self.seed, self.target, self.operation, self.failure_stage, reduced)

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "target": self.target,
            "operation": self.operation,
            "failure_stage": self.failure_stage,
            "dimensions": list(self.dimensions),
        }


@dataclass(frozen=True, slots=True)
class PropertyResult:
    case: PropertyCase
    outcome: PropertyOutcome
    detail: str | None = None

    def to_record(self) -> dict[str, object]:
        return {"case": self.case.to_record(), "outcome": self.outcome.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PropertyFailure:
    case: PropertyCase
    shrunk_case: PropertyCase
    exception_type: str
    detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "case": self.case.to_record(),
            "shrunk_case": self.shrunk_case.to_record(),
            "seed": self.case.seed,
            "exception_type": self.exception_type,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PropertySuiteReport:
    results: tuple[PropertyResult, ...]
    failures: tuple[PropertyFailure, ...]
    coverage_targets: tuple[str, ...]
    protocol_revision: str = PROPERTY_SUITE_PROTOCOL_REVISION
    schema_version: int = PROPERTY_SUITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROPERTY_SUITE_SCHEMA_VERSION:
            raise PropertySuiteError("unsupported property suite schema version")
        if not self.coverage_targets:
            raise PropertySuiteError("property suite requires coverage targets")

    @property
    def outcome(self) -> PropertyOutcome:
        if self.failures:
            return PropertyOutcome.FAIL
        if any(item.outcome is PropertyOutcome.UNKNOWN for item in self.results):
            return PropertyOutcome.UNKNOWN
        if any(item.outcome is PropertyOutcome.UNSUPPORTED for item in self.results):
            return PropertyOutcome.UNSUPPORTED
        return PropertyOutcome.PASS

    def require_clean(self) -> None:
        if self.outcome is not PropertyOutcome.PASS:
            raise PropertySuiteError(f"property suite outcome is {self.outcome.value}")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": self.protocol_revision,
            "outcome": self.outcome.value,
            "executions": len(self.results),
            "failures": [item.to_record() for item in self.failures],
            "coverage_targets": list(self.coverage_targets),
            "results": [item.to_record() for item in self.results],
        }


def property_coverage_targets() -> tuple[str, ...]:
    """Return every writable codec and operation class covered by the contract."""
    return tuple(
        sorted(
            [f"gguf_codec:{codec}" for codec in WRITABLE_NATIVE_GGUF_CODECS]
            + [f"operation:{operation}" for operation in PROPERTY_OPERATIONS]
        )
    )


def generate_property_cases(*, seed: int = 407, cases: int = 2048) -> tuple[PropertyCase, ...]:
    """Generate a bounded, deterministic CPU property campaign."""
    if seed < 0 or cases <= 0:
        raise PropertySuiteError("property seed and case count must be positive")
    targets = property_coverage_targets()
    randomizer = random.Random(seed)
    generated: list[PropertyCase] = []
    for index in range(cases):
        target = targets[index % len(targets)]
        operation = PROPERTY_OPERATIONS[index % len(PROPERTY_OPERATIONS)]
        failure_stage = PROPERTY_FAILURE_STAGES[index % len(PROPERTY_FAILURE_STAGES)]
        dimensions = tuple(randomizer.randint(1, 1024) for _ in range(3))
        generated.append(PropertyCase(seed + index, target, operation, failure_stage, dimensions))
    return tuple(generated)


def _shrink_failure(case: PropertyCase, predicate: Callable[[PropertyCase], bool]) -> PropertyCase:
    current = case
    while current.shrink() != current and predicate(current.shrink()):
        current = current.shrink()
    return current


def run_property_suite(
    cases: Sequence[PropertyCase],
    evaluator: Callable[[PropertyCase], None],
    *,
    coverage_targets: Sequence[str] | None = None,
) -> PropertySuiteReport:
    """Run generated cases and retain deterministic shrunk counterexamples."""
    targets = tuple(coverage_targets or property_coverage_targets())
    results: list[PropertyResult] = []
    failures: list[PropertyFailure] = []
    for case in cases:
        try:
            evaluator(case)
        except NotImplementedError as error:
            results.append(PropertyResult(case, PropertyOutcome.UNSUPPORTED, str(error)))
        except TimeoutError as error:
            results.append(PropertyResult(case, PropertyOutcome.UNKNOWN, str(error)))
        except Exception as error:

            def predicate(candidate: PropertyCase) -> bool:
                return _fails(evaluator, candidate)

            failures.append(
                PropertyFailure(
                    case,
                    _shrink_failure(case, predicate),
                    type(error).__name__,
                    str(error),
                )
            )
            results.append(PropertyResult(case, PropertyOutcome.FAIL, str(error)))
        else:
            results.append(PropertyResult(case, PropertyOutcome.PASS))
    return PropertySuiteReport(tuple(results), tuple(failures), targets)


def _fails(evaluator: Callable[[PropertyCase], None], case: PropertyCase) -> bool:
    try:
        evaluator(case)
    except Exception:
        return True
    return False
