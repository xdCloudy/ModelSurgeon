"""Tier 0 candidate validation: load, graph, shapes, then one bounded forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

TIER0_VALIDATION_VERSION = "1"


class Tier0Stage(StrEnum):
    LOAD = "load"
    GRAPH = "graph"
    SHAPES = "shapes"
    FORWARD = "forward"


class Tier0ValidationBackend(Protocol):
    device: str

    def load(self) -> object: ...

    def validate_graph(self, model: object) -> None: ...

    def validate_shapes(self, model: object) -> None: ...

    def forward(self, model: object, max_tokens: int) -> object: ...


@dataclass(frozen=True, slots=True)
class Tier0ValidationConfig:
    max_forward_tokens: int = 32

    def __post_init__(self) -> None:
        if self.max_forward_tokens <= 0:
            raise ValueError("Tier 0 forward token budget must be positive")


@dataclass(frozen=True, slots=True)
class Tier0ValidationResult:
    passed: bool
    completed_stages: tuple[Tier0Stage, ...]
    failure_stage: Tier0Stage | None
    failure_type: str | None
    failure_message: str | None
    device: str
    max_forward_tokens: int
    version: str = TIER0_VALIDATION_VERSION

    def __post_init__(self) -> None:
        expected = tuple(Tier0Stage)
        if self.passed:
            if self.completed_stages != expected or any(
                value is not None
                for value in (self.failure_stage, self.failure_type, self.failure_message)
            ):
                raise ValueError("passing Tier 0 results must complete every stage without failure")
        elif self.failure_stage is None or not self.failure_type or self.failure_message is None:
            raise ValueError("failing Tier 0 results require classified failure context")
        if not self.device:
            raise ValueError("Tier 0 validation requires a device identity")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "passed": self.passed,
            "completed_stages": [stage.value for stage in self.completed_stages],
            "failure_stage": None if self.failure_stage is None else self.failure_stage.value,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "device": self.device,
            "max_forward_tokens": self.max_forward_tokens,
        }


def run_tier0_validation(
    backend: Tier0ValidationBackend,
    config: Tier0ValidationConfig | None = None,
) -> Tier0ValidationResult:
    """Run Tier 0 sequentially and return the first stage-classified failure."""

    resolved = config or Tier0ValidationConfig()
    completed: list[Tier0Stage] = []
    model: object | None = None

    operations = (
        (Tier0Stage.LOAD, lambda: backend.load()),
        (Tier0Stage.GRAPH, lambda: backend.validate_graph(model)),
        (Tier0Stage.SHAPES, lambda: backend.validate_shapes(model)),
        (
            Tier0Stage.FORWARD,
            lambda: backend.forward(model, resolved.max_forward_tokens),
        ),
    )
    for stage, operation in operations:
        try:
            value = operation()
            if stage is Tier0Stage.LOAD:
                model = value
                if model is None:
                    raise ValueError("load stage returned no model")
        except Exception as error:
            return Tier0ValidationResult(
                False,
                tuple(completed),
                stage,
                type(error).__name__,
                str(error),
                backend.device,
                resolved.max_forward_tokens,
            )
        completed.append(stage)

    return Tier0ValidationResult(
        True,
        tuple(completed),
        None,
        None,
        None,
        backend.device,
        resolved.max_forward_tokens,
    )
