"""Framework-light architecture walking."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from modelsurgeon.graph.component_id import ComponentId


class NamedModuleProvider(Protocol):
    """Minimal protocol implemented by torch modules and tiny test doubles."""

    def named_modules(self) -> Iterable[tuple[str, object]]: ...


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    """A discovered module with canonical identity and lightweight metadata."""

    component_id: ComponentId
    module_type: str
    parameter_count: int | None


def _parameter_count(module: object) -> int | None:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return sum(int(parameter.numel()) for parameter in parameters(recurse=False))
    except (AttributeError, TypeError):
        return None


def walk_named_modules(model: NamedModuleProvider) -> Iterator[ComponentRecord]:
    """Yield canonical records in the order exposed by the model framework."""
    for name, module in model.named_modules():
        yield ComponentRecord(
            component_id=ComponentId.from_module_name(name),
            module_type=type(module).__name__,
            parameter_count=_parameter_count(module),
        )

