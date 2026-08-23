"""Exception-safe activation hook lifecycle management."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol, Self

from modelsurgeon.graph import ComponentId


class HookHandle(Protocol):
    def remove(self) -> None: ...


class HookableModule(Protocol):
    def register_forward_hook(self, hook: Callable[..., object]) -> HookHandle: ...


class HookLifecycleError(ValueError):
    """Raised for invalid targets or lifecycle use."""


@dataclass(frozen=True, slots=True)
class ActivationCapture:
    component_id: ComponentId
    value: object


class ActivationHookManager(AbstractContextManager["ActivationHookManager"]):
    """Own hooks and captures for exactly one exception-safe context lifetime."""

    def __init__(
        self,
        modules: Mapping[ComponentId, HookableModule],
        targets: Iterable[ComponentId],
        *,
        detach: Callable[[object], object] | None = None,
    ) -> None:
        ordered = tuple(targets)
        if len(ordered) != len(set(ordered)):
            raise HookLifecycleError("duplicate hook target registration is not allowed")
        missing = tuple(target for target in ordered if target not in modules)
        if missing:
            raise HookLifecycleError(
                "unknown hook targets: " + ", ".join(str(target) for target in missing)
            )
        self._modules = modules
        self.targets = tuple(sorted(ordered))
        self._detach = detach or (lambda value: value)
        self._handles: list[HookHandle] = []
        self._captures: list[ActivationCapture] = []
        self._active = False

    @property
    def captures(self) -> tuple[ActivationCapture, ...]:
        return tuple(self._captures)

    @property
    def active(self) -> bool:
        return self._active

    def clear(self) -> None:
        self._captures.clear()

    def _hook(self, component_id: ComponentId) -> Callable[..., None]:
        def capture(module: object, inputs: object, output: object) -> None:
            del module, inputs
            self._captures.append(ActivationCapture(component_id, self._detach(output)))

        return capture

    def __enter__(self) -> Self:
        if self._active or self._handles:
            raise HookLifecycleError("hook manager cannot be registered twice")
        self._active = True
        try:
            for target in self.targets:
                self._handles.append(
                    self._modules[target].register_forward_hook(self._hook(target))
                )
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        errors: list[BaseException] = []
        while self._handles:
            handle = self._handles.pop()
            try:
                handle.remove()
            except BaseException as error:
                errors.append(error)
        self._captures.clear()
        self._active = False
        if errors:
            raise BaseExceptionGroup("one or more activation hooks failed to remove", errors)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
