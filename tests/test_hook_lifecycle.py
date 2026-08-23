"""Tests for exception-safe activation hook ownership."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation import ActivationHookManager, HookLifecycleError


@dataclass
class Handle:
    module: Module
    hook: object
    removed: bool = False

    def remove(self) -> None:
        self.removed = True
        self.module.hooks.remove(self.hook)


class Module:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.hooks: list[object] = []
        self.handles: list[Handle] = []

    def register_forward_hook(self, hook: object) -> Handle:
        if self.fail:
            raise RuntimeError("registration failed")
        self.hooks.append(hook)
        handle = Handle(self, hook)
        self.handles.append(handle)
        return handle

    def emit(self, output: object) -> None:
        for hook in tuple(self.hooks):
            hook(self, (), output)  # type: ignore[operator]


def _ids() -> tuple[ComponentId, ComponentId]:
    return (
        ComponentId.parse("model.layers.0.self_attn"),
        ComponentId.parse("model.layers.0.mlp"),
    )


def test_hooks_capture_in_canonical_order_and_release_on_success() -> None:
    attention, mlp = _ids()
    modules = {attention: Module(), mlp: Module()}
    manager = ActivationHookManager(
        modules, (mlp, attention), detach=lambda value: ("detached", value)
    )

    with manager:
        assert manager.active
        modules[attention].emit("a")
        modules[mlp].emit("m")
        assert [(str(item.component_id), item.value) for item in manager.captures] == [
            (str(attention), ("detached", "a")),
            (str(mlp), ("detached", "m")),
        ]

    assert not manager.active
    assert manager.captures == ()
    assert all(not module.hooks for module in modules.values())
    assert all(handle.removed for module in modules.values() for handle in module.handles)


@pytest.mark.parametrize("error", [RuntimeError("body"), KeyboardInterrupt()])
def test_cleanup_runs_for_exception_and_interrupt(error: BaseException) -> None:
    target, _ = _ids()
    module = Module()
    manager = ActivationHookManager({target: module}, (target,))

    with pytest.raises(type(error)), manager:
        module.emit(object())
        raise error

    assert module.hooks == []
    assert manager.captures == ()
    assert not manager.active


def test_partial_registration_failure_removes_prior_hooks() -> None:
    attention, mlp = _ids()
    first = Module()
    failing = Module(fail=True)
    manager = ActivationHookManager({attention: failing, mlp: first}, (attention, mlp))

    with pytest.raises(RuntimeError, match="registration failed"):
        manager.__enter__()

    assert first.hooks == []
    assert first.handles[0].removed
    assert not manager.active


def test_duplicate_unknown_and_double_registration_are_rejected() -> None:
    target, missing = _ids()
    module = Module()
    with pytest.raises(HookLifecycleError, match="duplicate"):
        ActivationHookManager({target: module}, (target, target))
    with pytest.raises(HookLifecycleError, match="unknown"):
        ActivationHookManager({target: module}, (missing,))

    manager = ActivationHookManager({target: module}, (target,))
    with manager, pytest.raises(HookLifecycleError, match="twice"):
        manager.__enter__()


def test_captured_object_is_not_retained_after_exit() -> None:
    class Payload:
        pass

    target, _ = _ids()
    module = Module()
    manager = ActivationHookManager({target: module}, (target,))
    with manager:
        payload = Payload()
        reference = weakref.ref(payload)
        module.emit(payload)
        del payload
        assert reference() is not None

    assert reference() is None
