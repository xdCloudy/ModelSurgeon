"""Tests for platform process-I/O telemetry bindings."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import modelsurgeon.experiments.runtime_telemetry as telemetry


class _FakeFunction:
    def __init__(self, callback: Callable[..., object]) -> None:
        self.callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *arguments: Any) -> object:
        return self.callback(*arguments)


def test_windows_io_probe_uses_pointer_width_safe_signatures(monkeypatch: Any) -> None:
    large_handle = 2**48
    get_current_process = _FakeFunction(lambda: large_handle)

    def populate(handle: object, raw_counters: Any) -> int:
        assert handle == large_handle
        counters = ctypes.cast(
            raw_counters,
            ctypes.POINTER(telemetry._WindowsIOCounters),
        ).contents
        counters.ReadTransferCount = 123
        counters.WriteTransferCount = 456
        return 1

    get_io_counters = _FakeFunction(populate)
    kernel32 = SimpleNamespace(
        GetCurrentProcess=get_current_process,
        GetProcessIoCounters=get_io_counters,
    )
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=kernel32),
        raising=False,
    )

    result = telemetry._windows_io_counters()

    assert result == telemetry.ProcessIOCounters(123, 456)
    assert get_current_process.restype is ctypes.c_void_p
    assert get_io_counters.argtypes == (
        ctypes.c_void_p,
        ctypes.POINTER(telemetry._WindowsIOCounters),
    )
    assert get_io_counters.restype is ctypes.c_int
