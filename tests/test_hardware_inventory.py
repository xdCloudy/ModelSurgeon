"""Tests for CPU-first, optional-CUDA hardware inventory."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelsurgeon.experiments import hardware
from modelsurgeon.experiments.hardware import (
    CUDAInventory,
    MemoryInventory,
    collect_hardware_inventory,
)


def _base_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        hardware.platform,
        "uname",
        lambda: SimpleNamespace(
            system="TestOS",
            release="1.2",
            version="build-3",
            machine="test64",
            processor="Test CPU",
        ),
    )
    monkeypatch.setattr(hardware.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(hardware.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(hardware, "_memory_inventory", lambda: MemoryInventory(1024, 512))
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=4096, free=1536),
    )
    assert tmp_path.resolve().is_absolute()


def test_cpu_only_host_returns_complete_valid_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _base_host(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hardware,
        "_installed_version",
        lambda name: "0.0.1" if name == "modelsurgeon" else None,
    )
    monkeypatch.setattr(
        hardware,
        "_cuda_inventory",
        lambda version: (CUDAInventory(False, None, (), ()), ()),
    )

    inventory = collect_hardware_inventory(tmp_path)
    record = inventory.to_record()

    assert record == {
        "schema_version": 1,
        "os": {"name": "TestOS", "release": "1.2", "version": "build-3"},
        "cpu": {
            "architecture": "test64",
            "processor": "Test CPU",
            "logical_cores": 6,
        },
        "memory": {"total_bytes": 1024, "available_bytes": 512},
        "disk": {
            "path": str(tmp_path.resolve()),
            "total_bytes": 4096,
            "free_bytes": 1536,
        },
        "cuda": {
            "available": False,
            "compiled_version": None,
            "driver_versions": [],
            "devices": [],
        },
        "software": {
            "python_version": "3.12.9",
            "python_implementation": "CPython",
            "modelsurgeon_version": "0.0.1",
            "pytorch_version": None,
        },
        "warnings": [],
    }
    assert json.loads(json.dumps(record)) == record


def test_cuda_pytorch_driver_and_devices_are_recorded_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCUDA:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_properties(index: int) -> object:
            return SimpleNamespace(name=f"GPU {index}", total_memory=(index + 1) * 1024)

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            return (8, index)

    torch = SimpleNamespace(cuda=FakeCUDA(), version=SimpleNamespace(cuda="12.4"))
    monkeypatch.setattr(hardware, "import_module", lambda name: torch)
    monkeypatch.setattr(hardware, "_driver_versions", lambda: ("555.42",))

    cuda, warnings = hardware._cuda_inventory("2.5.1")

    assert warnings == ()
    assert cuda.to_record() == {
        "available": True,
        "compiled_version": "12.4",
        "driver_versions": ["555.42"],
        "devices": [
            {
                "index": 0,
                "name": "GPU 0",
                "total_memory_bytes": 1024,
                "compute_capability": "8.0",
            },
            {
                "index": 1,
                "name": "GPU 1",
                "total_memory_bytes": 2048,
                "compute_capability": "8.1",
            },
        ],
    }


def test_broken_optional_pytorch_becomes_warning_not_inventory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> object:
        raise OSError("broken runtime")

    monkeypatch.setattr(hardware, "import_module", fail_import)
    monkeypatch.setattr(hardware, "_driver_versions", lambda: ("550.1",))

    cuda, warnings = hardware._cuda_inventory("2.5.1")

    assert cuda.available is False
    assert cuda.driver_versions == ("550.1",)
    assert warnings == ("PyTorch 2.5.1 could not be imported: OSError",)


def test_driver_probe_is_bounded_and_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/bin/nvidia-smi")

    def fake_run(command: object, **options: object) -> object:
        calls.append((command, options))
        return SimpleNamespace(stdout="555.42\n555.42\n550.1\n")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._driver_versions() == ("550.1", "555.42")
    _, options = calls[0]  # type: ignore[misc]
    assert options["timeout"] == 3  # type: ignore[index]
    assert options["capture_output"] is True  # type: ignore[index]
