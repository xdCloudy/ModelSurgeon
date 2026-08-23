"""Best-effort, framework-optional reproducibility hardware inventory."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

HARDWARE_INVENTORY_SCHEMA_VERSION: Literal[1] = 1


@dataclass(frozen=True, slots=True)
class CPUInventory:
    architecture: str
    processor: str
    logical_cores: int | None

    def to_record(self) -> dict[str, str | int | None]:
        return {
            "architecture": self.architecture,
            "processor": self.processor,
            "logical_cores": self.logical_cores,
        }


@dataclass(frozen=True, slots=True)
class MemoryInventory:
    total_bytes: int | None
    available_bytes: int | None

    def to_record(self) -> dict[str, int | None]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
        }


@dataclass(frozen=True, slots=True)
class DiskInventory:
    path: str
    total_bytes: int
    free_bytes: int

    def to_record(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
        }


@dataclass(frozen=True, slots=True)
class GPUDeviceInventory:
    index: int
    name: str
    total_memory_bytes: int | None
    compute_capability: str | None

    def to_record(self) -> dict[str, str | int | None]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_capability": self.compute_capability,
        }


@dataclass(frozen=True, slots=True)
class CUDAInventory:
    available: bool
    compiled_version: str | None
    driver_versions: tuple[str, ...]
    devices: tuple[GPUDeviceInventory, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "available": self.available,
            "compiled_version": self.compiled_version,
            "driver_versions": list(self.driver_versions),
            "devices": [device.to_record() for device in self.devices],
        }


@dataclass(frozen=True, slots=True)
class SoftwareInventory:
    python_version: str
    python_implementation: str
    modelsurgeon_version: str | None
    pytorch_version: str | None

    def to_record(self) -> dict[str, str | None]:
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "modelsurgeon_version": self.modelsurgeon_version,
            "pytorch_version": self.pytorch_version,
        }


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    os_name: str
    os_release: str
    os_version: str
    cpu: CPUInventory
    memory: MemoryInventory
    disk: DiskInventory
    cuda: CUDAInventory
    software: SoftwareInventory
    warnings: tuple[str, ...] = ()
    schema_version: Literal[1] = HARDWARE_INVENTORY_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "os": {
                "name": self.os_name,
                "release": self.os_release,
                "version": self.os_version,
            },
            "cpu": self.cpu.to_record(),
            "memory": self.memory.to_record(),
            "disk": self.disk.to_record(),
            "cuda": self.cuda.to_record(),
            "software": self.software.to_record(),
            "warnings": list(self.warnings),
        }


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _memory_inventory() -> MemoryInventory:
    if sys.platform == "win32":
        status = _WindowsMemoryStatus()
        status.length = ctypes.sizeof(status)
        kernel32: Any = ctypes.windll.kernel32
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return MemoryInventory(
                int(status.total_physical),
                int(status.available_physical),
            )
        return MemoryInventory(None, None)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return MemoryInventory(None, None)
    return MemoryInventory(page_size * total_pages, page_size * available_pages)


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _driver_versions() -> tuple[str, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(sorted({line.strip() for line in result.stdout.splitlines() if line.strip()}))


def _cuda_inventory(pytorch_version: str | None) -> tuple[CUDAInventory, tuple[str, ...]]:
    if pytorch_version is None:
        return CUDAInventory(False, None, _driver_versions(), ()), ()
    try:
        torch: Any = import_module("torch")
    except Exception as error:  # a broken optional runtime must not invalidate CPU inventory
        warning = f"PyTorch {pytorch_version} could not be imported: {type(error).__name__}"
        return CUDAInventory(False, None, _driver_versions(), ()), (warning,)
    compiled = getattr(getattr(torch, "version", None), "cuda", None)
    compiled_version = str(compiled) if compiled is not None else None
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:
        warning = f"CUDA availability probe failed: {type(error).__name__}"
        return CUDAInventory(False, compiled_version, _driver_versions(), ()), (warning,)
    if not available:
        return CUDAInventory(False, compiled_version, _driver_versions(), ()), ()

    devices: list[GPUDeviceInventory] = []
    warnings: list[str] = []
    try:
        count = int(torch.cuda.device_count())
    except Exception as error:
        warnings.append(f"CUDA device count probe failed: {type(error).__name__}")
        count = 0
    for index in range(max(0, count)):
        try:
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            devices.append(
                GPUDeviceInventory(
                    index,
                    str(properties.name),
                    int(properties.total_memory),
                    f"{int(capability[0])}.{int(capability[1])}",
                )
            )
        except Exception as error:
            warnings.append(f"CUDA device {index} probe failed: {type(error).__name__}")
    return (
        CUDAInventory(True, compiled_version, _driver_versions(), tuple(devices)),
        tuple(warnings),
    )


def collect_hardware_inventory(disk_path: str | Path = ".") -> HardwareInventory:
    """Collect a valid CPU-first record and optional CUDA details without requiring CUDA."""
    resolved_disk = Path(disk_path).resolve()
    usage = shutil.disk_usage(resolved_disk)
    pytorch_version = _installed_version("torch")
    cuda, warnings = _cuda_inventory(pytorch_version)
    uname = platform.uname()
    return HardwareInventory(
        os_name=uname.system,
        os_release=uname.release,
        os_version=uname.version,
        cpu=CPUInventory(
            architecture=uname.machine,
            processor=uname.processor or platform.processor() or "unknown",
            logical_cores=os.cpu_count(),
        ),
        memory=_memory_inventory(),
        disk=DiskInventory(str(resolved_disk), usage.total, usage.free),
        cuda=cuda,
        software=SoftwareInventory(
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            modelsurgeon_version=_installed_version("modelsurgeon"),
            pytorch_version=pytorch_version,
        ),
        warnings=warnings,
    )
