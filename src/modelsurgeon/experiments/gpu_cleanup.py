"""Deterministic worker-boundary cleanup with optional CUDA cache release."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol, Self

from modelsurgeon.instrumentation.memory_telemetry import process_rss_bytes

GPU_CLEANUP_VERSION = "1"


class GPUCleanupError(RuntimeError):
    """Raised when deterministic worker cleanup cannot complete safely."""


class CudaCleanupProvider(Protocol):
    def allocated_bytes(self) -> int: ...

    def reserved_bytes(self) -> int: ...

    def empty_cache(self) -> None: ...

    def synchronize(self) -> None: ...


class TorchCudaCleanupProvider:
    """Lazy PyTorch CUDA cleanup adapter without a hard torch dependency."""

    def __init__(self, device: int | str | None = None) -> None:
        try:
            torch: Any = import_module("torch")
        except Exception as error:
            raise GPUCleanupError("PyTorch is unavailable for CUDA cleanup") from error
        try:
            available = bool(torch.cuda.is_available())
        except Exception as error:
            raise GPUCleanupError("CUDA availability probe failed during cleanup setup") from error
        if not available:
            raise GPUCleanupError("CUDA is unavailable for GPU cleanup")
        self._torch = torch
        self._device = device

    def allocated_bytes(self) -> int:
        return int(self._torch.cuda.memory_allocated(self._device))

    def reserved_bytes(self) -> int:
        return int(self._torch.cuda.memory_reserved(self._device))

    def empty_cache(self) -> None:
        self._torch.cuda.empty_cache()

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)


@dataclass(frozen=True, slots=True)
class CleanupMemorySnapshot:
    rss_bytes: int | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None

    def __post_init__(self) -> None:
        for value in (
            self.rss_bytes,
            self.cuda_allocated_bytes,
            self.cuda_reserved_bytes,
        ):
            if value is not None and value < 0:
                raise GPUCleanupError("cleanup memory snapshots cannot contain negative values")


@dataclass(frozen=True, slots=True)
class CleanupFailure:
    resource: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.resource or not self.exception_type:
            raise GPUCleanupError("cleanup failures require resource and exception type")

    def to_record(self) -> dict[str, str]:
        return {
            "resource": self.resource,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GPUCleanupReport:
    version: str
    released_resources: tuple[str, ...]
    failures: tuple[CleanupFailure, ...]
    gc_collected: int
    before: CleanupMemorySnapshot
    after: CleanupMemorySnapshot
    cuda_cache_cleared: bool

    def __post_init__(self) -> None:
        if self.version != GPU_CLEANUP_VERSION:
            raise GPUCleanupError(f"unsupported GPU cleanup report version {self.version}")
        if self.released_resources != tuple(dict.fromkeys(self.released_resources)):
            raise GPUCleanupError("cleanup report resource names must be unique")
        if self.gc_collected < 0:
            raise GPUCleanupError("GC collected count cannot be negative")

    @property
    def retained_rss_bytes(self) -> int | None:
        return self.after.rss_bytes

    @property
    def retained_cuda_allocated_bytes(self) -> int | None:
        return self.after.cuda_allocated_bytes

    @property
    def retained_cuda_reserved_bytes(self) -> int | None:
        return self.after.cuda_reserved_bytes

    def to_record(self) -> dict[str, object]:
        def snapshot(value: CleanupMemorySnapshot) -> dict[str, int | None]:
            return {
                "rss_bytes": value.rss_bytes,
                "cuda_allocated_bytes": value.cuda_allocated_bytes,
                "cuda_reserved_bytes": value.cuda_reserved_bytes,
            }

        return {
            "version": self.version,
            "released_resources": list(self.released_resources),
            "failures": [failure.to_record() for failure in self.failures],
            "gc_collected": self.gc_collected,
            "before": snapshot(self.before),
            "after": snapshot(self.after),
            "cuda_cache_cleared": self.cuda_cache_cleared,
        }


@dataclass(slots=True)
class _OwnedResource:
    name: str
    value: object | None
    cleanup: Callable[[], None] | None


class ExperimentGPUCleanup:
    """Own experiment resources and deterministically release them at a worker boundary."""

    def __init__(
        self,
        *,
        cuda: CudaCleanupProvider | None = None,
        rss_probe: Callable[[], int | None] = process_rss_bytes,
        collect_garbage: Callable[[], int] = gc.collect,
    ) -> None:
        self.cuda = cuda
        self.rss_probe = rss_probe
        self.collect_garbage = collect_garbage
        self._resources: list[_OwnedResource] = []
        self._names: set[str] = set()
        self._cleaned = False
        self.last_report: GPUCleanupReport | None = None

    def __enter__(self) -> Self:
        if self._cleaned:
            raise GPUCleanupError("cleanup boundary cannot be reused after release")
        return self

    def own(
        self,
        name: str,
        value: object,
        *,
        cleanup: Callable[[], None] | None = None,
    ) -> object:
        """Transfer one model/hook/gradient/cache reference into this worker boundary."""

        if self._cleaned:
            raise GPUCleanupError("cannot register resources after cleanup")
        if not name:
            raise GPUCleanupError("owned resource name is required")
        if name in self._names:
            raise GPUCleanupError(f"duplicate owned resource name {name!r}")
        self._names.add(name)
        self._resources.append(_OwnedResource(name, value, cleanup))
        return value

    def own_model(self, model: object, cleanup: Callable[[], None] | None = None) -> object:
        return self.own("model", model, cleanup=cleanup)

    def own_hooks(self, hooks: object, cleanup: Callable[[], None] | None = None) -> object:
        resolved = cleanup
        if resolved is None:
            close = getattr(hooks, "close", None)
            if callable(close):
                resolved = close
        return self.own("hooks", hooks, cleanup=resolved)

    def own_gradients(
        self,
        gradients: object,
        cleanup: Callable[[], None] | None = None,
    ) -> object:
        resolved = cleanup
        if resolved is None:
            clear = getattr(gradients, "clear", None)
            if callable(clear):
                resolved = clear
        return self.own("gradients", gradients, cleanup=resolved)

    def own_cache(self, cache: object, cleanup: Callable[[], None] | None = None) -> object:
        resolved = cleanup
        if resolved is None:
            clear = getattr(cache, "clear", None)
            if callable(clear):
                resolved = clear
        return self.own("cache", cache, cleanup=resolved)

    def _snapshot(self) -> CleanupMemorySnapshot:
        allocated = None
        reserved = None
        if self.cuda is not None:
            allocated = self.cuda.allocated_bytes()
            reserved = self.cuda.reserved_bytes()
        return CleanupMemorySnapshot(self.rss_probe(), allocated, reserved)

    def cleanup(self) -> GPUCleanupReport:
        if self._cleaned:
            if self.last_report is None:
                raise GPUCleanupError("cleanup boundary is inconsistent")
            return self.last_report

        before = self._snapshot()
        failures: list[CleanupFailure] = []
        released: list[str] = []
        for resource in reversed(self._resources):
            if resource.cleanup is not None:
                try:
                    resource.cleanup()
                except BaseException as error:
                    failures.append(
                        CleanupFailure(
                            resource.name,
                            type(error).__name__,
                            str(error),
                        )
                    )
            resource.value = None
            released.append(resource.name)
        self._resources.clear()
        self._names.clear()

        try:
            collected = self.collect_garbage()
        except BaseException as error:
            failures.append(CleanupFailure("python-gc", type(error).__name__, str(error)))
            collected = 0

        cuda_cleared = False
        if self.cuda is not None:
            try:
                self.cuda.synchronize()
                self.cuda.empty_cache()
                self.cuda.synchronize()
                cuda_cleared = True
            except BaseException as error:
                failures.append(CleanupFailure("cuda-cache", type(error).__name__, str(error)))

        after = self._snapshot()
        report = GPUCleanupReport(
            GPU_CLEANUP_VERSION,
            tuple(released),
            tuple(failures),
            collected,
            before,
            after,
            cuda_cleared,
        )
        self.last_report = report
        self._cleaned = True
        return report

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc_type, traceback
        report = self.cleanup()
        if report.failures and exc_value is None:
            messages = "; ".join(
                f"{failure.resource}: {failure.exception_type}: {failure.message}"
                for failure in report.failures
            )
            raise GPUCleanupError(f"one or more experiment resources failed cleanup: {messages}")
        if report.failures and exc_value is not None:
            exc_value.add_note(
                "ModelSurgeon cleanup failures: "
                + "; ".join(failure.resource for failure in report.failures)
            )
        return False
