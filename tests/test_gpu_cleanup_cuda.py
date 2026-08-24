"""Real-CUDA acceptance tests for deterministic experiment cleanup."""

from __future__ import annotations

import gc
from typing import Any

import pytest

from modelsurgeon.experiments.gpu_cleanup import (
    ExperimentGPUCleanup,
    TorchCudaCleanupProvider,
)

pytestmark = pytest.mark.gpu

_MIB = 1024 * 1024
_ALLOCATED_TOLERANCE = 1 * _MIB
_RESERVED_TOLERANCE = 8 * _MIB


def _torch_cuda() -> Any:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real CUDA device required")
    return torch


def _warm_allocator(torch: Any, provider: TorchCudaCleanupProvider) -> tuple[int, int]:
    """Stabilize one-time CUDA/PyTorch allocations before measuring retention."""

    for _ in range(2):
        boundary = ExperimentGPUCleanup(cuda=provider, rss_probe=lambda: None)
        boundary.own_model(torch.empty(2 * 1024 * 1024, device="cuda", dtype=torch.float32))
        report = boundary.cleanup()
        assert report.failures == ()
        assert report.cuda_cache_cleared is True
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return provider.allocated_bytes(), provider.reserved_bytes()


def test_real_cuda_repeated_tiny_experiments_have_bounded_vram_growth() -> None:
    torch = _torch_cuda()
    provider = TorchCudaCleanupProvider(device=0)
    baseline_allocated, baseline_reserved = _warm_allocator(torch, provider)

    retained_allocated: list[int] = []
    retained_reserved: list[int] = []
    for _ in range(16):
        boundary = ExperimentGPUCleanup(cuda=provider, rss_probe=lambda: None)
        boundary.own_model(torch.empty(2 * 1024 * 1024, device="cuda", dtype=torch.float32))
        boundary.own_gradients(
            {"gradient": torch.empty(512 * 1024, device="cuda", dtype=torch.float32)}
        )
        report = boundary.cleanup()

        assert report.failures == ()
        assert report.cuda_cache_cleared is True
        assert report.retained_cuda_allocated_bytes is not None
        assert report.retained_cuda_reserved_bytes is not None
        retained_allocated.append(report.retained_cuda_allocated_bytes)
        retained_reserved.append(report.retained_cuda_reserved_bytes)

    assert max(retained_allocated) <= baseline_allocated + _ALLOCATED_TOLERANCE
    assert max(retained_reserved) <= baseline_reserved + _RESERVED_TOLERANCE
    assert retained_allocated[-1] <= baseline_allocated + _ALLOCATED_TOLERANCE
    assert retained_reserved[-1] <= baseline_reserved + _RESERVED_TOLERANCE


def test_real_cuda_cleanup_runs_after_exception_and_releases_vram() -> None:
    torch = _torch_cuda()
    provider = TorchCudaCleanupProvider(device=0)
    baseline_allocated, baseline_reserved = _warm_allocator(torch, provider)
    boundary = ExperimentGPUCleanup(cuda=provider, rss_probe=lambda: None)

    with pytest.raises(RuntimeError, match="experiment failed"), boundary:
        boundary.own_model(torch.empty(2 * 1024 * 1024, device="cuda", dtype=torch.float32))
        boundary.own_gradients(
            {"gradient": torch.empty(512 * 1024, device="cuda", dtype=torch.float32)}
        )
        raise RuntimeError("experiment failed")

    report = boundary.last_report
    assert report is not None
    assert report.failures == ()
    assert report.cuda_cache_cleared is True
    assert report.retained_cuda_allocated_bytes is not None
    assert report.retained_cuda_reserved_bytes is not None
    assert report.retained_cuda_allocated_bytes <= baseline_allocated + _ALLOCATED_TOLERANCE
    assert report.retained_cuda_reserved_bytes <= baseline_reserved + _RESERVED_TOLERANCE
