"""Tests for append-only per-stage runtime and resource telemetry."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentIdentitySpec,
    ExperimentMetadataStore,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    ExperimentStoreError,
    GPUDeviceInventory,
    HardwareInventory,
    HardwareNormalizationContext,
    MemoryInventory,
    ModelTarget,
    ProcessIOCounters,
    SeedContext,
    SoftwareInventory,
    StageTelemetryError,
    StageTelemetryRecorder,
    StageTelemetrySnapshot,
    StageTelemetryState,
    StageThroughput,
    VersionContext,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation.memory_telemetry import MemoryTelemetryConfig
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
    MutationRunRecord,
)


class SimulatedInterrupt(BaseException):
    pass


class SequenceClock:
    def __init__(self, values: Iterable[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class SequenceIO:
    def __init__(self, values: Iterable[ProcessIOCounters | None]) -> None:
        self._values = iter(values)

    def __call__(self) -> ProcessIOCounters | None:
        return next(self._values)


class FakeCudaMemoryProvider:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_peak_stats(self) -> None:
        self.reset_calls += 1

    def allocated_bytes(self) -> int:
        return 128

    def reserved_bytes(self) -> int:
        return 192

    def max_allocated_bytes(self) -> int:
        return 256

    def max_reserved_bytes(self) -> int:
        return 512


def _hardware(*, cuda: bool = False, logical_cores: int = 8) -> HardwareInventory:
    devices = (
        (GPUDeviceInventory(0, "fixture-gpu", 1_000_000, "8.6"),) if cuda else ()
    )
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", logical_cores),
        MemoryInventory(4096, 2048),
        DiskInventory("/tmp", 8192, 4096),
        CUDAInventory(cuda, "12.8" if cuda else None, ("590.1",) if cuda else (), devices),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _record(hardware: HardwareInventory | None = None) -> ExperimentRecord:
    model = ModelTarget("tiny/model", "model-rev", "llama", "safetensors", 128)
    dataset = DatasetTarget(
        "tiny/data",
        "data-rev",
        "validation",
        "manifest-1",
        "tiny/tokenizer",
        "tokenizer-rev",
    )
    seeds = SeedContext(1, 2, 3)
    identity = derive_experiment_identity(
        ExperimentIdentitySpec(model, dataset, {}, seeds, "tool", "1", 1, 1)
    )
    run = derive_run_identity(identity.experiment_id)
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-1)
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(model.revision, "tool"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return ExperimentRecord(
        run.run_id,
        identity.experiment_id,
        "attempt-1",
        model,
        dataset,
        (component,),
        mutation,
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        hardware or _hardware(),
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
    )


def _recorder(
    store: ExperimentMetadataStore,
    candidate_id: str,
    *,
    hardware: HardwareInventory | None = None,
    wall_values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    cpu_values: tuple[float, ...] = (1.0, 1.25),
    io_values: tuple[ProcessIOCounters | None, ...] = (
        ProcessIOCounters(100, 200),
        ProcessIOCounters(160, 260),
    ),
    cuda: FakeCudaMemoryProvider | None = None,
) -> StageTelemetryRecorder:
    return StageTelemetryRecorder(
        store,
        candidate_id,
        hardware or _hardware(cuda=cuda is not None),
        MemoryTelemetryConfig(sampling_enabled=False),
        cuda=cuda,
        monotonic=SequenceClock(wall_values),
        process_time=SequenceClock(cpu_values),
        io_counters=SequenceIO(io_values),
    )


def test_completed_stage_persists_timing_throughput_memory_and_io(tmp_path: Path) -> None:
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        persisted = store.persist_experiment(_record())
        execution = _recorder(store, persisted.candidate_id).run(
            "evaluation",
            lambda: "ok",
            throughput=lambda: StageThroughput(tokens=40, candidates=2),
        )

        assert execution.result == "ok"
        stored = execution.telemetry
        assert stored.attempt == 0
        assert stored.state == StageTelemetryState.COMPLETE.value
        assert stored.wall_seconds == pytest.approx(0.4)
        assert stored.cpu_seconds == pytest.approx(0.25)
        assert (stored.tokens, stored.candidates) == (40, 2)
        assert (stored.io_read_bytes, stored.io_write_bytes) == (60, 60)
        assert stored.peak_rss_bytes is None or stored.peak_rss_bytes >= 0
        assert stored.peak_cuda_allocated_bytes is None
        assert stored.peak_cuda_reserved_bytes is None
        assert stored.hardware_context_id.startswith("hwctx_")
        assert store.list_stage_telemetry(persisted.candidate_id) == (stored,)
        assert store.latest_stage_telemetry(persisted.candidate_id) == (stored,)


def test_interrupted_stage_retains_partial_attempt_and_resume_appends(tmp_path: Path) -> None:
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        persisted = store.persist_experiment(_record())

        with pytest.raises(SimulatedInterrupt):
            _recorder(
                store,
                persisted.candidate_id,
                wall_values=(0.0, 0.1, 0.2, 0.3, 0.35),
                cpu_values=(2.0, 2.1),
                io_values=(ProcessIOCounters(10, 20), ProcessIOCounters(15, 28)),
            ).run(
                "mutation",
                lambda: (_ for _ in ()).throw(SimulatedInterrupt("worker interrupted")),
                throughput=lambda: StageThroughput(tokens=8, candidates=0),
            )

        resumed = _recorder(
            store,
            persisted.candidate_id,
            wall_values=(3.0, 3.1, 3.2, 3.3, 3.5),
            cpu_values=(4.0, 4.2),
            io_values=(ProcessIOCounters(20, 30), ProcessIOCounters(32, 50)),
        ).run(
            "mutation",
            lambda: "resumed",
            throughput=lambda: StageThroughput(tokens=16, candidates=1),
        )

        attempts = store.list_stage_telemetry(persisted.candidate_id, stage="mutation")
        assert [item.attempt for item in attempts] == [0, 1]
        assert [item.state for item in attempts] == [
            StageTelemetryState.PARTIAL.value,
            StageTelemetryState.COMPLETE.value,
        ]
        assert attempts[0].tokens == 8
        assert attempts[0].io_read_bytes == 5
        assert resumed.result == "resumed"
        assert resumed.telemetry == attempts[1]
        assert store.latest_stage_telemetry(persisted.candidate_id) == (attempts[1],)


def test_cuda_peaks_are_persisted_without_fabrication(tmp_path: Path) -> None:
    hardware = _hardware(cuda=True)
    cuda = FakeCudaMemoryProvider()
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        persisted = store.persist_experiment(_record(hardware))
        execution = _recorder(
            store,
            persisted.candidate_id,
            hardware=hardware,
            cuda=cuda,
        ).run("evaluation", lambda: None)

        assert cuda.reset_calls == 1
        assert execution.telemetry.peak_cuda_allocated_bytes == 256
        assert execution.telemetry.peak_cuda_reserved_bytes == 512
        assert execution.telemetry.hardware_context["cuda_available"] is True


def test_hardware_normalization_context_is_stable_and_comparable() -> None:
    first = HardwareNormalizationContext.from_inventory(_hardware())
    equivalent = HardwareNormalizationContext.from_inventory(_hardware())
    different = HardwareNormalizationContext.from_inventory(_hardware(logical_cores=16))

    assert first.context_id == equivalent.context_id
    assert first.comparable_to(equivalent)
    assert first.context_id != different.context_id
    assert not first.comparable_to(different)


def test_cpu_only_snapshot_rejects_cuda_metrics_and_store_rejects_unknown_candidate(
    tmp_path: Path,
) -> None:
    hardware = HardwareNormalizationContext.from_inventory(_hardware())
    with pytest.raises(StageTelemetryError, match="cannot fabricate CUDA peaks"):
        StageTelemetrySnapshot(
            "evaluation",
            StageTelemetryState.COMPLETE,
            1.0,
            0.5,
            StageThroughput(),
            1024,
            1,
            None,
            None,
            None,
            hardware,
        )

    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        snapshot = StageTelemetrySnapshot(
            "evaluation",
            StageTelemetryState.COMPLETE,
            1.0,
            0.5,
            StageThroughput(candidates=1),
            1024,
            None,
            None,
            0,
            0,
            hardware,
        )
        with pytest.raises(ExperimentStoreError, match="unknown candidate"):
            store.append_stage_telemetry("cand_missing", snapshot)
