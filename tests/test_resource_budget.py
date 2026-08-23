"""Tests for RAM, VRAM, disk, and runtime stage budgets."""

from __future__ import annotations

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
    ExperimentStateMachine,
    GPUDeviceInventory,
    HardwareInventory,
    MemoryInventory,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.resource_budget import (
    ResourceBudgetError,
    ResourceBudgetExceeded,
    ResourceKind,
    ResourceSnapshot,
    StageResourceBudget,
    StageResourceBudgetGuard,
    StageResourceEstimate,
    preflight_resource_budget,
    run_budgeted_stage,
)
from modelsurgeon.experiments.state_machine import CandidateState, CandidateWorkStage
from modelsurgeon.graph import ComponentId
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


class SequenceProbe:
    def __init__(self, snapshots: tuple[ResourceSnapshot, ...]) -> None:
        self._snapshots = snapshots
        self._index = 0

    def snapshot(self) -> ResourceSnapshot:
        if self._index >= len(self._snapshots):
            return self._snapshots[-1]
        value = self._snapshots[self._index]
        self._index += 1
        return value


def _hardware(*, cuda: bool = False) -> HardwareInventory:
    cuda_inventory = (
        CUDAInventory(
            True,
            "12.0",
            ("test",),
            (GPUDeviceInventory(0, "test-gpu", 1000, "9.0"),),
        )
        if cuda
        else CUDAInventory(False, None, (), ())
    )
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 4),
        MemoryInventory(2000, 1000),
        DiskInventory("/tmp", 5000, 2000),
        cuda_inventory,
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _record() -> ExperimentRecord:
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
    delta = MutationDelta(parameters=-1)
    mutation = MutationRunRecord(
        MutationPlan(
            MutationRequest(MutationKind.MASK, (component,)),
            (component,),
            (),
            delta,
        ),
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
        _hardware(),
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
    )


def _active_machine(
    tmp_path: Path,
    stage: CandidateWorkStage,
) -> tuple[ExperimentMetadataStore, ExperimentStateMachine, str]:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    candidate_id = store.persist_experiment(_record()).candidate_id
    machine = ExperimentStateMachine(store)
    machine.initialize(candidate_id)
    machine.transition(candidate_id, CandidateState.RUNNING)
    if stage is CandidateWorkStage.EVALUATION:
        machine.transition(candidate_id, CandidateState.EVALUATING)
    return store, machine, candidate_id


def test_cpu_only_preflight_accepts_bounded_ram_disk_and_runtime() -> None:
    preflight_resource_budget(
        "mutation",
        StageResourceBudget(
            max_ram_bytes=500,
            max_disk_bytes=1000,
            max_runtime_seconds=10,
        ),
        _hardware(),
        StageResourceEstimate(
            ram_bytes=400,
            disk_bytes=900,
            runtime_seconds=5,
        ),
    )


def test_preflight_rejects_estimate_over_budget_or_host_capacity() -> None:
    with pytest.raises(ResourceBudgetExceeded) as captured:
        preflight_resource_budget(
            "mutation",
            StageResourceBudget(max_ram_bytes=100),
            _hardware(),
            StageResourceEstimate(ram_bytes=101),
        )
    assert captured.value.violation.resource is ResourceKind.RAM

    with pytest.raises(ResourceBudgetExceeded) as captured_disk:
        preflight_resource_budget(
            "mutation",
            StageResourceBudget(),
            _hardware(),
            StageResourceEstimate(disk_bytes=2001),
        )
    assert captured_disk.value.violation.resource is ResourceKind.DISK


def test_cuda_budget_is_supported_and_cpu_only_vram_request_fails_explicitly() -> None:
    with pytest.raises(ResourceBudgetError, match="CPU-only"):
        preflight_resource_budget(
            "evaluation",
            StageResourceBudget(max_vram_bytes=100),
            _hardware(cuda=False),
            StageResourceEstimate(vram_bytes=10),
        )

    guard = StageResourceBudgetGuard(
        "evaluation",
        StageResourceBudget(max_vram_bytes=40),
        _hardware(cuda=True),
        StageResourceEstimate(vram_bytes=20),
        SequenceProbe(
            (
                ResourceSnapshot(100, 100, 1000, 0.0),
                ResourceSnapshot(100, 150, 1000, 1.0),
            )
        ),
    )
    with pytest.raises(ResourceBudgetExceeded) as captured:
        with guard:
            guard.check()
    assert captured.value.violation.resource is ResourceKind.VRAM


def test_mutation_budget_violation_persists_resumable_interruption(tmp_path: Path) -> None:
    store, machine, candidate_id = _active_machine(tmp_path, CandidateWorkStage.MUTATION)
    guard = StageResourceBudgetGuard(
        "mutation",
        StageResourceBudget(max_ram_bytes=10),
        _hardware(),
        StageResourceEstimate(ram_bytes=5),
        SequenceProbe(
            (
                ResourceSnapshot(100, None, 1000, 0.0),
                ResourceSnapshot(120, None, 1000, 1.0),
            )
        ),
    )
    try:
        with pytest.raises(ResourceBudgetExceeded):
            run_budgeted_stage(
                machine,
                candidate_id,
                CandidateWorkStage.MUTATION,
                guard,
                lambda active: active.check(),
            )
        assert machine.current(candidate_id) is CandidateState.INTERRUPTED
        assert machine.recovery_plan(candidate_id).next_stage is CandidateWorkStage.MUTATION
    finally:
        store.close()


def test_final_runtime_check_preserves_evaluation_resume_point(tmp_path: Path) -> None:
    store, machine, candidate_id = _active_machine(tmp_path, CandidateWorkStage.EVALUATION)
    guard = StageResourceBudgetGuard(
        "evaluation",
        StageResourceBudget(max_runtime_seconds=5),
        _hardware(),
        StageResourceEstimate(runtime_seconds=4),
        SequenceProbe(
            (
                ResourceSnapshot(100, None, 1000, 0.0),
                ResourceSnapshot(100, None, 1000, 6.0),
            )
        ),
    )
    try:
        with pytest.raises(ResourceBudgetExceeded) as captured:
            run_budgeted_stage(
                machine,
                candidate_id,
                CandidateWorkStage.EVALUATION,
                guard,
                lambda _active: "result",
            )
        assert captured.value.violation.resource is ResourceKind.RUNTIME
        assert machine.current(candidate_id) is CandidateState.INTERRUPTED
        assert machine.recovery_plan(candidate_id).next_stage is CandidateWorkStage.EVALUATION
    finally:
        store.close()


def test_disk_checkpoint_uses_stage_local_consumption() -> None:
    guard = StageResourceBudgetGuard(
        "export",
        StageResourceBudget(max_disk_bytes=100),
        _hardware(),
        StageResourceEstimate(disk_bytes=50),
        SequenceProbe(
            (
                ResourceSnapshot(100, None, 1000, 0.0),
                ResourceSnapshot(100, None, 899, 1.0),
            )
        ),
    )
    with pytest.raises(ResourceBudgetExceeded) as captured:
        with guard:
            guard.check()
    assert captured.value.violation.resource is ResourceKind.DISK
    assert captured.value.violation.observed == 101
