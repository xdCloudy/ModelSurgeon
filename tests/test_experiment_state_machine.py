"""Tests for atomic resumable experiment candidate state transitions."""

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
    HardwareInventory,
    MemoryInventory,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.state_machine import (
    CandidateState,
    CandidateWorkStage,
    ExperimentStateError,
    ExperimentStateMachine,
    _atomic_append,
)
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


def _hardware() -> HardwareInventory:
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 4),
        MemoryInventory(1024, 512),
        DiskInventory("/tmp", 4096, 2048),
        CUDAInventory(False, None, (), ()),
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
        _hardware(),
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
    )


def _machine(tmp_path: Path) -> tuple[ExperimentMetadataStore, ExperimentStateMachine, str]:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    candidate_id = store.persist_experiment(_record()).candidate_id
    return store, ExperimentStateMachine(store), candidate_id


def test_valid_lifecycle_reaches_terminal_state(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        planned = machine.initialize(candidate_id)
        assert (planned.sequence, planned.state) == (0, CandidateState.PLANNED.value)
        machine.transition(candidate_id, CandidateState.RUNNING)
        machine.transition(candidate_id, CandidateState.EVALUATING)
        machine.transition(candidate_id, CandidateState.SUCCEEDED)

        plan = machine.recovery_plan(candidate_id)
        assert plan.current_state is CandidateState.SUCCEEDED
        assert plan.next_stage is None
        assert plan.terminal is True
        assert machine.history(candidate_id) == (
            CandidateState.PLANNED,
            CandidateState.RUNNING,
            CandidateState.EVALUATING,
            CandidateState.SUCCEEDED,
        )


def test_invalid_transition_is_rejected_without_event(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        machine.initialize(candidate_id)
        with pytest.raises(ExperimentStateError, match="planned -> evaluating"):
            machine.transition(candidate_id, CandidateState.EVALUATING)
        assert machine.history(candidate_id) == (CandidateState.PLANNED,)


def test_restart_resumes_evaluation_without_repeating_mutation(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        machine.initialize(candidate_id)
        machine.transition(candidate_id, CandidateState.RUNNING)
        machine.transition(candidate_id, CandidateState.EVALUATING)
        machine.transition(candidate_id, CandidateState.INTERRUPTED, "worker shutdown")

        plan = machine.recovery_plan(candidate_id)
        assert plan.next_stage is CandidateWorkStage.EVALUATION
        assert plan.resume is True
        machine.transition(candidate_id, CandidateState.EVALUATING, "resumed")
        machine.transition(candidate_id, CandidateState.SUCCEEDED)


def test_recovery_cannot_jump_back_to_completed_mutation_stage(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        machine.initialize(candidate_id)
        machine.transition(candidate_id, CandidateState.RUNNING)
        machine.transition(candidate_id, CandidateState.EVALUATING)
        machine.transition(candidate_id, CandidateState.RECOVERABLE_OOM)

        with pytest.raises(ExperimentStateError, match="return to evaluating"):
            machine.transition(candidate_id, CandidateState.RUNNING)
        assert machine.recovery_plan(candidate_id).next_stage is CandidateWorkStage.EVALUATION


def test_running_restart_repeats_only_unfinished_mutation_stage(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        machine.initialize(candidate_id)
        machine.transition(candidate_id, CandidateState.RUNNING)
        plan = machine.recovery_plan(candidate_id)
        assert plan.next_stage is CandidateWorkStage.MUTATION
        assert plan.resume is True


def test_compare_and_append_rejects_stale_expected_state(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        machine.initialize(candidate_id)
        machine.transition(candidate_id, CandidateState.RUNNING)
        with pytest.raises(ExperimentStateError, match="changed concurrently"):
            _atomic_append(
                store,
                candidate_id,
                CandidateState.PLANNED,
                CandidateState.REJECTED,
                None,
            )
        assert machine.current(candidate_id) is CandidateState.RUNNING


def test_uninitialized_or_unknown_candidate_fails_closed(tmp_path: Path) -> None:
    store, machine, candidate_id = _machine(tmp_path)
    with store:
        with pytest.raises(ExperimentStateError, match="not initialized"):
            machine.recovery_plan(candidate_id)
        with pytest.raises(ExperimentStateError, match="unknown experiment candidate"):
            machine.initialize("cand_missing")
