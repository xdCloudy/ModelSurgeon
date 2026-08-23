"""Tests for bounded CUDA/host OOM recovery and campaign isolation."""

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
from modelsurgeon.experiments.gpu_cleanup import ExperimentGPUCleanup
from modelsurgeon.experiments.oom_recovery import (
    OOMAdaptationAction,
    OOMAttemptConfig,
    OOMKind,
    OOMRecoveryError,
    OOMRetryPolicy,
    adapt_oom_config,
    classify_oom,
    run_with_oom_recovery,
)
from modelsurgeon.experiments.queue import ExperimentWorkQueue
from modelsurgeon.experiments.state_machine import (
    CandidateState,
    CandidateWorkStage,
    ExperimentStateMachine,
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
        MemoryInventory(2048, 1024),
        DiskInventory("/tmp", 4096, 2048),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _record(index: int) -> ExperimentRecord:
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
    run = derive_run_identity(identity.experiment_id, f"run-{index}")
    component = ComponentId.parse(f"model.layers.{index}.mlp.up_proj")
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
        f"attempt-{index}",
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


def _active_candidate(
    store: ExperimentMetadataStore,
    index: int,
    stage: CandidateWorkStage,
) -> tuple[ExperimentStateMachine, str]:
    candidate_id = store.persist_experiment(_record(index)).candidate_id
    machine = ExperimentStateMachine(store)
    machine.initialize(candidate_id)
    machine.transition(candidate_id, CandidateState.RUNNING)
    if stage is CandidateWorkStage.EVALUATION:
        machine.transition(candidate_id, CandidateState.EVALUATING)
    return machine, candidate_id


def _cleanup() -> ExperimentGPUCleanup:
    return ExperimentGPUCleanup(rss_probe=lambda: 100, collect_garbage=lambda: 0)


def test_default_classifier_distinguishes_cuda_host_and_non_oom() -> None:
    assert classify_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")) is OOMKind.CUDA
    assert classify_oom(MemoryError("host allocation")) is OOMKind.HOST
    assert classify_oom(OSError("Cannot allocate memory")) is OOMKind.HOST
    assert classify_oom(RuntimeError("ordinary evaluation failure")) is None


def test_adaptation_order_is_batch_then_chunk_then_offload() -> None:
    policy = OOMRetryPolicy(max_retries=10, min_batch_size=1, min_chunk_size=2)
    config = OOMAttemptConfig(4, 4)

    action, config = adapt_oom_config(config, policy) or pytest.fail("expected adaptation")
    assert action is OOMAdaptationAction.REDUCE_BATCH
    assert config == OOMAttemptConfig(2, 4)

    action, config = adapt_oom_config(config, policy) or pytest.fail("expected adaptation")
    assert action is OOMAdaptationAction.REDUCE_BATCH
    assert config == OOMAttemptConfig(1, 4)

    action, config = adapt_oom_config(config, policy) or pytest.fail("expected adaptation")
    assert action is OOMAdaptationAction.REDUCE_CHUNK
    assert config == OOMAttemptConfig(1, 2)

    action, config = adapt_oom_config(config, policy) or pytest.fail("expected adaptation")
    assert action is OOMAdaptationAction.ENABLE_OFFLOAD
    assert config == OOMAttemptConfig(1, 2, True)
    assert adapt_oom_config(config, policy) is None


def test_cuda_oom_retries_records_state_and_allows_none_result(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    machine, candidate_id = _active_candidate(store, 0, CandidateWorkStage.MUTATION)
    calls: list[OOMAttemptConfig] = []
    heartbeat_calls = 0

    def heartbeat() -> None:
        nonlocal heartbeat_calls
        heartbeat_calls += 1

    def operation(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> None:
        calls.append(config)
        cleanup.own_cache([])
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")

    try:
        result = run_with_oom_recovery(
            machine,
            candidate_id,
            CandidateWorkStage.MUTATION,
            OOMAttemptConfig(8, 16),
            OOMRetryPolicy(max_retries=2),
            _cleanup,
            operation,
            lease_heartbeat=heartbeat,
        )

        assert result.succeeded
        assert result.value is None
        assert result.attempts == 2
        assert result.final_config == OOMAttemptConfig(4, 16)
        assert result.events[0].kind is OOMKind.CUDA
        assert result.events[0].action is OOMAdaptationAction.REDUCE_BATCH
        assert len(result.cleanup_reports) == 2
        assert heartbeat_calls == 3
        assert machine.history(candidate_id) == (
            CandidateState.PLANNED,
            CandidateState.RUNNING,
            CandidateState.RECOVERABLE_OOM,
            CandidateState.RUNNING,
        )
        details = tuple(event.detail for event in store.list_states(candidate_id))
        assert any(detail is not None and detail.startswith("oom:") for detail in details)
        assert any(detail is not None and detail.startswith("oom-retry:") for detail in details)
    finally:
        store.close()


def test_host_oom_resumes_evaluation_stage(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    machine, candidate_id = _active_candidate(store, 0, CandidateWorkStage.EVALUATION)
    attempts = 0

    def operation(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> str:
        nonlocal attempts
        del config, cleanup
        attempts += 1
        if attempts == 1:
            raise MemoryError("host memory exhausted")
        return "ok"

    try:
        result = run_with_oom_recovery(
            machine,
            candidate_id,
            CandidateWorkStage.EVALUATION,
            OOMAttemptConfig(2, 8),
            OOMRetryPolicy(max_retries=1),
            _cleanup,
            operation,
        )
        assert result.value == "ok"
        assert result.events[0].kind is OOMKind.HOST
        assert machine.current(candidate_id) is CandidateState.EVALUATING
        assert machine.recovery_plan(candidate_id).next_stage is CandidateWorkStage.EVALUATION
    finally:
        store.close()


def test_retry_exhaustion_fails_only_current_candidate(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    first_machine, first_id = _active_candidate(store, 0, CandidateWorkStage.MUTATION)
    second_machine, second_id = _active_candidate(store, 1, CandidateWorkStage.MUTATION)

    def always_oom(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> str:
        del config, cleanup
        raise RuntimeError("CUDA out of memory")

    try:
        result = run_with_oom_recovery(
            first_machine,
            first_id,
            CandidateWorkStage.MUTATION,
            OOMAttemptConfig(4, 1),
            OOMRetryPolicy(
                max_retries=2,
                min_batch_size=1,
                min_chunk_size=1,
                allow_offload=False,
            ),
            _cleanup,
            always_oom,
        )
        assert not result.succeeded
        assert result.attempts == 3
        assert result.exhausted_kind is OOMKind.CUDA
        assert tuple(event.action for event in result.events) == (
            OOMAdaptationAction.REDUCE_BATCH,
            OOMAdaptationAction.REDUCE_BATCH,
            None,
        )
        assert first_machine.current(first_id) is CandidateState.FAILED
        assert second_machine.current(second_id) is CandidateState.RUNNING
        final_detail = store.list_states(first_id)[-1].detail
        assert final_detail is not None and final_detail.startswith("oom-exhausted:")
    finally:
        store.close()


def test_non_oom_exception_propagates_without_recovery_transition(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    machine, candidate_id = _active_candidate(store, 0, CandidateWorkStage.MUTATION)

    def fail(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> None:
        del config, cleanup
        raise ValueError("not an OOM")

    try:
        with pytest.raises(ValueError, match="not an OOM"):
            run_with_oom_recovery(
                machine,
                candidate_id,
                CandidateWorkStage.MUTATION,
                OOMAttemptConfig(2, 2),
                OOMRetryPolicy(),
                _cleanup,
                fail,
            )
        assert machine.current(candidate_id) is CandidateState.RUNNING
    finally:
        store.close()


def test_real_queue_heartbeat_can_hold_lease_across_retry(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    machine, candidate_id = _active_candidate(store, 0, CandidateWorkStage.MUTATION)
    queue = ExperimentWorkQueue(
        store,
        lease_duration_ns=100,
        token_factory=lambda: "lease-token",
    )
    lease = queue.claim(
        candidate_id,
        attempt_id="worker-attempt",
        worker_id="worker-1",
        now_ns=0,
    )
    assert lease is not None
    now = 0
    attempts = 0

    def heartbeat() -> None:
        nonlocal now
        now += 10
        queue.heartbeat("lease-token", now_ns=now)

    def operation(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> str:
        nonlocal attempts
        del config, cleanup
        attempts += 1
        if attempts == 1:
            raise MemoryError("host OOM")
        return "done"

    try:
        result = run_with_oom_recovery(
            machine,
            candidate_id,
            CandidateWorkStage.MUTATION,
            OOMAttemptConfig(2, 2),
            OOMRetryPolicy(max_retries=1),
            _cleanup,
            operation,
            lease_heartbeat=heartbeat,
        )
        assert result.value == "done"
        current = queue.current(candidate_id)
        assert current is not None
        assert current.heartbeat_at_ns == 30
        assert not current.completed
    finally:
        store.close()


def test_invalid_minima_fail_before_work_or_state_change(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    machine, candidate_id = _active_candidate(store, 0, CandidateWorkStage.MUTATION)
    called = False

    def operation(config: OOMAttemptConfig, cleanup: ExperimentGPUCleanup) -> None:
        nonlocal called
        del config, cleanup
        called = True

    try:
        with pytest.raises(OOMRecoveryError, match="min_batch_size"):
            run_with_oom_recovery(
                machine,
                candidate_id,
                CandidateWorkStage.MUTATION,
                OOMAttemptConfig(2, 2),
                OOMRetryPolicy(min_batch_size=4),
                _cleanup,
                operation,
            )
        assert not called
        assert machine.current(candidate_id) is CandidateState.RUNNING
    finally:
        store.close()
