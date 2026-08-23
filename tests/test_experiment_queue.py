"""Tests for resumable candidate work leases and heartbeat recovery."""

from __future__ import annotations

import threading
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
from modelsurgeon.experiments.queue import ExperimentWorkQueue, WorkLeaseError
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
        "attempt-seed",
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


def _database_with_candidate(tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "metadata.sqlite3"
    with ExperimentMetadataStore(database) as store:
        candidate_id = store.persist_experiment(_record()).candidate_id
    return database, candidate_id


def test_crashed_worker_lease_expires_and_stale_token_cannot_complete(tmp_path: Path) -> None:
    database, candidate_id = _database_with_candidate(tmp_path)
    with (
        ExperimentMetadataStore(database) as first_store,
        ExperimentMetadataStore(database) as second_store,
    ):
        first = ExperimentWorkQueue(
            first_store,
            lease_duration_ns=10,
            token_factory=lambda: "token-first",
        )
        second = ExperimentWorkQueue(
            second_store,
            lease_duration_ns=10,
            token_factory=lambda: "token-second",
        )
        lease = first.claim(
            candidate_id,
            attempt_id="attempt-1",
            worker_id="worker-1",
            now_ns=100,
        )
        assert lease is not None
        assert second.claim(
            candidate_id,
            attempt_id="attempt-2",
            worker_id="worker-2",
            now_ns=109,
        ) is None

        takeover = second.claim(
            candidate_id,
            attempt_id="attempt-2",
            worker_id="worker-2",
            now_ns=110,
        )
        assert takeover is not None
        assert takeover.generation == 2
        assert takeover.lease_token == "token-second"
        with pytest.raises(WorkLeaseError, match="not current"):
            first.complete("token-first", now_ns=111)


def test_heartbeat_extends_lease_and_completion_is_idempotent(tmp_path: Path) -> None:
    database, candidate_id = _database_with_candidate(tmp_path)
    with ExperimentMetadataStore(database) as store:
        queue = ExperimentWorkQueue(
            store,
            lease_duration_ns=10,
            token_factory=lambda: "token",
        )
        lease = queue.claim(
            candidate_id,
            attempt_id="attempt-1",
            worker_id="worker-1",
            now_ns=100,
        )
        assert lease is not None
        heartbeat = queue.heartbeat("token", now_ns=105)
        assert heartbeat.expires_at_ns == 115
        assert queue.claim(
            candidate_id,
            attempt_id="attempt-2",
            worker_id="worker-2",
            now_ns=110,
        ) is None

        completed = queue.complete("token", now_ns=111)
        assert completed.completed_at_ns == 111
        repeated = queue.complete("token", now_ns=112)
        assert repeated == completed
        assert queue.claim(
            candidate_id,
            attempt_id="attempt-3",
            worker_id="worker-3",
            now_ns=1000,
        ) is None


def test_expired_lease_cannot_heartbeat_or_complete(tmp_path: Path) -> None:
    database, candidate_id = _database_with_candidate(tmp_path)
    with ExperimentMetadataStore(database) as store:
        queue = ExperimentWorkQueue(
            store,
            lease_duration_ns=10,
            token_factory=lambda: "token",
        )
        assert queue.claim(
            candidate_id,
            attempt_id="attempt-1",
            worker_id="worker-1",
            now_ns=100,
        ) is not None
        with pytest.raises(WorkLeaseError, match="expired before heartbeat"):
            queue.heartbeat("token", now_ns=110)
        with pytest.raises(WorkLeaseError, match="expired lease cannot complete"):
            queue.complete("token", now_ns=110)


def test_two_concurrent_workers_cannot_both_claim_current_lease(tmp_path: Path) -> None:
    database, candidate_id = _database_with_candidate(tmp_path)
    first_store = ExperimentMetadataStore(database)
    second_store = ExperimentMetadataStore(database)
    barrier = threading.Barrier(2)
    results: list[str | None] = []
    errors: list[BaseException] = []

    def claim(queue: ExperimentWorkQueue, attempt_id: str, worker_id: str) -> None:
        try:
            barrier.wait()
            lease = queue.claim(
                candidate_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                now_ns=100,
            )
            results.append(None if lease is None else lease.lease_token)
        except BaseException as error:
            errors.append(error)

    try:
        first = ExperimentWorkQueue(
            first_store,
            lease_duration_ns=10,
            token_factory=lambda: "token-first",
        )
        second = ExperimentWorkQueue(
            second_store,
            lease_duration_ns=10,
            token_factory=lambda: "token-second",
        )
        threads = (
            threading.Thread(target=claim, args=(first, "attempt-1", "worker-1")),
            threading.Thread(target=claim, args=(second, "attempt-2", "worker-2")),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(results) == 2
        assert sum(item is not None for item in results) == 1
    finally:
        first_store.close()
        second_store.close()


def test_unknown_candidate_and_backward_time_fail_closed(tmp_path: Path) -> None:
    database, candidate_id = _database_with_candidate(tmp_path)
    with ExperimentMetadataStore(database) as store:
        queue = ExperimentWorkQueue(
            store,
            lease_duration_ns=10,
            token_factory=lambda: "token",
        )
        with pytest.raises(WorkLeaseError, match="unknown experiment candidate"):
            queue.claim(
                "cand_missing",
                attempt_id="attempt-1",
                worker_id="worker-1",
                now_ns=100,
            )
        assert queue.claim(
            candidate_id,
            attempt_id="attempt-1",
            worker_id="worker-1",
            now_ns=100,
        ) is not None
        queue.heartbeat("token", now_ns=105)
        with pytest.raises(WorkLeaseError, match="moved backwards"):
            queue.heartbeat("token", now_ns=104)
