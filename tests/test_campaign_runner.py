"""Tests for resumable automated mutation campaign execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.evaluation.baseline_cache import BaselineArtifact, BaselineCache
from modelsurgeon.evaluation.numerics import Tier0NumericsResult
from modelsurgeon.evaluation.tier0 import Tier0Stage, Tier0ValidationResult
from modelsurgeon.evaluation.tiered import EvaluationTier, TieredEvaluationConfig
from modelsurgeon.experiments.campaign import (
    CampaignContext,
    CampaignError,
    CampaignOutcome,
    CampaignRunner,
    CampaignRunnerConfig,
    MutationCheckpoint,
    campaign_status,
    query_campaign_progress,
    register_campaign,
)
from modelsurgeon.experiments.candidates import (
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    MutationCandidate,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.gpu_cleanup import ExperimentGPUCleanup
from modelsurgeon.experiments.hardware import (
    CPUInventory,
    CUDAInventory,
    DiskInventory,
    HardwareInventory,
    MemoryInventory,
    SoftwareInventory,
)
from modelsurgeon.experiments.identity import (
    ExperimentIdentitySpec,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.oom_recovery import OOMAttemptConfig, OOMRetryPolicy
from modelsurgeon.experiments.schema import (
    DatasetTarget,
    ModelTarget,
    SeedContext,
    VersionContext,
)
from modelsurgeon.experiments.state_machine import CandidateState, ExperimentStateMachine
from modelsurgeon.experiments.store import ExperimentMetadataStore
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode


class SimulatedCrash(BaseException):
    pass


class Tier0PassBackend:
    def run_tier0_load(self) -> Tier0ValidationResult:
        return Tier0ValidationResult(
            True,
            tuple(Tier0Stage),
            None,
            None,
            None,
            "cpu",
            32,
        )

    def run_tier0_numerics(self) -> Tier0NumericsResult:
        return Tier0NumericsResult(True, 1, 1, None, 4096)

    def run_tier1_perplexity(self) -> object:
        raise AssertionError("Tier 1 should not execute")

    def run_tier2_logit_metrics(self) -> object:
        raise AssertionError("Tier 2 should not execute")

    def run_tier3_latency(self) -> object:
        raise AssertionError("Tier 3 should not execute")


class Tier0RejectBackend(Tier0PassBackend):
    def run_tier0_load(self) -> Tier0ValidationResult:
        return Tier0ValidationResult(
            False,
            (),
            Tier0Stage.LOAD,
            "ValueError",
            "rejected fixture",
            "cpu",
            32,
        )


class ManualClock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class FakeWorker:
    def __init__(self) -> None:
        self.baseline_calls = 0
        self.mutation_calls: list[str] = []
        self.evaluation_calls: list[str] = []
        self.fail_mutation: set[str] = set()
        self.reject: set[str] = set()
        self.crash_evaluation_once: set[str] = set()

    def compute_baseline(self) -> BaselineArtifact:
        self.baseline_calls += 1
        return BaselineArtifact(((1.0, 0.0),), 1.0, 1, (("fixture", 1.0),))

    def mutate(
        self,
        candidate: MutationCandidate,
        baseline: BaselineArtifact,
        config: OOMAttemptConfig,
        cleanup: ExperimentGPUCleanup,
        heartbeat: object,
    ) -> MutationCheckpoint:
        del baseline, config
        self.mutation_calls.append(candidate.candidate_id)
        cleanup.own_cache([])
        heartbeat()
        if candidate.candidate_id in self.fail_mutation:
            raise ValueError("mutation fixture failed")
        return MutationCheckpoint(
            f"checkpoint:{candidate.candidate_id}",
            (("mutation_id", candidate.mutation_id),),
        )

    def evaluation_backend(
        self,
        candidate: MutationCandidate,
        checkpoint: MutationCheckpoint,
        baseline: BaselineArtifact,
        config: OOMAttemptConfig,
        cleanup: ExperimentGPUCleanup,
        heartbeat: object,
    ) -> Tier0PassBackend:
        del checkpoint, baseline, config
        self.evaluation_calls.append(candidate.candidate_id)
        cleanup.own_cache([])
        heartbeat()
        if candidate.candidate_id in self.crash_evaluation_once:
            self.crash_evaluation_once.remove(candidate.candidate_id)
            raise SimulatedCrash("worker process vanished")
        if candidate.candidate_id in self.reject:
            return Tier0RejectBackend()
        return Tier0PassBackend()


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


def _context() -> CampaignContext:
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
    run = derive_run_identity(identity.experiment_id, "campaign")
    return CampaignContext(
        identity.experiment_id,
        run.run_id,
        "attempt-campaign",
        model,
        dataset,
        _hardware(),
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
    )


def _candidates(context: CampaignContext) -> tuple[MutationCandidate, ...]:
    graph = ComponentGraph.build(
        (
            GraphNode(ComponentId.parse("model"), "model"),
            GraphNode(ComponentId.parse("model.layers.0.self_attn.q_proj"), "projection"),
            GraphNode(ComponentId.parse("model.layers.1.self_attn.q_proj"), "projection"),
        )
    )
    return enumerate_mutation_candidates(
        graph,
        context.run_id,
        CandidateEnumeratorConfig(
            seed=7,
            filters=CandidateFilter(
                scopes=(CandidateScope.COMPONENT,),
                include_kinds=("projection",),
            ),
        ),
    ).candidates


def _config(*, lease_duration_ns: int = 60_000_000_000) -> CampaignRunnerConfig:
    return CampaignRunnerConfig(
        "worker-1",
        lease_duration_ns,
        OOMAttemptConfig(2, 2),
        OOMAttemptConfig(2, 2),
        OOMRetryPolicy(max_retries=1),
        TieredEvaluationConfig(max_tier=EvaluationTier.TIER0),
    )


def test_campaign_reuses_one_baseline_and_completed_candidates_are_not_rerun(
    tmp_path: Path,
) -> None:
    context = _context()
    candidates = _candidates(context)
    worker = FakeWorker()
    cache = BaselineCache(tmp_path / "baseline")
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    try:
        first = CampaignRunner(store, cache, context, candidates, worker, _config()).run()
        assert first.progress.succeeded == 2
        assert first.progress.completed == 2
        assert worker.baseline_calls == 1
        assert len(worker.mutation_calls) == 2
        assert len(worker.evaluation_calls) == 2

        second = CampaignRunner(store, cache, context, candidates, worker, _config()).run()
        assert second.progress.succeeded == 2
        assert second.processed == ()
        assert second.skipped_completed == tuple(candidate.candidate_id for candidate in candidates)
        assert worker.baseline_calls == 1
        assert len(worker.mutation_calls) == 2
        assert len(worker.evaluation_calls) == 2
        for candidate in candidates:
            assert campaign_status(store, candidate.candidate_id).evaluation is not None
    finally:
        store.close()


def test_candidate_failure_is_isolated_and_campaign_progress_remains_queryable(
    tmp_path: Path,
) -> None:
    context = _context()
    candidates = _candidates(context)
    worker = FakeWorker()
    worker.fail_mutation.add(candidates[0].candidate_id)
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    try:
        report = CampaignRunner(
            store,
            BaselineCache(tmp_path / "baseline"),
            context,
            candidates,
            worker,
            _config(),
        ).run()
        assert report.progress.failed == 1
        assert report.progress.succeeded == 1
        assert report.progress.completed == 2
        assert len(report.failures) == 1
        assert report.failures[0].candidate_id == candidates[0].candidate_id
        assert report.failures[0].exception_type == "ValueError"
        progress = query_campaign_progress(store, context.run_id)
        assert progress == report.progress
        assert campaign_status(store, candidates[0].candidate_id).outcome is CampaignOutcome.FAILED
        assert campaign_status(store, candidates[1].candidate_id).outcome is CampaignOutcome.SUCCEEDED
    finally:
        store.close()


def test_rejected_tiered_evaluation_persists_rejected_candidate(tmp_path: Path) -> None:
    context = _context()
    candidates = _candidates(context)
    worker = FakeWorker()
    worker.reject.add(candidates[0].candidate_id)
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    try:
        report = CampaignRunner(
            store,
            BaselineCache(tmp_path / "baseline"),
            context,
            candidates,
            worker,
            _config(),
        ).run()
        assert report.progress.rejected == 1
        assert report.progress.succeeded == 1
        status = campaign_status(store, candidates[0].candidate_id)
        assert status.outcome is CampaignOutcome.REJECTED
        assert status.evaluation is not None
        assert status.evaluation["accepted"] is False
    finally:
        store.close()


def test_restart_after_mutation_checkpoint_resumes_evaluation_without_remutation(
    tmp_path: Path,
) -> None:
    context = _context()
    candidates = _candidates(context)[:1]
    worker = FakeWorker()
    worker.crash_evaluation_once.add(candidates[0].candidate_id)
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    cache = BaselineCache(tmp_path / "baseline")
    clock = ManualClock(0)
    config = _config(lease_duration_ns=10)
    try:
        with pytest.raises(SimulatedCrash):
            CampaignRunner(
                store,
                cache,
                context,
                candidates,
                worker,
                config,
                clock_ns=clock,
            ).run()
        machine = ExperimentStateMachine(store)
        assert machine.current(candidates[0].candidate_id) is CandidateState.EVALUATING
        assert campaign_status(store, candidates[0].candidate_id).checkpoint is not None
        assert len(worker.mutation_calls) == 1
        assert len(worker.evaluation_calls) == 1

        clock.value = 10
        resumed = CampaignRunner(
            store,
            cache,
            context,
            candidates,
            worker,
            config,
            clock_ns=clock,
        ).run()
        assert resumed.progress.succeeded == 1
        assert len(worker.mutation_calls) == 1
        assert len(worker.evaluation_calls) == 2
        assert worker.baseline_calls == 1
    finally:
        store.close()


def test_registration_is_idempotent_but_plan_drift_fails_closed(tmp_path: Path) -> None:
    context = _context()
    candidates = _candidates(context)
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    try:
        register_campaign(store, context, candidates)
        register_campaign(store, context, candidates)
        progress = query_campaign_progress(store, context.run_id)
        assert progress.total == 2
        assert progress.planned == 2

        with pytest.raises(CampaignError, match="conflicts"):
            register_campaign(store, context, candidates[:1])
    finally:
        store.close()
