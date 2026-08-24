"""End-to-end interrupt/resume campaign-to-dataset integration coverage."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from modelsurgeon.datasets import (
    DeltaTargetPolicy,
    ExperimentFeatureJoin,
    GroupedSplitConfig,
    GroupedSplitMode,
    MutationExampleBuildPolicy,
    SplitRatios,
    audit_dataset_leakage,
    build_mutation_examples,
    create_grouped_split,
    validate_mutation_dataset,
)
from modelsurgeon.evaluation.baseline_cache import BaselineArtifact, BaselineCache
from modelsurgeon.evaluation.numerics import Tier0NumericsResult
from modelsurgeon.evaluation.tier0 import Tier0Stage, Tier0ValidationResult
from modelsurgeon.evaluation.tiered import EvaluationTier, TieredEvaluationConfig
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
    canonical_identity_json,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.campaign import (
    CampaignContext,
    CampaignRunner,
    CampaignRunnerConfig,
    CampaignWorker,
    MutationCheckpoint,
    campaign_status,
)
from modelsurgeon.experiments.candidates import (
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    MutationCandidate,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.gpu_cleanup import ExperimentGPUCleanup
from modelsurgeon.experiments.oom_recovery import OOMAttemptConfig, OOMRetryPolicy
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionKey
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode
from modelsurgeon.surgery.contracts import MutationDelta, MutationPlan
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
    MutationRunRecord,
)


class SimulatedCampaignInterrupt(BaseException):
    pass


class ManualClock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


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


class IntegrationWorker(CampaignWorker):
    def __init__(self, *, interrupt_candidate: str | None = None) -> None:
        self.interrupt_candidate = interrupt_candidate
        self.interrupted = False
        self.baseline_calls = 0
        self.mutation_calls: list[str] = []
        self.evaluation_calls: list[str] = []

    def compute_baseline(self) -> BaselineArtifact:
        self.baseline_calls += 1
        return BaselineArtifact(((1.0, 0.0),), 1.0, 1, (("fixture", 1.0),))

    def mutate(
        self,
        candidate: MutationCandidate,
        baseline: BaselineArtifact,
        config: OOMAttemptConfig,
        cleanup: ExperimentGPUCleanup,
        heartbeat: Callable[[], None],
    ) -> MutationCheckpoint:
        del baseline, config
        self.mutation_calls.append(candidate.candidate_id)
        cleanup.own_cache([])
        heartbeat()
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
        heartbeat: Callable[[], None],
    ) -> Tier0PassBackend:
        del checkpoint, baseline, config
        self.evaluation_calls.append(candidate.candidate_id)
        cleanup.own_cache([])
        heartbeat()
        if candidate.candidate_id == self.interrupt_candidate and not self.interrupted:
            self.interrupted = True
            raise SimulatedCampaignInterrupt("fixture process interruption")
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
    run = derive_run_identity(identity.experiment_id, "campaign-dataset-integration")
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
            GraphNode(ComponentId.parse("model.layers.2.self_attn.q_proj"), "projection"),
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


def _runner_config() -> CampaignRunnerConfig:
    return CampaignRunnerConfig(
        "worker-1",
        10,
        OOMAttemptConfig(2, 2),
        OOMAttemptConfig(2, 2),
        OOMRetryPolicy(max_retries=1),
        TieredEvaluationConfig(max_tier=EvaluationTier.TIER0),
    )


def _feature_partition(
    context: CampaignContext,
    candidate: MutationCandidate,
) -> FeaturePartition:
    feature = FeatureRecord(
        candidate.component_id,
        "pre_mutation_fixture",
        FeatureKind.SCALAR,
        float(candidate.layer_index or 0),
        "float32",
        "campaign-integration",
        "1",
        PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "float32",
            "float32",
        ),
    )
    digest = hashlib.sha256(
        canonical_identity_json([feature.to_record()]).encode("utf-8")
    ).hexdigest()
    return FeaturePartition(
        FeaturePartitionKey(
            context.model.revision,
            context.dataset.manifest_id,
            candidate.component_id,
            feature.extractor,
            feature.extractor_version,
        ),
        (feature,),
        digest,
    )


def _experiment_record(
    store: ExperimentMetadataStore,
    context: CampaignContext,
    candidate: MutationCandidate,
) -> ExperimentRecord:
    status = campaign_status(store, candidate.candidate_id)
    if status.evaluation is None or status.outcome.value != "succeeded":
        raise AssertionError("completed campaign candidate lacks persisted evaluation")
    delta = MutationDelta()
    mutation = MutationRunRecord(
        MutationPlan(candidate.request, candidate.affected_components, (), delta),
        MutationProvenance(context.model.revision, context.versions.tool_revision),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return ExperimentRecord(
        context.run_id,
        context.experiment_id,
        context.attempt_id,
        context.model,
        context.dataset,
        candidate.affected_components,
        mutation,
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        context.hardware,
        context.versions,
        context.seeds,
        quantization_control=context.quantization_control,
    )


def _dataset_pipeline(
    store: ExperimentMetadataStore,
    context: CampaignContext,
    candidates: tuple[MutationCandidate, ...],
) -> str:
    joins = tuple(
        ExperimentFeatureJoin(
            _experiment_record(store, context, candidate),
            (_feature_partition(context, candidate),),
        )
        for candidate in candidates
    )
    built = build_mutation_examples(
        joins,
        MutationExampleBuildPolicy(
            delta_target_policy=DeltaTargetPolicy.PRESERVE_MISSING
        ),
    )
    assert not built.exclusions
    assert len(built.examples) == len(candidates)
    assert len({item.example_id for item in built.examples}) == len(candidates)

    validation = validate_mutation_dataset(built.examples)
    assert validation.valid
    split = create_grouped_split(
        built.examples,
        GroupedSplitConfig(
            GroupedSplitMode.COMPONENT,
            seed=19,
            ratios=SplitRatios(1 / 3, 1 / 3, 1 / 3),
        ),
    )
    leakage = audit_dataset_leakage(built.examples, split)
    leakage.require_clean()
    return canonical_identity_json(
        {
            "build": built.to_record(),
            "examples": [item.to_record() for item in built.examples],
            "validation": validation.to_record(),
            "split": split.to_record(),
            "leakage": leakage.to_record(),
        }
    )


def _candidate_row_count(store: ExperimentMetadataStore, run_id: str) -> int:
    with store.reader() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM experiment_candidates WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise AssertionError("candidate count query returned no row")
    return int(row[0])


def test_interrupted_resume_produces_same_validated_dataset_as_uninterrupted(
    tmp_path: Path,
) -> None:
    context = _context()
    candidates = _candidates(context)
    assert len(candidates) == 3
    config = _runner_config()

    uninterrupted_store = ExperimentMetadataStore(tmp_path / "uninterrupted.sqlite3")
    resumed_store = ExperimentMetadataStore(tmp_path / "resumed.sqlite3")
    try:
        uninterrupted_worker = IntegrationWorker()
        uninterrupted = CampaignRunner(
            uninterrupted_store,
            BaselineCache(tmp_path / "baseline-uninterrupted"),
            context,
            candidates,
            uninterrupted_worker,
            config,
            clock_ns=ManualClock(0),
        ).run()
        assert uninterrupted.progress.succeeded == 3
        uninterrupted_dataset = _dataset_pipeline(
            uninterrupted_store,
            context,
            candidates,
        )

        interrupted_worker = IntegrationWorker(
            interrupt_candidate=candidates[0].candidate_id
        )
        clock = ManualClock(0)
        resumed_cache = BaselineCache(tmp_path / "baseline-resumed")
        with pytest.raises(SimulatedCampaignInterrupt):
            CampaignRunner(
                resumed_store,
                resumed_cache,
                context,
                candidates,
                interrupted_worker,
                config,
                clock_ns=clock,
            ).run()

        assert len(interrupted_worker.mutation_calls) == 1
        assert len(interrupted_worker.evaluation_calls) == 1
        assert campaign_status(
            resumed_store,
            candidates[0].candidate_id,
        ).checkpoint is not None

        clock.value = config.lease_duration_ns
        resumed = CampaignRunner(
            resumed_store,
            resumed_cache,
            context,
            candidates,
            interrupted_worker,
            config,
            clock_ns=clock,
        ).run()
        assert resumed.progress.succeeded == 3
        assert resumed.progress.completed == 3
        assert len(interrupted_worker.mutation_calls) == 3
        assert len(set(interrupted_worker.mutation_calls)) == 3
        assert len(interrupted_worker.evaluation_calls) == 4
        assert _candidate_row_count(resumed_store, context.run_id) == 3

        resumed_dataset = _dataset_pipeline(resumed_store, context, candidates)
        assert resumed_dataset == uninterrupted_dataset
    finally:
        uninterrupted_store.close()
        resumed_store.close()
