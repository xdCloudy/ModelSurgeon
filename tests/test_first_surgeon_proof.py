"""Integration tests for the generic First Surgeon proof campaign orchestrator."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import pytest

from modelsurgeon.cli.experiment import ResolvedExperiment, SingleMutationExperimentResult
from modelsurgeon.cli.proof import (
    FirstSurgeonProofConfig,
    FirstSurgeonProofError,
    run_first_surgeon_proof,
    write_first_surgeon_proof,
)
from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.evaluation.tiered import (
    EscalationAction,
    EvaluationTier,
    TierDecision,
    TieredEvaluationReport,
)
from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    HardwareInventory,
    MemoryInventory,
    MetricObservation,
    MetricState,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
    derive_run_identity,
)
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionKey
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationPlan,
    MutationRequest,
    MutationTransaction,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction


class SnapshotTarget:
    def __init__(self) -> None:
        self.restores = 0

    def snapshot(self) -> object:
        return b"original"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"original"
        self.restores += 1


def _evaluation() -> TieredEvaluationReport:
    return TieredEvaluationReport(
        (
            TierDecision(
                EvaluationTier.TIER0,
                True,
                True,
                EscalationAction.COMPLETE,
                None,
                (),
            ),
            TierDecision(
                EvaluationTier.TIER1,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
            TierDecision(
                EvaluationTier.TIER2,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
            TierDecision(
                EvaluationTier.TIER3,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
        ),
        True,
        EvaluationTier.TIER0,
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


class FakeProofRuntime:
    def __init__(self) -> None:
        self.owner = object()
        self.target = SnapshotTarget()
        self.last_transaction: InMemoryMutationTransaction | None = None
        self._graph = ComponentGraph.build(
            tuple(
                GraphNode(
                    ComponentId.parse(f"model.layers.{index}.self_attn.q_proj"),
                    "projection",
                )
                for index in range(6)
            )
        )
        self._run_id = derive_run_identity("exp_" + "a" * 64).run_id
        self.model = ModelTarget(
            "tiny/model",
            "model-rev-1",
            "llama",
            "safetensors",
            128,
        )
        self.dataset = DatasetTarget(
            "tiny-dataset",
            "dataset-rev-1",
            "validation",
            "manifest-1",
            "tiny-tokenizer",
            "tokenizer-rev-1",
        )

    @property
    def component_graph(self) -> ComponentGraph:
        return self._graph

    @property
    def run_id(self) -> str:
        return self._run_id

    def resolve(self, request: MutationRequest) -> ResolvedExperiment:
        return ResolvedExperiment(
            MutationPlan(request, request.targets, (), MutationDelta()),
            MutationProvenance("model-rev-1", "tool-rev", "/private/model"),
        )

    def transaction(
        self,
        plan: MutationPlan,
    ) -> AbstractContextManager[MutationTransaction]:
        del plan
        transaction = InMemoryMutationTransaction(
            self.owner,
            {"target": self.target},
            ("target",),
        )
        self.last_transaction = transaction
        return transaction

    def mutation_scope(
        self,
        plan: MutationPlan,
        transaction: MutationTransaction,
    ) -> AbstractContextManager[object]:
        del plan, transaction
        return nullcontext(object())

    def evaluate(self, plan: MutationPlan) -> TieredEvaluationReport:
        del plan
        return _evaluation()

    def rolled_back_outcome(
        self,
        plan: MutationPlan,
        evaluation: TieredEvaluationReport,
    ) -> MutationOutcome:
        del plan, evaluation
        return MutationOutcome(
            MutationOutcomeStatus.ROLLED_BACK,
            MutationDelta(),
            (),
            "proof mutation reverted after evaluation",
        )

    def pre_mutation_feature_partitions(
        self,
        candidate: MutationCandidate,
    ) -> tuple[FeaturePartition, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "float32",
            "float64",
        )
        component = candidate.component_id
        weight = FeatureRecord(
            component,
            "weight_l1_norm",
            FeatureKind.SCALAR,
            1.0 + float(candidate.layer_index or 0),
            "float64",
            "weight_statistics",
            "1",
            precision,
        )
        activation = FeatureRecord(
            component,
            "activation_rms",
            FeatureKind.SCALAR,
            0.5 + float(candidate.layer_index or 0),
            "float64",
            "activation_summary",
            "1",
            precision,
        )
        return (
            FeaturePartition(
                FeaturePartitionKey(
                    "model-rev-1",
                    "manifest-1",
                    component,
                    "weight_statistics",
                    "1",
                ),
                (weight,),
                "a" * 64,
            ),
            FeaturePartition(
                FeaturePartitionKey(
                    "model-rev-1",
                    "manifest-1",
                    component,
                    "activation_summary",
                    "1",
                ),
                (activation,),
                "b" * 64,
            ),
        )

    def experiment_record(
        self,
        candidate: MutationCandidate,
        result: SingleMutationExperimentResult,
    ) -> ExperimentRecord:
        layer = float(candidate.layer_index or 0)
        return ExperimentRecord(
            self.run_id,
            f"experiment-{candidate.candidate_id}",
            f"attempt-{candidate.candidate_id}",
            self.model,
            self.dataset,
            result.run_record.plan.affected_components,
            result.run_record,
            (MetricObservation("perplexity", MetricState.MEASURED, 10.0),),
            (MetricObservation("perplexity", MetricState.MEASURED, 10.1 + layer / 100.0),),
            (MetricObservation("perplexity_delta", MetricState.MEASURED, 0.1 + layer / 100.0),),
            ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
            _hardware(),
            VersionContext("tool-rev", "config-digest", "eval-v1", 1, 1),
            SeedContext(1, 2, 3),
        )


def test_proof_campaign_builds_component_heldout_leakage_clean_dataset(tmp_path: Path) -> None:
    runtime = FakeProofRuntime()
    result = run_first_surgeon_proof(
        runtime,
        FirstSurgeonProofConfig(
            seed=7,
            split_seed=11,
            max_candidates=6,
            scopes=(CandidateScope.COMPONENT,),
            ratios=SplitRatios(0.5, 0.25, 0.25),
        ),
    )

    assert len(result.examples) == 6
    assert result.build.exclusions == ()
    assert result.leakage.clean
    assert result.split.mode.value == "component"
    assert all(result.split.example_counts[partition] > 0 for partition in SplitPartition)
    assert runtime.target.restores == 6

    paths = write_first_surgeon_proof(tmp_path / "proof", result)
    assert set(paths) == {"examples", "split", "leakage", "campaign"}
    lines = paths["examples"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert all(json.loads(line)["mutation"]["plan"]["request"]["kind"] == "mask" for line in lines)
    assert json.loads(paths["leakage"].read_text(encoding="utf-8"))["clean"] is True

    with pytest.raises(FirstSurgeonProofError, match="already exists"):
        write_first_surgeon_proof(tmp_path / "proof", result)


def test_proof_campaign_rejects_too_few_independent_groups() -> None:
    runtime = FakeProofRuntime()
    with pytest.raises(FirstSurgeonProofError, match="empty partitions"):
        run_first_surgeon_proof(
            runtime,
            FirstSurgeonProofConfig(
                seed=0,
                split_seed=0,
                max_candidates=2,
                scopes=(CandidateScope.COMPONENT,),
            ),
        )
