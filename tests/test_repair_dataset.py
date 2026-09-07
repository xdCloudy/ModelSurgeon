from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets import (
    RepairBaseline,
    RepairDataset,
    RepairDatasetConfig,
    RepairDatasetError,
    RepairDatasetExample,
    RepairDatasetProvenance,
    RepairDatasetSplit,
    build_repair_dataset,
)
from modelsurgeon.surgery import (
    ArtifactOutputMode,
    MetricDirection,
    RepairArtifact,
    RepairArtifactLineage,
    RepairBudget,
    RepairCost,
    RepairMethod,
    RepairMetricEvidence,
    RepairModelIdentity,
    RepairOutcomeStatus,
    RepairRequest,
    RepairResult,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _budget() -> RepairBudget:
    return RepairBudget(10, 100, 20.0, 20.0, 100.0, 1_000)


def _identity(checkpoint: str) -> RepairModelIdentity:
    return RepairModelIdentity(checkpoint, "model-rev", _digest(4), "data-v1")


def _request(method: RepairMethod = RepairMethod.LORA) -> RepairRequest:
    return RepairRequest(
        request_id=f"request-{method.value}",
        parent_candidate_id="candidate-1",
        source=_identity("source"),
        candidate=_identity("candidate"),
        teacher=None,
        method=method,
        output_mode=(
            ArtifactOutputMode.SEPARATE_LORA
            if method is not RepairMethod.NO_REPAIR
            else ArtifactOutputMode.NONE
        ),
        trainable_scope=("layer.0",),
        budget=_budget(),
        compatibility={"architecture": "supported", "codec": "dense"},
        best_state_metric="heldout_loss",
        best_state_direction=MetricDirection.LOWER,
        quantization_order="repair_then_quantize",
    )


def _evidence(repaired: float) -> RepairMetricEvidence:
    return RepairMetricEvidence(
        "heldout_loss",
        MetricDirection.LOWER,
        1.0,
        repaired,
        0.1,
        heldout_group_ids=("heldout-eval",),
    )


def _baseline() -> RepairBaseline:
    return RepairBaseline(
        "candidate-1",
        _digest(1),
        _digest(2),
        _budget(),
        RepairCost(0, 0, 0.0, 0.0, 0.0, 0),
        (_evidence(1.0),),
    )


def _repair(*, failed: bool = False) -> RepairResult:
    request = _request()
    artifact = RepairArtifact(
        "artifact-child",
        _digest(3),
        32,
        ArtifactOutputMode.SEPARATE_LORA,
        _digest(2),
        not failed,
    )
    cost = RepairCost(0 if failed else 4, 0 if failed else 80, 1.0, 1.0, 2.0, 0 if failed else 32)
    return RepairResult(
        request,
        RepairOutcomeStatus.FAILED if failed else RepairOutcomeStatus.ACCEPTED,
        cost,
        (artifact,) if not failed else (),
        RepairArtifactLineage(
            _digest(2), _digest(1), _digest(3), None, "data-v1", _digest(4)
        ),
        () if failed else (_evidence(0.75),),
        failed,
        not failed,
        "out of memory" if failed else None,
    )


def _example(
    *,
    family: str = "gemma",
    mutation_kind: str = "mlp_mask",
    budget_name: str = "zero",
    seed: int = 0,
    group: str = "lineage-1",
    failed: bool = False,
) -> RepairDatasetExample:
    return RepairDatasetExample(
        family,
        "model-rev",
        "state-damaged",
        mutation_kind,
        0.5,
        _digest(1),
        _digest(2),
        "data-v1",
        _digest(4),
        "cpu-test",
        budget_name,
        seed,
        group,
        _baseline(),
        _repair(failed=failed),
        RepairDatasetProvenance(
            f"record-{family}-{mutation_kind}-{budget_name}-{seed}",
            "source-rev",
            "repair-dataset-v1",
            "tool-rev",
            ("runner", "--seed", str(seed)),
            {"budget": budget_name},
        ),
    )


def test_dataset_pairs_exact_control_and_retains_terminal_failure() -> None:
    examples = (
        _example(),
        _example(
            family="llama",
            mutation_kind="attention_head_mask",
            budget_name="short",
            seed=1,
            group="lineage-2",
            failed=True,
        ),
        _example(
            family="llama",
            mutation_kind="attention_head_mask",
            budget_name="medium",
            seed=2,
            group="lineage-3",
        ),
    )
    dataset = build_repair_dataset(
        examples,
        RepairDatasetSplit(("lineage-1",), ("lineage-2",), ("lineage-3",)),
        RepairDatasetConfig("data-v1"),
    )

    assert isinstance(dataset, RepairDataset)
    assert dataset.to_record()["coverage"] == {
        "examples": 3,
        "model_families": 2,
        "mutation_kinds": 2,
        "seeds": 3,
        "budgets": ["medium", "short", "zero"],
        "outcomes": {"accepted": 2, "rejected": 0, "failed": 1, "unsupported": 0, "unknown": 0},
    }
    assert dataset.examples[1].outcome is RepairOutcomeStatus.FAILED
    assert dataset.examples[1].repair.failure_reason == "out of memory"
    assert dataset.examples[0].no_repair.outcome is RepairOutcomeStatus.REJECTED
    assert dataset.examples[0].no_repair.outcome_id == dataset.examples[0].no_repair.outcome_id


def test_pair_rejects_control_with_different_damaged_candidate() -> None:
    baseline = replace(_baseline(), candidate_artifact_digest=_digest(9))
    with pytest.raises(RepairDatasetError, match="no-repair candidate artifact"):
        RepairDatasetExample(
            "gemma",
            "model-rev",
            "state-damaged",
            "mlp_mask",
            0.5,
            _digest(1),
            _digest(2),
            "data-v1",
            _digest(4),
            "cpu-test",
            "zero",
            0,
            "lineage-1",
            baseline,
            _repair(),
            RepairDatasetProvenance(
                "record-bad", "source-rev", "repair-dataset-v1", "tool-rev", ("runner",), {}
            ),
        )


def test_pair_rejects_repair_evidence_on_different_heldout_groups() -> None:
    result = _repair()
    changed = replace(
        result,
        heldout_evidence=(
            replace(_evidence(0.75), heldout_group_ids=("different-eval",)),
        ),
    )
    with pytest.raises(RepairDatasetError, match="same held-out groups"):
        replace(_example(), repair=changed)


def test_split_rejects_lineage_leakage_and_config_rejects_missing_budget() -> None:
    with pytest.raises(RepairDatasetError, match="leak"):
        RepairDatasetSplit(("lineage-1",), ("lineage-1",), ("lineage-2",))

    with pytest.raises(RepairDatasetError, match="missing required budgets"):
        build_repair_dataset(
            (_example(), _example(family="llama", mutation_kind="head", seed=1, group="lineage-2")),
            RepairDatasetSplit(("lineage-1",), ("lineage-2",), ("lineage-3",)),
            RepairDatasetConfig("data-v1"),
        )
