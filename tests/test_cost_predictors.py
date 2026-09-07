from __future__ import annotations

from modelsurgeon.datasets.hardware_cost import (
    HardwareCostDataset,
    HardwareCostExample,
    HardwareCostMetric,
    HardwareCostOutcome,
    HardwareCostProvenance,
    HardwareCostSplit,
)
from modelsurgeon.surgeon import (
    CostPredictionStatus,
    CostPredictorConfig,
    CostPredictorOutcome,
    fit_cost_predictors,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _example(index: int) -> HardwareCostExample:
    size = 1_000 + index * 100
    metrics = tuple(
        sorted(
            (
                HardwareCostMetric.from_samples(
                    name,
                    tuple(base * (index + 1) * factor for factor in range(1, 8)),
                )
                for name, base in (
                    ("load_time_seconds", 0.1),
                    ("prefill_tokens_per_second", 100.0),
                    ("decode_tokens_per_second", 80.0),
                    ("latency_seconds", 0.02),
                    ("peak_ram_bytes", 1_000.0),
                    ("peak_vram_bytes", 2_000.0),
                    ("disk_bytes", 3_000.0),
                )
            ),
            key=lambda item: item.name,
        )
    )
    provenance = HardwareCostProvenance(
        f"benchmark-{index}",
        "protocol-v1",
        "tool-v1",
        ("runner", "--seed", str(index)),
        {"threads": 1},
    )
    return HardwareCostExample(
        "tiny-family",
        f"model-revision-{index}",
        f"architecture-{index}",
        _digest(100 + index),
        _digest(200 + index),
        size,
        f"profile-{index}",
        f"context-{index}",
        "runtime",
        "runtime-v1",
        {"threads": 1, "offload": index},
        index,
        HardwareCostOutcome.MEASURED,
        metrics,
        provenance,
        f"lineage-{index}",
    )


def test_cost_predictors_retain_evidence_and_reject_unseen_context() -> None:
    assert CostPredictorConfig().target_names == tuple(sorted(CostPredictorConfig().target_names))
    examples = tuple(_example(index) for index in range(7))
    dataset = HardwareCostDataset(
        examples,
        HardwareCostSplit(
            tuple(f"lineage-{index}" for index in range(3)),
            ("lineage-3", "lineage-4"),
            ("lineage-5", "lineage-6"),
        ),
        "dataset-revision-v1",
    )

    study = fit_cost_predictors(
        dataset,
        config=CostPredictorConfig(
            target_names=("latency_seconds",),
            minimum_train_examples=3,
        ),
    )

    evaluation = study.evaluations[0]
    assert evaluation.outcome in {
        CostPredictorOutcome.MEASURED,
        CostPredictorOutcome.NEGATIVE_RESULT,
    }
    assert evaluation.predictor is not None
    assert evaluation.predictor.coverage is not None
    assert evaluation.predictor.model_size_bytes > 0
    assert (
        study.study_id
        == fit_cost_predictors(
            dataset,
            config=CostPredictorConfig(
                target_names=("latency_seconds",),
                minimum_train_examples=3,
            ),
        ).study_id
    )
    prediction = study.predict(examples[-1], "latency_seconds")
    assert prediction.status is CostPredictionStatus.OUT_OF_DISTRIBUTION
    assert prediction.out_of_distribution
