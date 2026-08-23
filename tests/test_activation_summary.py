"""Tests for masked per-component activation summaries."""

from __future__ import annotations

import pytest

from modelsurgeon.features import (
    ActivationBatch,
    ActivationSummaryCollector,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId


def _context(ids: tuple[str, ...]) -> FeatureSampleContext:
    return FeatureSampleContext(
        "org/data", "rev", "validation", ids, "prep-v1", "org/tok", "tok-rev"
    )


def _collector() -> ActivationSummaryCollector:
    return ActivationSummaryCollector(
        ComponentId.parse("model.layers.0.self_attn"),
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
    )


def test_token_mask_excludes_values_and_records_samples_and_axes() -> None:
    collector = _collector()
    collector.update(
        ActivationBatch(
            ("a", "b"),
            (
                ((1.0, 2.0), (1000.0, 1000.0), (3.0, 4.0)),
                ((-1000.0, -1000.0), (5.0, 6.0), (7.0, 8.0)),
            ),
            ((True, False, True), (False, True, True)),
        )
    )

    records = collector.records(_context(("a", "b")))
    by_name = {record.name: record for record in records}
    assert by_name["activation_mean"].value == pytest.approx(4.5)
    assert by_name["activation_minimum"].value == 1.0
    assert by_name["activation_maximum"].value == 8.0
    assert len(records) == 8
    attributes = dict(by_name["activation_mean"].metadata)
    assert attributes == {
        "aggregation_axes": "batch,token,feature",
        "feature_width": 2,
        "observation_count": 8,
    }
    assert by_name["activation_mean"].sample_context.sample_ids == ("a", "b")  # type: ignore[union-attr]


def test_multiple_batches_preserve_ordered_sample_identity() -> None:
    collector = _collector()
    collector.update(ActivationBatch(("a",), (((1.0,),),), ((True,),)))
    collector.update(ActivationBatch(("b",), (((2.0,),),), ((True,),)))

    records = collector.records(_context(("a", "b")))

    assert records[0].value == pytest.approx(1.5)
    assert records[0].sample_context.sample_ids == ("a", "b")  # type: ignore[union-attr]


def test_malformed_masks_duplicate_samples_and_context_mismatch_fail() -> None:
    with pytest.raises(ValueError, match="mask length"):
        ActivationBatch(("a",), (((1.0,),),), ((True, False),))

    collector = _collector()
    collector.update(ActivationBatch(("a",), (((1.0,),),), ((True,),)))
    with pytest.raises(ValueError, match="twice"):
        collector.update(ActivationBatch(("a",), (((2.0,),),), ((True,),)))
    with pytest.raises(ValueError, match="context"):
        collector.records(_context(("wrong",)))


def test_all_masked_batch_fails_empty_summary_explicitly() -> None:
    collector = _collector()
    collector.update(ActivationBatch(("a",), (((99.0,),),), ((False,),)))

    with pytest.raises(ValueError, match="empty"):
        collector.records(_context(("a",)))
