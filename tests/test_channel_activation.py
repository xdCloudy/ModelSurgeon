"""Tests for graph-aligned channel and token-dependent activation features."""

from __future__ import annotations

import pytest

from modelsurgeon.features import ChannelActivationCollector, ChannelActivationConfig
from modelsurgeon.graph import ComponentId


def _channels() -> tuple[ComponentId, ...]:
    return tuple(
        ComponentId.parse(f"model.layers.0.mlp.channels.{index}") for index in range(3)
    )


def _collector() -> ChannelActivationCollector:
    return ChannelActivationCollector(
        ChannelActivationConfig(_channels(), position_buckets=2, token_classes=("word", "code"))
    )


def test_channel_features_preserve_graph_identity_order_and_statistics() -> None:
    collector = _collector()
    collector.update(((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)), (True, True))

    summary = collector.summary()

    assert tuple(feature.channel_id for feature in summary.channels) == _channels()
    assert tuple(feature.statistics.mean for feature in summary.channels) == (2.0, 3.0, 4.0)
    assert summary.channels[0].statistics.minimum == 1.0
    assert summary.channels[0].statistics.maximum == 3.0


def test_mask_position_overflow_and_token_classes_are_aggregated() -> None:
    collector = _collector()
    collector.update(
        (
            (1.0, 2.0, 3.0),
            (1000.0, 1000.0, 1000.0),
            (7.0, 8.0, 9.0),
        ),
        (True, False, True),
        token_classes=("word", "code", "code"),
    )

    summary = collector.summary()

    assert summary.position_means == pytest.approx((2.0, 8.0))
    assert dict(summary.token_class_means)["word"] == pytest.approx(2.0)
    assert dict(summary.token_class_means)["code"] == pytest.approx(8.0)


def test_accumulator_count_is_bounded_by_channels_and_configured_buckets() -> None:
    collector = _collector()
    expected = len(_channels()) + 2 + 2

    for _ in range(1000):
        collector.update(((1.0, 2.0, 3.0),), (True,))

    assert collector.accumulator_count == expected


def test_empty_buckets_are_explicit_and_all_channels_must_be_observed() -> None:
    collector = _collector()
    collector.update(((1.0, 2.0, 3.0),), (True,))

    summary = collector.summary()
    assert summary.position_means == (2.0, None)
    assert summary.token_class_means == (("word", None), ("code", None))

    empty = _collector()
    empty.update(((1.0, 2.0, 3.0),), (False,))
    with pytest.raises(ValueError, match="every graph channel"):
        empty.summary()


def test_invalid_shapes_classes_and_configuration_fail() -> None:
    collector = _collector()
    with pytest.raises(ValueError, match="graph channel count"):
        collector.update(((1.0, 2.0),), (True,))
    with pytest.raises(ValueError, match="unconfigured token class"):
        collector.update(((1.0, 2.0, 3.0),), (True,), token_classes=("unknown",))
    with pytest.raises(ValueError, match="identities must be unique"):
        ChannelActivationConfig((_channels()[0], _channels()[0]), 1)
