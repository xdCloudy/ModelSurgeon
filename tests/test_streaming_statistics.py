"""Tests for mergeable bounded-memory statistics."""

from __future__ import annotations

import math
import sys

import pytest

from modelsurgeon.instrumentation import StatisticsConfig, StatisticsError, StreamingStatistics


def test_chunked_merged_and_unchunked_results_agree() -> None:
    values = [math.sin(index / 17) * 4 for index in range(10_000)]
    direct = StreamingStatistics()
    direct.update(values)
    merged = StreamingStatistics()
    for start in range(0, len(values), 137):
        chunk = StreamingStatistics()
        chunk.update(values[start : start + 137])
        merged.merge(chunk)

    left = direct.snapshot()
    right = merged.snapshot()
    assert right.count == left.count
    assert right.mean == pytest.approx(left.mean, abs=1e-14)
    assert right.variance == pytest.approx(left.variance, rel=1e-13)
    assert right.rms == pytest.approx(left.rms, rel=1e-13)
    assert right.percentiles == left.percentiles


def test_moments_sparsity_activation_and_extrema_match_exact_values() -> None:
    stats = StreamingStatistics(StatisticsConfig(-4, 4, 4096, zero_epsilon=0.01))
    stats.update((-2.0, 0.0, 0.005, 2.0, 4.0))
    result = stats.snapshot((0.0, 0.5, 1.0))

    assert result.count == 5
    assert result.mean == pytest.approx(0.801)
    assert result.variance == pytest.approx(4.158404)
    assert result.rms == pytest.approx(math.sqrt((4 + 0.000025 + 4 + 16) / 5))
    assert (result.minimum, result.maximum) == (-2.0, 4.0)
    assert result.zero_frequency == 0.4
    assert result.activation_frequency == 0.4
    assert result.percentiles[0] == (0.0, -2.0)
    assert result.percentiles[-1] == (1.0, 4.0)
    assert result.percentiles[1][1] == pytest.approx(0.005, abs=0.002)


def test_accumulator_memory_is_independent_of_token_count() -> None:
    stats = StreamingStatistics(StatisticsConfig(histogram_bins=64))
    before = sys.getsizeof(stats) + sys.getsizeof(stats._histogram)  # type: ignore[attr-defined]
    stats.update(float(index % 7) for index in range(200_000))
    after = sys.getsizeof(stats) + sys.getsizeof(stats._histogram)  # type: ignore[attr-defined]

    assert before == after
    assert len(stats.histogram) == 64
    assert sum(stats.histogram) == 200_000


def test_invalid_values_empty_summary_and_incompatible_merge_fail() -> None:
    stats = StreamingStatistics()
    with pytest.raises(StatisticsError, match="empty"):
        stats.snapshot()
    with pytest.raises(StatisticsError, match="finite"):
        stats.update((float("nan"),))
    with pytest.raises(StatisticsError, match="configurations"):
        stats.merge(StreamingStatistics(StatisticsConfig(histogram_bins=32)))
