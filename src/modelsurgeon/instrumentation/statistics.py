"""Mergeable bounded-memory streaming statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


class StatisticsError(ValueError):
    """Raised for invalid values, configurations, or merges."""


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    histogram_min: float = -16.0
    histogram_max: float = 16.0
    histogram_bins: int = 4096
    zero_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.histogram_min)
            or not math.isfinite(self.histogram_max)
            or self.histogram_min >= self.histogram_max
            or self.histogram_bins < 2
            or not math.isfinite(self.zero_epsilon)
            or self.zero_epsilon < 0
        ):
            raise StatisticsError("invalid bounded statistics configuration")


@dataclass(frozen=True, slots=True)
class StatisticsSnapshot:
    count: int
    mean: float
    variance: float
    rms: float
    minimum: float
    maximum: float
    zero_frequency: float
    activation_frequency: float
    percentiles: tuple[tuple[float, float], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "rms": self.rms,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "zero_frequency": self.zero_frequency,
            "activation_frequency": self.activation_frequency,
            "percentiles": {str(q): value for q, value in self.percentiles},
        }


class StreamingStatistics:
    """Fixed-memory moments and histogram supporting associative merges."""

    def __init__(self, config: StatisticsConfig | None = None) -> None:
        self.config = config or StatisticsConfig()
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.sum_squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.zero_count = 0
        self.positive_count = 0
        self._histogram = [0] * self.config.histogram_bins

    @property
    def histogram(self) -> tuple[int, ...]:
        return tuple(self._histogram)

    def _bin(self, value: float) -> int:
        if value <= self.config.histogram_min:
            return 0
        if value >= self.config.histogram_max:
            return self.config.histogram_bins - 1
        fraction = (value - self.config.histogram_min) / (
            self.config.histogram_max - self.config.histogram_min
        )
        return min(self.config.histogram_bins - 1, int(fraction * self.config.histogram_bins))

    def update(self, values: Iterable[float]) -> None:
        for raw in values:
            value = float(raw)
            if not math.isfinite(value):
                raise StatisticsError("statistics values must be finite")
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)
            self.sum_squares += value * value
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            self.zero_count += abs(value) <= self.config.zero_epsilon
            self.positive_count += value > self.config.zero_epsilon
            self._histogram[self._bin(value)] += 1

    def merge(self, other: StreamingStatistics) -> None:
        if self.config != other.config:
            raise StatisticsError("statistics configurations must match for merge")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.sum_squares = other.sum_squares
            self.minimum = other.minimum
            self.maximum = other.maximum
            self.zero_count = other.zero_count
            self.positive_count = other.positive_count
            self._histogram[:] = other._histogram
            return
        combined = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / combined
        self.mean += delta * other.count / combined
        self.count = combined
        self.sum_squares += other.sum_squares
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.zero_count += other.zero_count
        self.positive_count += other.positive_count
        for index, value in enumerate(other._histogram):
            self._histogram[index] += value

    def percentile(self, quantile: float) -> float:
        if self.count == 0:
            raise StatisticsError("cannot summarize empty statistics")
        if not 0 <= quantile <= 1:
            raise StatisticsError("percentile quantile must be between zero and one")
        if quantile == 0:
            return self.minimum
        if quantile == 1:
            return self.maximum
        target = math.ceil(quantile * self.count)
        cumulative = 0
        width = (self.config.histogram_max - self.config.histogram_min) / self.config.histogram_bins
        for index, amount in enumerate(self._histogram):
            cumulative += amount
            if cumulative >= target:
                return self.config.histogram_min + (index + 0.5) * width
        return self.maximum

    def snapshot(self, quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> StatisticsSnapshot:
        if self.count == 0:
            raise StatisticsError("cannot summarize empty statistics")
        return StatisticsSnapshot(
            self.count,
            self.mean,
            self.m2 / self.count,
            math.sqrt(self.sum_squares / self.count),
            self.minimum,
            self.maximum,
            self.zero_count / self.count,
            self.positive_count / self.count,
            tuple((quantile, self.percentile(quantile)) for quantile in quantiles),
        )
