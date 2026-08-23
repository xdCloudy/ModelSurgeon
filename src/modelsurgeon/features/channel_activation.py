"""Bounded channel and token-bucket activation aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation import (
    StatisticsConfig,
    StatisticsSnapshot,
    StreamingStatistics,
)


@dataclass(frozen=True, slots=True)
class ChannelActivationConfig:
    """Graph channels and the finite token buckets to collect for them."""

    channel_ids: tuple[ComponentId, ...]
    position_buckets: int
    token_classes: tuple[str, ...] = ()
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    def __post_init__(self) -> None:
        if not self.channel_ids or self.position_buckets <= 0:
            raise ValueError("graph channels and position buckets must be non-empty")
        if len(self.channel_ids) != len(set(self.channel_ids)):
            raise ValueError("graph channel identities must be unique")
        if len(self.token_classes) != len(set(self.token_classes)) or any(
            not item for item in self.token_classes
        ):
            raise ValueError("token classes must be non-empty and unique")

    @property
    def channel_count(self) -> int:
        return len(self.channel_ids)


@dataclass(frozen=True, slots=True)
class ChannelActivationFeature:
    """Statistics for exactly one canonical graph channel."""

    channel_id: ComponentId
    statistics: StatisticsSnapshot


@dataclass(frozen=True, slots=True)
class ChannelActivationSummary:
    """Per-channel features plus optional token-dependent aggregates."""

    channels: tuple[ChannelActivationFeature, ...]
    position_means: tuple[float | None, ...]
    token_class_means: tuple[tuple[str, float | None], ...]


class ChannelActivationCollector:
    """Keep only one accumulator per graph channel and configured token bucket.

    Positions beyond ``position_buckets`` are folded into the final overflow bucket.
    Masked tokens update no accumulator.
    """

    def __init__(self, config: ChannelActivationConfig) -> None:
        self.config = config
        self.channels = tuple(
            StreamingStatistics(config.statistics) for _ in config.channel_ids
        )
        self.positions = tuple(
            StreamingStatistics(config.statistics) for _ in range(config.position_buckets)
        )
        self.classes = {
            name: StreamingStatistics(config.statistics) for name in config.token_classes
        }

    @property
    def accumulator_count(self) -> int:
        """Return the fixed accumulator count, independent of tokens observed."""

        return len(self.channels) + len(self.positions) + len(self.classes)

    def update(
        self,
        tokens: tuple[tuple[float, ...], ...],
        mask: tuple[bool, ...],
        *,
        token_classes: tuple[str | None, ...] | None = None,
    ) -> None:
        if len(tokens) != len(mask):
            raise ValueError("token values and mask must align")
        if token_classes is not None and len(token_classes) != len(tokens):
            raise ValueError("token classes and values must align")
        for position, (token, include) in enumerate(zip(tokens, mask, strict=True)):
            if len(token) != self.config.channel_count:
                raise ValueError(
                    f"activation width {len(token)} does not match graph channel count "
                    f"{self.config.channel_count}"
                )
            if not include:
                continue
            for accumulator, value in zip(self.channels, token, strict=True):
                accumulator.update((value,))
            self.positions[min(position, self.config.position_buckets - 1)].update(token)
            if token_classes is not None:
                label = token_classes[position]
                if label is not None:
                    if label not in self.classes:
                        raise ValueError(f"unconfigured token class {label!r}")
                    self.classes[label].update(token)

    @staticmethod
    def _mean(accumulator: StreamingStatistics) -> float | None:
        return None if accumulator.count == 0 else accumulator.snapshot(()).mean

    def summary(self) -> ChannelActivationSummary:
        """Summarize in the exact channel order supplied by the component graph."""

        if any(channel.count == 0 for channel in self.channels):
            raise ValueError("every graph channel requires at least one observation")
        return ChannelActivationSummary(
            tuple(
                ChannelActivationFeature(channel_id, accumulator.snapshot(()))
                for channel_id, accumulator in zip(
                    self.config.channel_ids, self.channels, strict=True
                )
            ),
            tuple(self._mean(item) for item in self.positions),
            tuple((name, self._mean(self.classes[name])) for name in self.config.token_classes),
        )
