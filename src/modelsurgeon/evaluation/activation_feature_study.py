"""Paired static versus static-plus-activation ablation for v0.8 Q2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest
from modelsurgeon.surgeon.metrics import (
    grouped_bootstrap_interval,
    mae,
    precision_recall_at_n,
    rmse,
    roc_auc,
)

from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyConfig,
    StaticFeatureStudyError,
    StaticFeatureStudyResult,
    run_feature_profile_study,
)


@dataclass(frozen=True, slots=True)
class PairedGainEstimate:
    name: str
    value: float
    confidence_low: float
    confidence_high: float
    bootstrap_repetitions: int

    def __post_init__(self) -> None:
        if not self.name or not all(
            math.isfinite(value)
            for value in (self.value, self.confidence_low, self.confidence_high)
        ):
            raise StaticFeatureStudyError("paired gains require finite named estimates")
        if self.confidence_low > self.confidence_high or self.bootstrap_repetitions <= 0:
            raise StaticFeatureStudyError("paired gain confidence interval is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
        }


@dataclass(frozen=True, slots=True)
class ActivationFeatureAblationResult:
    static: StaticFeatureStudyResult
    static_activation: StaticFeatureStudyResult
    gains: tuple[PairedGainEstimate, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "model": {
                "identifier": self.static.model_identifier,
                "revision": self.static.model_revision,
                "family": self.static.family,
            },
            "static": self.static.to_record(),
            "static_activation": self.static_activation.to_record(),
            "paired_gains": [gain.to_record() for gain in self.gains],
        }


def _gain(
    name: str,
    group_ids: Sequence[str],
    point: float,
    metric: Callable[[Sequence[int]], float | None],
    config: StaticFeatureStudyConfig,
    seed_offset: int,
) -> PairedGainEstimate:
    interval = grouped_bootstrap_interval(
        group_ids,
        metric,
        repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
        seed=config.seed + seed_offset,
    )
    if interval is None:
        raise StaticFeatureStudyError(f"paired bootstrap could not define {name}")
    return PairedGainEstimate(name, point, interval[0], interval[1], interval[2])


def _indexes(values: Sequence[float], indexes: Sequence[int]) -> tuple[float, ...]:
    return tuple(values[index] for index in indexes)


def paired_feature_gains(
    static: StaticFeatureStudyResult,
    activation: StaticFeatureStudyResult,
    config: StaticFeatureStudyConfig,
) -> tuple[PairedGainEstimate, ...]:
    if (
        static.test_labels != activation.test_labels
        or static.test_targets != activation.test_targets
        or static.test_group_ids != activation.test_group_ids
    ):
        raise StaticFeatureStudyError("paired feature profiles must share held-out rows")
    labels = static.test_labels
    targets = static.test_targets
    groups = static.test_group_ids
    static_class = static.test_classifier_predictions
    activation_class = activation.test_classifier_predictions
    static_reg = static.test_regressor_predictions
    activation_reg = activation.test_regressor_predictions

    static_auc = roc_auc(labels, static_class)
    activation_auc = roc_auc(labels, activation_class)
    if static_auc is None or activation_auc is None:
        raise StaticFeatureStudyError("paired AUC requires both held-out classes")

    def auc_gain(indexes: Sequence[int]) -> float | None:
        subset_labels = tuple(labels[index] for index in indexes)
        left = roc_auc(subset_labels, _indexes(static_class, indexes))
        right = roc_auc(subset_labels, _indexes(activation_class, indexes))
        return None if left is None or right is None else right - left

    def precision(scores: Sequence[float], indexes: Sequence[int]) -> float | None:
        value, _ = precision_recall_at_n(
            tuple(labels[index] for index in indexes),
            _indexes(scores, indexes),
            config.top_n,
        )
        return value

    all_indexes = tuple(range(len(labels)))
    static_precision = precision(static_class, all_indexes)
    activation_precision = precision(activation_class, all_indexes)
    if static_precision is None or activation_precision is None:
        raise StaticFeatureStudyError("paired precision requires held-out rows")

    def precision_gain(indexes: Sequence[int]) -> float | None:
        left = precision(static_class, indexes)
        right = precision(activation_class, indexes)
        return None if left is None or right is None else right - left

    def error_gain(
        function: Callable[[Sequence[float], Sequence[float]], float],
    ) -> Callable[[Sequence[int]], float]:
        def apply(indexes: Sequence[int]) -> float:
            actual = _indexes(targets, indexes)
            return function(actual, _indexes(static_reg, indexes)) - function(
                actual, _indexes(activation_reg, indexes)
            )

        return apply

    gains = (
        _gain("auc_gain", groups, activation_auc - static_auc, auc_gain, config, 20),
        _gain(
            "mae_reduction",
            groups,
            mae(targets, static_reg) - mae(targets, activation_reg),
            error_gain(mae),
            config,
            21,
        ),
        _gain(
            f"precision_at_{config.top_n}_gain",
            groups,
            activation_precision - static_precision,
            precision_gain,
            config,
            22,
        ),
        _gain(
            "rmse_reduction",
            groups,
            rmse(targets, static_reg) - rmse(targets, activation_reg),
            error_gain(rmse),
            config,
            23,
        ),
    )
    return tuple(sorted(gains, key=lambda gain: gain.name))


def run_activation_feature_ablation(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: StaticFeatureStudyConfig | None = None,
) -> ActivationFeatureAblationResult:
    """Fit both profiles on identical rows and bootstrap their paired gains."""

    resolved = config or StaticFeatureStudyConfig()
    static = run_feature_profile_study(records, split, FeatureProfile.STATIC_ONLY, resolved)
    activation = run_feature_profile_study(
        records, split, FeatureProfile.STATIC_ACTIVATION, resolved
    )
    return ActivationFeatureAblationResult(
        static,
        activation,
        paired_feature_gains(static, activation, resolved),
    )
