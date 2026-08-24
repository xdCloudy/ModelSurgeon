"""Paired ablation of source quantization, precision, model, and hardware context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest

from .activation_feature_study import PairedGainEstimate, paired_feature_gains
from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyConfig,
    StaticFeatureStudyResult,
    run_feature_profile_study,
)


@dataclass(frozen=True, slots=True)
class QuantizationContextAblationResult:
    context_blind: StaticFeatureStudyResult
    context_aware: StaticFeatureStudyResult
    gains: tuple[PairedGainEstimate, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "context_blind": self.context_blind.to_record(),
            "context_aware": self.context_aware.to_record(),
            "paired_gains": [item.to_record() for item in self.gains],
        }


def run_quantization_context_ablation(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: StaticFeatureStudyConfig | None = None,
) -> QuantizationContextAblationResult:
    """Compare identical gradient-profile rows with and without source-only context."""

    resolved = config or StaticFeatureStudyConfig()
    blind = run_feature_profile_study(
        records,
        split,
        FeatureProfile.STATIC_ACTIVATION_GRADIENT,
        resolved,
        include_context=False,
    )
    aware = run_feature_profile_study(
        records,
        split,
        FeatureProfile.STATIC_ACTIVATION_GRADIENT,
        resolved,
        include_context=True,
    )
    return QuantizationContextAblationResult(
        blind,
        aware,
        paired_feature_gains(blind, aware, resolved),
    )
