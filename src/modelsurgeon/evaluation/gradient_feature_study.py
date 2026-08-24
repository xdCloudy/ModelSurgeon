"""Paired activation versus activation-plus-gradient ablation for v0.8 Q3."""

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
class GradientFeatureAblationResult:
    static_activation: StaticFeatureStudyResult
    static_activation_gradient: StaticFeatureStudyResult
    gains: tuple[PairedGainEstimate, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "model": {
                "identifier": self.static_activation.model_identifier,
                "revision": self.static_activation.model_revision,
                "family": self.static_activation.family,
            },
            "static_activation": self.static_activation.to_record(),
            "static_activation_gradient": self.static_activation_gradient.to_record(),
            "paired_gains": [gain.to_record() for gain in self.gains],
        }


def run_gradient_feature_ablation(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: StaticFeatureStudyConfig | None = None,
) -> GradientFeatureAblationResult:
    """Fit both profiles on identical rows and bootstrap gradient-feature gains."""

    resolved = config or StaticFeatureStudyConfig()
    activation = run_feature_profile_study(
        records, split, FeatureProfile.STATIC_ACTIVATION, resolved
    )
    gradient = run_feature_profile_study(
        records, split, FeatureProfile.STATIC_ACTIVATION_GRADIENT, resolved
    )
    return GradientFeatureAblationResult(
        activation,
        gradient,
        paired_feature_gains(activation, gradient, resolved),
    )
