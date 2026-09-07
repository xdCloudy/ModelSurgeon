from __future__ import annotations

import pytest

from modelsurgeon.features import (
    ArchitectureKind,
    MetaFeatureError,
    MetaFeatureSample,
    MetaFeatureStatus,
    fit_meta_feature_normalizer,
)


def _sample(sample_id: str, partition: str, family: str, *, moe: bool = False) -> MetaFeatureSample:
    return MetaFeatureSample(
        sample_id,
        f"model-{sample_id}",
        partition,
        "mlp.channel",
        family,
        ArchitectureKind.MOE if moe else ArchitectureKind.DENSE,
        2,
        8,
        4096,
        4096,
        32,
        8,
        64 if moe else None,
        1_000_000,
        "q4_k",
        "cpu",
        ("remove:0", "remove:1"),
        (("quality_loss", 0.1),),
    )


def test_meta_normalizer_is_source_only_and_marks_unknowns() -> None:
    normalizer = fit_meta_feature_normalizer((_sample("source-a", "source", "llama"),))
    transformed = normalizer.transform((_sample("target-a", "target", "qwen"),))[0]

    assert transformed.schema_id == normalizer.schema_id
    assert transformed.categorical_known[0] is False
    assert dict(transformed.field_status)["architecture_family"] is MetaFeatureStatus.UNKNOWN
    assert "qwen" not in normalizer.categorical_vocabularies[0]
    assert normalizer.coverage((_sample("target-a", "target", "qwen"),)).unknown_fields > 0
    assert normalizer.compatibility("llama", _sample("target-a", "target", "qwen")).unknown_fields


def test_meta_normalizer_keeps_moe_explicitly_unsupported_and_rejects_target_fit() -> None:
    normalizer = fit_meta_feature_normalizer((_sample("source-a", "source", "llama"),))
    moe = normalizer.transform((_sample("moe", "target", "mixtral", moe=True),))[0]
    assert MetaFeatureStatus.UNSUPPORTED in dict(moe.field_status).values()

    with pytest.raises(MetaFeatureError, match="source-partition"):
        fit_meta_feature_normalizer((_sample("target", "target", "qwen"),))
