from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.surgeon import (
    PretrainedRegistryError,
    PretrainedSurgeonCard,
    PretrainedSurgeonRegistry,
    TrainingModelIdentity,
)


def _card(bundle_digest: str, evidence_digest: str) -> PretrainedSurgeonCard:
    return PretrainedSurgeonCard(
        bundle_digest,
        "meta-surgeon-v1",
        "model-rev",
        (TrainingModelIdentity("tiny/source", "source-rev", "Q4_K_M"),),
        (evidence_digest,),
        1,
        1,
        1,
        ("mask",),
        ("q4_k",),
        ("cpu",),
        {"minimum_support_coverage": 0.8},
        {"level": "adaptable", "decision_id": "decision-1"},
        "Apache-2.0",
        (("calibration", None), ("ranking", 0.8)),
        ("not validated on MoE",),
    )


def test_signed_card_verifies_offline_and_lists_by_content_digest(tmp_path: Path) -> None:
    registry = PretrainedSurgeonRegistry(tmp_path / "registry")
    bundle = registry.bundles.artifacts.put_bytes(b"bundle-placeholder")
    evidence = registry.bundles.artifacts.put_bytes(b"evidence-placeholder")
    published = registry.publish(
        _card(str(bundle.metadata.digest), str(evidence.metadata.digest)),
        key_id="test-key",
        secret=b"test-secret",
    )
    verified = registry.verify(str(published.artifact.metadata.digest), secret=b"test-secret")
    assert verified.signed_identity is not None
    assert registry.list_cards() == (str(published.artifact.metadata.digest),)
    with pytest.raises(PretrainedRegistryError, match="signature"):
        registry.verify(str(published.artifact.metadata.digest), secret=b"wrong-secret")


def test_incompatible_card_is_rejected_before_model_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = PretrainedSurgeonRegistry(tmp_path / "registry")
    bundle = registry.bundles.artifacts.put_bytes(b"bundle-placeholder")
    evidence = registry.bundles.artifacts.put_bytes(b"evidence-placeholder")
    published = registry.publish(
        _card(str(bundle.metadata.digest), str(evidence.metadata.digest)),
        key_id="test-key",
        secret=b"test-secret",
    )
    monkeypatch.setattr(
        registry.bundles,
        "load",
        lambda digest: pytest.fail("bundle deserialized before card guard"),
    )
    with pytest.raises(PretrainedRegistryError, match="feature schema"):
        registry.resolve(
            str(published.artifact.metadata.digest),
            secret=b"test-secret",
            expected_feature_schema_version=2,
        )


def test_adapted_child_requires_immutable_parent_lineage() -> None:
    with pytest.raises(PretrainedRegistryError, match="lineage"):
        PretrainedSurgeonCard(
            "sha256:" + "a" * 64,
            "child",
            "rev",
            (TrainingModelIdentity("source", "rev"),),
            ("sha256:" + "b" * 64,),
            1,
            1,
            1,
            ("mask",),
            ("q4_k",),
            ("cpu",),
            {"threshold": 0.1},
            {"decision": "compatible"},
            "Apache-2.0",
            (),
            ("unmeasured",),
            parent_bundle_digest="sha256:" + "c" * 64,
        )
