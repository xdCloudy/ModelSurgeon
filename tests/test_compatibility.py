from __future__ import annotations

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.surgeon import (
    CompatibilityContext,
    CompatibilityError,
    CompatibilityLevel,
    CompatibilityOperation,
    ModelLineageGraph,
    ModelLineageNode,
    decide_compatibility,
    validate_heldout_lineage,
)


def _context(
    model_id: str, family: ModelFamily, *, feature_id: str = "meta-v1"
) -> CompatibilityContext:
    return CompatibilityContext(
        model_id,
        "rev-1",
        family,
        "arch-v1",
        feature_id,
        1,
        "target-v1",
        1,
        "q4_k",
        "cpu",
        1,
        (CompatibilityOperation.INFER,),
    )


def test_lineage_rejects_cycles_and_heldout_ancestry_overlap() -> None:
    with pytest.raises(CompatibilityError, match="cycle"):
        ModelLineageGraph(
            (
                ModelLineageNode("a", "1", ModelFamily.LLAMA, "v1", parent_model_ids=("b",)),
                ModelLineageNode("b", "1", ModelFamily.LLAMA, "v1", parent_model_ids=("a",)),
            )
        )
    graph = ModelLineageGraph(
        (
            ModelLineageNode("base", "1", ModelFamily.LLAMA, "v1"),
            ModelLineageNode("derived", "2", ModelFamily.LLAMA, "v1", parent_model_ids=("base",)),
            ModelLineageNode("other", "1", ModelFamily.QWEN, "v1"),
        )
    )
    with pytest.raises(CompatibilityError, match="held-out"):
        validate_heldout_lineage(graph, ("base",), ("derived",))


def test_compatibility_is_machine_readable_and_fails_closed() -> None:
    source = _context("source", ModelFamily.LLAMA)
    target = _context("target", ModelFamily.QWEN, feature_id="meta-v2")
    decision = decide_compatibility(source, target, operation=CompatibilityOperation.INFER)
    assert decision.level is CompatibilityLevel.UNKNOWN
    assert not decision.allowed
    assert any(item.code == "missing-lineage" for item in decision.reasons)
    assert decision.to_record()["decision_id"] == decision.decision_id

    adaptable = decide_compatibility(
        source,
        _context("target", ModelFamily.QWEN),
        operation=CompatibilityOperation.INFER,
        lineage=ModelLineageGraph(
            (
                ModelLineageNode("source", "1", ModelFamily.LLAMA, "v1"),
                ModelLineageNode("target", "1", ModelFamily.QWEN, "v1"),
            )
        ),
        allow_adaptation=True,
    )
    assert adaptable.level is CompatibilityLevel.ADAPTABLE
    assert adaptable.allowed
