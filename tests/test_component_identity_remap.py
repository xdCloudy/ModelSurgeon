"""Tests for explicit and composable post-surgery component identities."""

from __future__ import annotations

import pytest

from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
    IdentityDisposition,
    RemovedSourceIdentityError,
    UnknownSourceIdentityError,
)


def _id(name: str) -> ComponentId:
    return ComponentId.parse(f"model.{name}")


def test_retained_removed_renumbered_split_and_merged_are_explicit() -> None:
    a, b, c, d, e, f, g, h, i, j = map(_id, "abcdefghij")
    remap = ComponentIdentityRemap.build(
        (
            ComponentIdentityMapping(a, (a,), "unchanged"),
            ComponentIdentityMapping(b, (), "pruned"),
            ComponentIdentityMapping(c, (d,), "renumbered"),
            ComponentIdentityMapping(e, (f, g), "split"),
            ComponentIdentityMapping(h, (j,), "merged"),
            ComponentIdentityMapping(i, (j,), "merged"),
        )
    )
    assert remap.disposition(a) is IdentityDisposition.RETAINED
    assert remap.disposition(b) is IdentityDisposition.REMOVED
    assert remap.disposition(c) is IdentityDisposition.RENUMBERED
    assert remap.disposition(e) is IdentityDisposition.SPLIT
    assert remap.disposition(h) is IdentityDisposition.MERGED
    assert remap.resolve(c) == (d,)
    assert remap.to_record()["mappings"][1]["disposition"] == "removed"


def test_source_that_splits_into_a_shared_target_is_split_merged() -> None:
    a, b, left, shared = map(_id, ("a", "b", "left", "shared"))
    remap = ComponentIdentityRemap.build(
        (
            ComponentIdentityMapping(a, (left, shared), "split"),
            ComponentIdentityMapping(b, (shared,), "merge"),
        )
    )
    assert remap.disposition(a) is IdentityDisposition.SPLIT_MERGED


def test_removed_and_unknown_sources_never_resolve_silently() -> None:
    source = _id("source")
    remap = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(source, (), "removed"),)
    )
    with pytest.raises(RemovedSourceIdentityError, match="was removed"):
        remap.resolve(source)
    with pytest.raises(UnknownSourceIdentityError, match="no explicit"):
        remap.resolve(_id("unknown"))


def test_sequential_renumber_and_split_mappings_compose() -> None:
    old, middle, left, right = map(_id, ("old", "middle", "left", "right"))
    first = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(old, (middle,), "stage one"),)
    )
    second = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(middle, (left, right), "stage two"),)
    )
    composed = first.compose(second)
    assert composed.resolve(old) == (left, right)
    assert composed.disposition(old) is IdentityDisposition.SPLIT
    assert composed.mapping(old).reason == "stage one -> stage two"


def test_sequential_removal_propagates_and_incomplete_composition_fails() -> None:
    old, middle, other = map(_id, ("old", "middle", "other"))
    first = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(old, (middle,), "renumber"),)
    )
    removed = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(middle, (), "removed later"),)
    )
    composed = first.compose(removed)
    assert composed.disposition(old) is IdentityDisposition.REMOVED
    with pytest.raises(RemovedSourceIdentityError):
        composed.resolve(old)

    incomplete = ComponentIdentityRemap.retained((other,))
    with pytest.raises(UnknownSourceIdentityError, match="intermediate"):
        first.compose(incomplete)


def test_split_merged_composition_is_classified_without_information_loss() -> None:
    a, b, middle_a, middle_b, final = map(
        _id, ("a", "b", "middle_a", "middle_b", "final")
    )
    first = ComponentIdentityRemap.build(
        (
            ComponentIdentityMapping(a, (middle_a, middle_b), "split"),
            ComponentIdentityMapping(b, (middle_b,), "retarget"),
        )
    )
    second = ComponentIdentityRemap.build(
        (
            ComponentIdentityMapping(middle_a, (final,), "merge"),
            ComponentIdentityMapping(middle_b, (final,), "merge"),
        )
    )
    composed = first.compose(second)
    assert composed.resolve(a) == (final,)
    assert composed.disposition(a) is IdentityDisposition.MERGED
    assert composed.disposition(b) is IdentityDisposition.MERGED
