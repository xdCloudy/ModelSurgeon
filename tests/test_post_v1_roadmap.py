from __future__ import annotations

from collections import Counter

from tools.bootstrap_post_v1 import (
    REQUIRED_BODY_SECTIONS,
    blocks_by_key,
    render_body,
    validate_catalog,
)
from tools.post_v1_roadmap_catalog import ISSUES, MILESTONES
from tools.roadmap_catalog import ISSUES as LEGACY_ISSUES


def test_post_v1_catalog_is_valid_and_milestones_are_bounded() -> None:
    counts = validate_catalog()

    assert counts == Counter(spec.milestone for spec in ISSUES)
    assert set(counts) == {title for title, _ in MILESTONES}
    assert all(6 <= count <= 15 for count in counts.values())


def test_post_v1_graph_is_dependency_ordered() -> None:
    seen = {spec.key for spec in LEGACY_ISSUES}

    for spec in ISSUES:
        assert set(spec.dependencies) <= seen
        seen.add(spec.key)


def test_rendered_issue_bodies_are_implementation_ready() -> None:
    all_specs = [*LEGACY_ISSUES, *ISSUES]
    numbers = {spec.key: index for index, spec in enumerate(all_specs, 1)}
    blocks = blocks_by_key()

    for spec in ISSUES:
        body = render_body(spec, numbers, blocks)
        assert f"<!-- roadmap-key:{spec.key} -->" in body
        assert all(section in body for section in REQUIRED_BODY_SECTIONS)
        assert "model families/checkpoints" in body
        assert "random seeds" in body
        assert "confidence intervals" in body
        assert "failure/unsupported-cell retention" in body
        assert "provenance" in body
