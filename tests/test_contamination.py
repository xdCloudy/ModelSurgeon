"""Tests for the bounded corpus contamination audit."""

from __future__ import annotations

import pytest

from modelsurgeon.datasets.contamination import (
    ContaminationAuditError,
    ContaminationPolicy,
    ContaminationStatus,
    CorpusItem,
    CorpusSource,
    FindingAction,
    FindingKind,
    audit_corpora,
)


def _sources(*, available: bool = True) -> tuple[CorpusSource, ...]:
    return (
        CorpusSource("train", "r1", "train", "MIT", "calibration", available, ("base",)),
        CorpusSource("test", "r1", "validation", "MIT", "benchmark", available, ("derived",)),
    )


def test_exact_overlap_blocks_publishability_and_is_deterministic() -> None:
    items = (CorpusItem("a", "train", "same text"), CorpusItem("b", "test", "same text"))
    first = audit_corpora(_sources(), items)
    second = audit_corpora(_sources(), items)
    assert first.status is ContaminationStatus.BLOCKED
    assert first.report_id == second.report_id
    assert first.findings[0].kind is FindingKind.EXACT_OVERLAP
    assert first.findings[0].action is FindingAction.BLOCK
    with pytest.raises(ContaminationAuditError, match="blocked"):
        first.require_publishable()


def test_near_overlap_moves_sample_and_changes_protocol_identity() -> None:
    items = (
        CorpusItem("a", "train", "alpha beta gamma delta epsilon"),
        CorpusItem("b", "test", "alpha beta gamma delta zeta"),
    )
    report = audit_corpora(
        _sources(), items, ContaminationPolicy(ngram_size=2, near_jaccard_threshold=0.5)
    )
    assert report.status is ContaminationStatus.BLOCKED
    assert report.findings[0].kind is FindingKind.NEAR_OVERLAP
    assert report.findings[0].action is FindingAction.MOVE
    assert report.remediated_protocol_id("protocol-v1") != "protocol-v1"


def test_ancestry_and_unavailable_sources_are_not_clean() -> None:
    items = (
        CorpusItem("a", "train", "one", ancestry_ids=("model-parent",)),
        CorpusItem("b", "test", "two", ancestry_ids=("model-parent",)),
    )
    blocked = audit_corpora(_sources(), items)
    unknown = audit_corpora(
        _sources(available=False), (CorpusItem("a", "train", "one"), CorpusItem("b", "test", "two"))
    )
    assert blocked.status is ContaminationStatus.BLOCKED
    assert any(item.kind is FindingKind.ANCESTRY_OVERLAP for item in blocked.findings)
    assert unknown.status is ContaminationStatus.UNKNOWN
    assert any(item.kind is FindingKind.UNAVAILABLE_SOURCE for item in unknown.findings)


def test_resource_limit_retains_audit_limitation() -> None:
    report = audit_corpora(
        _sources(), (CorpusItem("a", "train", "one"),), ContaminationPolicy(max_items=0 + 1)
    )
    assert report.status is ContaminationStatus.CLEAN
    limited = audit_corpora(
        _sources(),
        (CorpusItem("a", "train", "one"), CorpusItem("b", "test", "two")),
        ContaminationPolicy(max_items=1),
    )
    assert limited.status is ContaminationStatus.UNKNOWN
    assert limited.findings[0].kind is FindingKind.RESOURCE_LIMIT
