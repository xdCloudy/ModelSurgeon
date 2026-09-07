from __future__ import annotations

import json

import pytest

from modelsurgeon.surgery import (
    RepairSelectionCandidate,
    RepairSelectionConfig,
    RepairSelectionStrategy,
    SelectionExclusionReason,
    TeacherCacheChunk,
    TeacherCacheEncoding,
    TeacherCacheError,
    TeacherCacheKey,
    TeacherCacheLimits,
    TeacherCacheStore,
    TeacherCacheTarget,
    select_repair_examples,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _key(*, tokenizer: int = 1) -> TeacherCacheKey:
    return TeacherCacheKey(
        "teacher/model",
        "teacher-rev-1",
        _digest(tokenizer),
        "repair-examples-v1",
        TeacherCacheTarget.LOGITS,
        "vocab-4",
        TeacherCacheEncoding.FP32,
    )


def _candidate(
    index: int,
    *,
    domain: str = "general",
    partition: str = "train",
    tokens: int = 3,
) -> RepairSelectionCandidate:
    return RepairSelectionCandidate(
        f"sample-{index}",
        _digest(index + 10),
        domain,
        tokens,
        float(index),
        float(index) / 10.0,
        float(index) / 10.0,
        partition,
        {"index": index},
    )


def test_teacher_cache_is_chunked_immutable_resumable_and_keyed(tmp_path) -> None:
    store = TeacherCacheStore(
        tmp_path,
        TeacherCacheLimits(max_chunk_values=3, max_total_bytes=10_000),
    )
    key = _key()
    first = TeacherCacheChunk(0, 2, (0.1, 0.2))
    second = TeacherCacheChunk(1, 1, (0.3,))

    assert store.load(key) is None
    store.write_chunk(key, first)
    with pytest.raises(TeacherCacheError, match="incomplete"):
        store.publish(key, 2, 3)
    store.write_chunk(key, second)
    manifest = store.publish(key, 2, 3)
    loaded, telemetry = store.access(key)

    assert loaded == manifest
    assert telemetry.hit
    assert telemetry.chunk_count == 2
    assert telemetry.bytes_read == manifest.serialized_bytes
    assert store.load(_key(tokenizer=2)) is None
    with pytest.raises(TeacherCacheError, match="immutable"):
        store.write_chunk(key, TeacherCacheChunk(0, 2, (9.0, 9.0)))


def test_teacher_cache_detects_corrupted_chunk_before_reuse(tmp_path) -> None:
    store = TeacherCacheStore(tmp_path)
    key = _key()
    store.write_chunk(key, TeacherCacheChunk(0, 1, (0.5,)))
    store.publish(key, 1, 1)
    chunk_path = tmp_path / key.cache_id / "00000000.chunk.json"
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    payload["values"] = [0.6]
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TeacherCacheError, match="checksum"):
        store.load(key)


def test_selection_is_deterministic_bounded_and_excludes_heldout() -> None:
    candidates = (
        _candidate(0, domain="a"),
        _candidate(1, domain="b"),
        _candidate(2, domain="a"),
        _candidate(3, domain="b", partition="validation"),
        _candidate(4, domain="c", partition="test"),
    )
    config = RepairSelectionConfig(
        max_examples=2,
        max_tokens=6,
        seed=17,
        strategy=RepairSelectionStrategy.DOMAIN_BALANCED,
    )
    first = select_repair_examples(candidates, config)
    second = select_repair_examples(candidates, config)

    assert first.selection_id == second.selection_id
    assert {item.domain for item in first.selected} == {"a", "b"}
    assert first.token_count == 6
    assert {item.sample_id for item in first.selected}.isdisjoint({"sample-3", "sample-4"})
    assert {item.reason for item in first.exclusions} == {
        SelectionExclusionReason.HELDOUT,
        SelectionExclusionReason.EXAMPLE_BUDGET,
    }
    assert first.to_record()["selection_id"] == first.selection_id


def test_selection_retains_budget_exclusions_and_predicted_order() -> None:
    candidates = (_candidate(0, tokens=4), _candidate(1, tokens=4), _candidate(2, tokens=4))
    report = select_repair_examples(
        candidates,
        RepairSelectionConfig(
            max_examples=3,
            max_tokens=8,
            seed=0,
            strategy=RepairSelectionStrategy.PREDICTED_RECOVERABILITY,
        ),
    )

    assert report.token_count == 8
    assert len(report.selected) == 2
    assert report.exclusions[-1].reason is SelectionExclusionReason.TOKEN_BUDGET
