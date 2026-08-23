"""Tests for immutable revision-keyed baseline evaluation artifacts."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.evaluation.baseline_cache import (
    BaselineArtifact,
    BaselineCache,
    BaselineCacheError,
    BaselineCacheKey,
)


def _key(**overrides: str) -> BaselineCacheKey:
    values = {
        "model_revision": "model-a@abc",
        "dataset_revision": "data@def",
        "tokenizer_revision": "tok@ghi",
        "evaluator_version": "perplexity-v1",
    }
    values.update(overrides)
    return BaselineCacheKey(**values)


def _artifact(loss: float = 1.25) -> BaselineArtifact:
    return BaselineArtifact(
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        loss,
        2,
        (("accuracy", 0.5),),
    )


def test_matching_mutations_reuse_one_immutable_baseline(tmp_path) -> None:
    cache = BaselineCache(tmp_path)
    key = _key()
    calls = 0

    def compute() -> BaselineArtifact:
        nonlocal calls
        calls += 1
        return _artifact()

    first = cache.get_or_compute(key, compute)
    second = cache.get_or_compute(key, compute)

    assert first == second == _artifact()
    assert calls == 1
    assert cache.load(key) == _artifact()
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "model-a@other"),
        ("dataset_revision", "data@other"),
        ("tokenizer_revision", "tok@other"),
        ("evaluator_version", "perplexity-v2"),
    ],
)
def test_mismatched_revisions_cannot_hit_existing_partition(
    tmp_path, field: str, value: str
) -> None:
    cache = BaselineCache(tmp_path)
    cache.write(_key(), _artifact())

    assert cache.load(_key(**{field: value})) is None
    assert cache.path_for(_key()) != cache.path_for(_key(**{field: value}))


def test_same_key_cannot_be_overwritten_with_different_artifact(tmp_path) -> None:
    cache = BaselineCache(tmp_path)
    key = _key()
    cache.write(key, _artifact())

    with pytest.raises(BaselineCacheError, match="immutable"):
        cache.write(key, _artifact(loss=2.0))
    assert cache.load(key) == _artifact()


def test_checksum_corruption_fails_closed(tmp_path) -> None:
    cache = BaselineCache(tmp_path)
    key = _key()
    cache.write(key, _artifact())
    path = cache.path_for(key)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["artifact"]["mean_loss"] = 9.0
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BaselineCacheError, match="checksum"):
        cache.load(key)


def test_serialized_budget_is_enforced_before_publication(tmp_path) -> None:
    cache = BaselineCache(tmp_path, max_serialized_bytes=32)
    key = _key()
    with pytest.raises(BaselineCacheError, match="budget"):
        cache.write(key, _artifact())
    assert not cache.path_for(key).exists()


def test_invalid_artifact_and_key_values_fail_early() -> None:
    with pytest.raises(BaselineCacheError, match="revisions"):
        _key(model_revision="")
    with pytest.raises(BaselineCacheError, match="finite"):
        BaselineArtifact(((float("nan"),),), 1.0, 1)
    with pytest.raises(BaselineCacheError, match="canonical"):
        BaselineArtifact(((1.0,),), 1.0, 1, (("z", 1.0), ("a", 2.0)))
