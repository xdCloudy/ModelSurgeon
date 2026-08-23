import json
from pathlib import Path

import pytest

from modelsurgeon.features.cache import (
    FEATURE_CACHE_SCHEMA_VERSION,
    FeatureCacheError,
    FeaturePartitionCache,
    FeaturePartitionKey,
)
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId


def _record(version: str = "1") -> FeatureRecord:
    return FeatureRecord(
        component_id=ComponentId.parse("model.layers.0"),
        name="weight_mean",
        kind=FeatureKind.SCALAR,
        value=1.25,
        dtype="float64",
        extractor="weight_stats",
        extractor_version=version,
        precision=PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "float32",
            "float64",
        ),
        metadata=(("source", "fixture"),),
    )


def _key(
    *,
    model_revision: str = "model-a",
    input_revision: str = "input-a",
    extractor_version: str = "1",
) -> FeaturePartitionKey:
    return FeaturePartitionKey(
        model_revision=model_revision,
        input_revision=input_revision,
        component_id=ComponentId.parse("model.layers.0"),
        extractor="weight_stats",
        extractor_version=extractor_version,
    )


def test_partition_round_trip_is_typed_and_atomic(tmp_path: Path) -> None:
    cache = FeaturePartitionCache(tmp_path)
    key = _key()
    written = cache.write(key, (_record(),))
    loaded = cache.load(key)

    assert loaded == written
    assert loaded is not None
    assert loaded.records == (_record(),)
    assert cache.path_for(key).exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


def test_revision_or_extractor_change_is_a_cache_miss(tmp_path: Path) -> None:
    cache = FeaturePartitionCache(tmp_path)
    cache.write(_key(), (_record(),))

    assert cache.load(_key(model_revision="model-b")) is None
    assert cache.load(_key(input_revision="input-b")) is None
    assert cache.load(_key(extractor_version="2")) is None


def test_incomplete_published_partition_is_not_a_cache_hit(tmp_path: Path) -> None:
    cache = FeaturePartitionCache(tmp_path)
    key = _key()
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache.path_for(key).write_text(
        json.dumps(
            {
                "cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                "complete": False,
            }
        ),
        encoding="utf-8",
    )

    assert cache.load(key) is None


def test_checksum_corruption_fails_closed(tmp_path: Path) -> None:
    cache = FeaturePartitionCache(tmp_path)
    key = _key()
    cache.write(key, (_record(),))
    path = cache.path_for(key)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["records_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(FeatureCacheError, match="checksum mismatch"):
        cache.load(key)


def test_partition_rejects_record_key_mismatch(tmp_path: Path) -> None:
    cache = FeaturePartitionCache(tmp_path)
    mismatched_key = _key(extractor_version="2")

    with pytest.raises(FeatureCacheError, match="extractor version"):
        cache.write(mismatched_key, (_record(),))
