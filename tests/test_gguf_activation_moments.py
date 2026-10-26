from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from tools.collect_gguf_activation_moments import _verify_model_and_tokenizer

from modelsurgeon.adapters.gguf import load_gguf_activation_moments

_SOURCE_SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_sha256": _SOURCE_SHA256,
        "calibration": {
            "dataset": "example/dataset",
            "dataset_revision": "revision",
            "split": "train",
            "text_sha256": "b" * 64,
            "tokenizer": "example/model",
            "tokenizer_revision": "revision",
        },
        "tensors": {
            "blk.0.ffn_down.weight": {
                "input_axis": 0,
                "sample_count": 4,
                "second_moments": [0.25, 1.0],
            }
        },
    }


def _write(path: Path, payload: str | dict[str, object]) -> None:
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


def test_loads_source_bound_activation_moments(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    record = _manifest()
    _write(path, record)

    tensors, calibration = load_gguf_activation_moments(
        path, expected_source_sha256=_SOURCE_SHA256
    )

    assert tensors["blk.0.ffn_down.weight"] == {
        "input_axis": 0,
        "sample_count": 4,
        "second_moments": [0.25, 1.0],
    }
    assert calibration["split"] == "train"


def test_loads_legacy_flat_collector_provenance(tmp_path: Path) -> None:
    path = tmp_path / "legacy-moments.json"
    record = _manifest()
    nested = record.pop("calibration")
    record.update(
        calibration_dataset=nested["dataset"],
        calibration_split=nested["split"],
        calibration_sha256=nested["text_sha256"],
        tokenizer=nested["tokenizer"],
        tokenizer_revision=nested["tokenizer_revision"],
        tokenizer_vocab_size=128,
        bos_token_id=1,
        added_bos=False,
        calibration_chunks=8,
        context_size=512,
        calibration_token_count=4096,
    )
    _write(path, record)

    tensors, calibration = load_gguf_activation_moments(
        path, expected_source_sha256=_SOURCE_SHA256
    )

    assert tensors["blk.0.ffn_down.weight"]["sample_count"] == 4
    assert calibration["dataset_revision"] == f"sha256:{nested['text_sha256']}"
    assert calibration["calibration_token_count"] == 4096


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item.update(source_sha256="c" * 64), "different source"),
        (lambda item: item.update(schema_version=2), "unsupported"),
        (lambda item: item["calibration"].pop("split"), "needs split"),
        (lambda item: item["calibration"].update(text_sha256="bad"), "SHA-256"),
        (lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(input_axis=True), "axis"),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(sample_count=False),
            "sample count",
        ),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(
                second_moments=[float("nan")]
            ),
            "finite",
        ),
        (
            lambda item: item["tensors"]["blk.0.ffn_down.weight"].update(second_moments=[-1]),
            "non-negative",
        ),
        (lambda item: item.update(tensors={}), "non-empty"),
    ],
)
def test_rejects_invalid_activation_moment_records(
    tmp_path: Path, mutate: object, message: str
) -> None:
    record = _manifest()
    mutate(record)  # type: ignore[operator]
    path = tmp_path / "moments.json"
    _write(path, record)

    with pytest.raises(ValueError, match=message):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, '{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate key"):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_nonstandard_nan_json_values(tmp_path: Path) -> None:
    record = _manifest()
    record["tensors"]["blk.0.ffn_down.weight"]["second_moments"] = [float("nan")]
    path = tmp_path / "moments.json"
    _write(path, record)

    with pytest.raises(ValueError, match="finite"):
        load_gguf_activation_moments(path, expected_source_sha256=_SOURCE_SHA256)


def test_rejects_manifest_over_configured_limit(tmp_path: Path) -> None:
    path = tmp_path / "moments.json"
    _write(path, _manifest())

    with pytest.raises(ValueError, match="file limit"):
        load_gguf_activation_moments(
            path, expected_source_sha256=_SOURCE_SHA256, max_file_bytes=8
        )


class _FixtureTokenizer:
    vocab_size = 3
    bos_token_id = None
    eos_token_id = 2
    unk_token_id = None
    add_bos_token = False
    _merges: ClassVar[list[tuple[str, str]]] = [("a", "b")]
    _tokens: ClassVar[list[str]] = ["a", "b", "c", "added", "reserved"]

    def __len__(self) -> int:
        return 4

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._tokens[index] for index in ids]


def _qwen2_fixture() -> tuple[SimpleNamespace, _FixtureTokenizer, dict[str, object]]:
    tokenizer = _FixtureTokenizer()
    config = SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=2,
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_parameters={"rope_theta": 10000.0},
        rms_norm_eps=1e-6,
        vocab_size=5,
    )
    metadata: dict[str, object] = {
        "general.architecture": "qwen2",
        "qwen2.block_count": 2,
        "qwen2.embedding_length": 4,
        "qwen2.feed_forward_length": 8,
        "qwen2.attention.head_count": 2,
        "qwen2.attention.head_count_kv": 1,
        "qwen2.rope.freq_base": 10000.0,
        "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
        "tokenizer.ggml.tokens": [*tokenizer._tokens],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.bos_token_id": 3,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.add_bos_token": False,
    }
    return config, tokenizer, metadata


def test_verifies_dense_qwen2_metadata_and_padded_vocabulary() -> None:
    config, tokenizer, metadata = _qwen2_fixture()

    _verify_model_and_tokenizer(config, tokenizer, metadata)


def test_rejects_qwen2_vocabulary_that_does_not_match_model_config() -> None:
    config, tokenizer, metadata = _qwen2_fixture()
    config.vocab_size = 6

    with pytest.raises(ValueError, match="vocabulary differs"):
        _verify_model_and_tokenizer(config, tokenizer, metadata)


def test_rejects_moe_architecture_in_qwen2_activation_collector() -> None:
    config, tokenizer, metadata = _qwen2_fixture()
    metadata["general.architecture"] = "qwen2_moe"
    config.model_type = "qwen2_moe"

    with pytest.raises(ValueError, match="dense llama and qwen2"):
        _verify_model_and_tokenizer(config, tokenizer, metadata)
