"""Dependency-free contract tests for the concrete Hugging Face proof runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofError,
    _channel_coordinates,
    _DownProjectionChannelMask,
    _DownProjectionChannelMaskSet,
    _find_module,
    _read_calibration_text,
    _token_chunks,
    _tokenizer_revision,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationKind, MutationRequest


def _channel_request(layer: int = 2, channel: int = 7) -> MutationRequest:
    return MutationRequest(
        MutationKind.MASK,
        (ComponentId.parse(f"model.layers.{layer}.mlp.channel.{channel}"),),
        (
            ("candidate_scope", "channel"),
            ("channel_index", channel),
            ("layer_index", layer),
        ),
    )


def test_config_and_token_chunk_contracts(tmp_path: Path) -> None:
    text = tmp_path / "calibration.txt"
    text.write_text("hello world", encoding="utf-8")
    config = HuggingFaceMLPProofConfig("model", text, sequence_length=3, max_tokens=5)
    assert config.sequence_length == 3
    assert _token_chunks([1, 2, 3, 4, 5, 6], sequence_length=3, max_tokens=5) == (
        (1, 2, 3),
        (4, 5),
    )

    with pytest.raises(HuggingFaceMLPProofError, match="sequence_length"):
        HuggingFaceMLPProofConfig("model", text, sequence_length=1)
    with pytest.raises(HuggingFaceMLPProofError, match="safe_perplexity_delta"):
        HuggingFaceMLPProofConfig("model", text, safe_perplexity_delta=-1.0)
    with pytest.raises(HuggingFaceMLPProofError, match="input_ids"):
        _token_chunks([1, "bad"], sequence_length=2, max_tokens=2)


def test_calibration_text_revision_is_content_addressed(tmp_path: Path) -> None:
    text = tmp_path / "calibration.txt"
    text.write_text("stable calibration corpus", encoding="utf-8")
    payload, revision = _read_calibration_text(text)
    assert payload == "stable calibration corpus"
    assert len(revision) == 64


def test_channel_request_must_match_scope_and_target() -> None:
    assert _channel_coordinates(_channel_request()) == (2, 7)

    wrong_scope = MutationRequest(
        MutationKind.MASK,
        (ComponentId.parse("model.layers.2.mlp.channel.7"),),
        (
            ("candidate_scope", "component"),
            ("channel_index", 7),
            ("layer_index", 2),
        ),
    )
    with pytest.raises(HuggingFaceMLPProofError, match="MLP-channel"):
        _channel_coordinates(wrong_scope)

    mismatched = MutationRequest(
        MutationKind.MASK,
        (ComponentId.parse("model.layers.2.mlp.channel.8"),),
        (
            ("candidate_scope", "channel"),
            ("channel_index", 7),
            ("layer_index", 2),
        ),
    )
    with pytest.raises(HuggingFaceMLPProofError, match="disagrees"):
        _channel_coordinates(mismatched)


def test_module_resolution_requires_one_unambiguous_suffix() -> None:
    sentinel = object()
    assert (
        _find_module(
            {"model.layers.0.mlp.down_proj": sentinel},
            "model.layers.0.mlp.down_proj",
        )
        is sentinel
    )
    assert (
        _find_module(
            {"prefix.model.layers.0.mlp.down_proj": sentinel},
            "model.layers.0.mlp.down_proj",
        )
        is sentinel
    )
    with pytest.raises(HuggingFaceMLPProofError, match="exactly one"):
        _find_module({}, "model.layers.0.mlp.down_proj")


class _FakeTensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = [list(row) for row in rows]
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def clone(self) -> _FakeTensor:
        return _FakeTensor(self.rows)

    def __setitem__(self, key: tuple[object, int], value: float) -> None:
        leading, column = key
        assert leading is Ellipsis
        for row in self.rows:
            row[column] = value


class _FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)


class _FakeHandle:
    def __init__(self, module: _FakeModule, hook: object) -> None:
        self.module = module
        self.hook = hook

    def remove(self) -> None:
        self.module.hooks.remove(self.hook)


class _FakeModule:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_pre_hook(self, hook: object) -> _FakeHandle:
        self.hooks.append(hook)
        return _FakeHandle(self, hook)

    def emit(self, tensor: _FakeTensor) -> _FakeTensor:
        inputs: tuple[object, ...] = (tensor,)
        for hook in tuple(self.hooks):
            inputs = hook(self, inputs)  # type: ignore[operator]
        result = inputs[0]
        assert isinstance(result, _FakeTensor)
        return result


def test_down_projection_mask_zeroes_only_selected_channel_and_removes_hook() -> None:
    module = _FakeModule()
    source = _FakeTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with _DownProjectionChannelMask(module, 1, _FakeTorch()):
        masked = module.emit(source)
        assert masked.rows == [[1.0, 0, 3.0], [4.0, 0, 6.0]]
        assert source.rows == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert len(module.hooks) == 1

    assert module.hooks == []
    assert module.emit(source).rows == source.rows


def test_down_projection_mask_set_uses_one_reversible_hook() -> None:
    module = _FakeModule()
    source = _FakeTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with _DownProjectionChannelMaskSet(module, (0, 2), _FakeTorch()):
        masked = module.emit(source)
        assert masked.rows == [[0, 2.0, 0], [0, 5.0, 0]]
        assert source.rows == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert len(module.hooks) == 1

    assert module.hooks == []


class _Tokenizer:
    def __init__(self, commit: str | None) -> None:
        self.init_kwargs = {} if commit is None else {"_commit_hash": commit}


def test_tokenizer_revision_prefers_resolved_commit_and_fails_closed() -> None:
    assert (
        _tokenizer_revision(
            _Tokenizer("abc123"),
            source="tokenizer",
            requested=None,
            model_source="model",
            model_revision="model-sha",
        )
        == "abc123"
    )
    assert (
        _tokenizer_revision(
            _Tokenizer(None),
            source="model",
            requested=None,
            model_source="model",
            model_revision="model-sha",
        )
        == "model-sha"
    )
    with pytest.raises(HuggingFaceMLPProofError, match="pin"):
        _tokenizer_revision(
            _Tokenizer(None),
            source="other-tokenizer",
            requested=None,
            model_source="model",
            model_revision="model-sha",
        )
