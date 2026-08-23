"""Focused tests for the safe Hugging Face causal LM loader."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from modelsurgeon.adapters.huggingface import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
    loader,
)


class FakeAutoModel:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    commit_hash = "a" * 40

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> object:
        cls.calls.append((model, kwargs))
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=cls.commit_hash))


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAutoModel.calls.clear()
    FakeAutoModel.commit_hash = "a" * 40

    def fake_import(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(AutoModelForCausalLM=FakeAutoModel)
        if name == "torch":
            return SimpleNamespace(float16="torch.float16")
        raise ImportError(name)

    monkeypatch.setattr(loader, "import_module", fake_import)


def test_safe_defaults_load_on_cpu_and_disable_remote_code() -> None:
    result = load_causal_lm(HuggingFaceLoadRequest(model="org/model"))

    source, options = FakeAutoModel.calls[-1]
    assert source == "org/model"
    assert options == {
        "revision": None,
        "trust_remote_code": False,
        "device_map": {"": "cpu"},
        "torch_dtype": "auto",
        "local_files_only": False,
        "low_cpu_mem_usage": True,
    }
    assert result.provenance.resolved_revision == "a" * 40
    assert result.provenance.to_record()["loader_options"] == {
        "trust_remote_code": False,
        "device_map": "cpu",
        "dtype": "auto",
        "local_files_only": False,
        "low_cpu_mem_usage": True,
    }


def test_explicit_loader_controls_and_revision_are_forwarded() -> None:
    request = HuggingFaceLoadRequest(
        model="org/model",
        revision="release-tag",
        trust_remote_code=True,
        device_map="auto",
        dtype=HuggingFaceDType.FLOAT16,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    result = load_causal_lm(request)

    _, options = FakeAutoModel.calls[-1]
    assert options["revision"] == "release-tag"
    assert options["trust_remote_code"] is True
    assert options["device_map"] == "auto"
    assert options["torch_dtype"] == "torch.float16"
    assert options["local_files_only"] is True
    assert options["low_cpu_mem_usage"] is False
    assert result.provenance.requested_revision == "release-tag"
    assert result.provenance.resolved_revision == "a" * 40


def test_pinned_revision_is_fallback_when_loader_omits_commit_hash() -> None:
    FakeAutoModel.commit_hash = ""

    result = load_causal_lm(
        HuggingFaceLoadRequest(model="org/model", revision="immutable-revision")
    )

    assert result.provenance.resolved_revision == "immutable-revision"


def test_unresolved_remote_revision_fails_closed() -> None:
    FakeAutoModel.commit_hash = ""

    with pytest.raises(RuntimeError, match="resolved Hub commit"):
        load_causal_lm(HuggingFaceLoadRequest(model="org/model"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model cannot be blank"),
        ({"model": "x", "revision": " "}, "revision cannot be blank"),
        ({"model": "x", "device_map": "cuda:0"}, "device_map must be one of"),
    ],
)
def test_invalid_requests_fail_before_loading(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HuggingFaceLoadRequest(**kwargs)  # type: ignore[arg-type]
