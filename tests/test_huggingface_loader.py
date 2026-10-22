"""Focused tests for the safe Hugging Face causal LM loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from modelsurgeon.adapters.huggingface import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    HuggingFaceModelError,
    load_causal_lm,
    loader,
)


class FakeAutoModel:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    commit_hash: str | None = None

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> object:
        cls.calls.append((model, kwargs))
        resolved_revision = cls.commit_hash or kwargs["revision"]
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=resolved_revision))


class FakeEntryNotFoundError(Exception):
    pass


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeAutoModel.calls.clear()
    FakeAutoModel.commit_hash = None
    resolved_revision = "b" * 40
    snapshot_root = tmp_path / "models--org--model" / "snapshots" / resolved_revision
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "config.json").write_text("{}", encoding="utf-8")

    def fake_hf_hub_download(
        *, repo_id: str, filename: str, revision: str | None, local_files_only: bool
    ) -> str:
        if filename == "config.json":
            return str(snapshot_root / filename)
        raise FakeEntryNotFoundError(filename)

    def fake_import(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(AutoModelForCausalLM=FakeAutoModel)
        if name == "huggingface_hub":
            return SimpleNamespace(hf_hub_download=fake_hf_hub_download)
        if name == "huggingface_hub.errors":
            return SimpleNamespace(
                EntryNotFoundError=FakeEntryNotFoundError,
                LocalEntryNotFoundError=FakeEntryNotFoundError,
            )
        if name == "torch":
            return SimpleNamespace(float16="torch.float16")
        raise ImportError(name)

    monkeypatch.setattr(loader, "import_module", fake_import)


def test_safe_defaults_load_on_cpu_and_disable_remote_code() -> None:
    result = load_causal_lm(HuggingFaceLoadRequest(model="org/model"))

    source, options = FakeAutoModel.calls[-1]
    assert source == "org/model"
    assert options == {
        "revision": "b" * 40,
        "trust_remote_code": False,
        "device_map": {"": "cpu"},
        "torch_dtype": "auto",
        "local_files_only": False,
        "low_cpu_mem_usage": True,
    }
    assert result.provenance.resolved_revision == "b" * 40
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
    assert options["revision"] == "b" * 40
    assert options["trust_remote_code"] is True
    assert options["device_map"] == "auto"
    assert options["torch_dtype"] == "torch.float16"
    assert options["local_files_only"] is True
    assert options["low_cpu_mem_usage"] is False
    assert result.provenance.requested_revision == "release-tag"
    assert result.provenance.resolved_revision == "b" * 40


def test_requested_revision_is_pinned_before_loading_when_model_omits_commit_hash() -> None:
    FakeAutoModel.commit_hash = ""

    result = load_causal_lm(
        HuggingFaceLoadRequest(model="org/model", revision="immutable-revision")
    )

    assert result.provenance.resolved_revision == "b" * 40


def test_unpinned_remote_revision_is_resolved_before_loading() -> None:
    FakeAutoModel.commit_hash = ""

    result = load_causal_lm(HuggingFaceLoadRequest(model="org/model"))

    assert result.provenance.resolved_revision == "b" * 40


def test_local_checkpoint_rejects_traversal_before_model_loading(tmp_path: Path) -> None:
    model_root = tmp_path / "local-model"
    model_root.mkdir()
    (model_root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"../outside.safetensors"}}', encoding="utf-8"
    )

    with pytest.raises(HuggingFaceModelError, match="escapes"):
        load_causal_lm(HuggingFaceLoadRequest(model=str(model_root)))

    assert FakeAutoModel.calls == []


def test_local_checkpoint_index_size_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "local-model"
    model_root.mkdir()
    (model_root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(loader, "MAX_CHECKPOINT_INDEX_BYTES", 1)

    with pytest.raises(HuggingFaceModelError, match="exceeds"):
        load_causal_lm(HuggingFaceLoadRequest(model=str(model_root)))

    assert FakeAutoModel.calls == []


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
