"""Tests for deterministic hostile-input corpus and bounded boundary runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from modelsurgeon.validation import (
    HostileFuzzError,
    HostileFuzzLimits,
    HostileOutcome,
    make_hostile_fuzz_corpus,
    resolve_scoped_path,
    run_parser_case,
    run_subprocess_case,
)


def test_corpus_is_deterministic_and_covers_all_boundaries() -> None:
    first = make_hostile_fuzz_corpus(seed=406)
    second = make_hostile_fuzz_corpus(seed=406)
    assert first.corpus_id == second.corpus_id
    assert {case.kind.value for case in first.cases} == {
        "hf_checkpoint", "safetensors", "gguf", "config", "manifest", "path", "subprocess"
    }
    assert all(case.minimized for case in first.cases)
    assert all(len(case.payload) <= case.limits.max_input_bytes for case in first.cases)
    assert first.to_record() == second.to_record()


def test_case_ids_and_manifest_are_content_addressed(tmp_path: Path) -> None:
    corpus = make_hostile_fuzz_corpus()
    path = tmp_path / "corpus.json"
    corpus.write_manifest(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["corpus_id"] == corpus.corpus_id
    assert record["schema_version"] == "1.0"
    assert record["cases"][0]["payload_sha256"] == corpus.cases[0].digest


def test_parser_runner_retains_rejected_unsupported_and_unknown() -> None:
    corpus = make_hostile_fuzz_corpus()
    rejected = run_parser_case(corpus.cases[0], lambda payload: (_ for _ in ()).throw(ValueError()))
    unsupported = run_parser_case(
        corpus.cases[0], lambda payload: (_ for _ in ()).throw(NotImplementedError())
    )
    failed = run_parser_case(corpus.cases[0], lambda payload: (_ for _ in ()).throw(RuntimeError()))
    assert rejected.outcome is HostileOutcome.REJECTED
    assert unsupported.outcome is HostileOutcome.UNSUPPORTED
    assert failed.outcome is HostileOutcome.FAILED


def test_path_runner_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    root.mkdir()
    assert resolve_scoped_path(root, "nested/model.gguf") == root / "nested/model.gguf"
    with pytest.raises(HostileFuzzError, match="escapes"):
        resolve_scoped_path(root, "../outside.gguf")
    alias = root / "alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(HostileFuzzError, match="symlink"):
        resolve_scoped_path(root, "alias/file.gguf")


def test_subprocess_runner_has_timeout_and_no_shell() -> None:
    case = make_hostile_fuzz_corpus(
        limits=HostileFuzzLimits(timeout_seconds=0.05, max_output_bytes=32)
    ).cases[-1]
    result = run_subprocess_case(
        case,
        [sys.executable, "-c", "import time; time.sleep(1)"],
    )
    assert result.outcome is HostileOutcome.TIMEOUT
