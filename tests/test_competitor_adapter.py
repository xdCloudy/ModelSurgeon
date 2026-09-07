"""Conformance tests for bounded external competitor execution."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from modelsurgeon.adapters import (
    CompetitorBudget,
    CompetitorIdentity,
    CompetitorOutcome,
    ModelFormat,
    ModelSource,
    SubprocessCompetitorAdapter,
)
from modelsurgeon.adapters.competitor import CompetitorAdapterError

_SUCCESS_CODE = (
    "import hashlib,json,pathlib,shutil,sys; "
    "s=pathlib.Path(sys.argv[1]); o=pathlib.Path(sys.argv[2]); "
    "shutil.copyfile(s,o); b=o.read_bytes(); "
    "print(json.dumps(dict(status='success', artifact_path=str(o), "
    "artifact_sha256=hashlib.sha256(b).hexdigest(), artifact_size_bytes=len(b))))"
)
_BASELINE_IDENTITY = CompetitorIdentity(
    "python-copy-baseline",
    "3.12.13",
    "Python-2.0",
    "https://www.python.org/psf/license/",
)
_SOURCE = ModelSource(ModelFormat.SAFETENSORS, "fixture/source.safetensors", "fixture-v1")


def _adapter(code: str, *, budget: CompetitorBudget | None = None) -> SubprocessCompetitorAdapter:
    return SubprocessCompetitorAdapter(
        _BASELINE_IDENTITY,
        model_format=ModelFormat.SAFETENSORS,
        operations=frozenset({"compress"}),
        version_command=(sys.executable, "--version"),
        command_template=(sys.executable, "-c", code, "{source}", "{output}"),
        budget=budget
        or CompetitorBudget(
            max_wall_seconds=3.0,
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
            max_artifact_bytes=1024 * 1024,
            max_disk_bytes=1024 * 1024,
        ),
        config={"baseline": "copy", "revision": "3.12.13"},
    )


def _source_file(tmp_path: Path) -> tuple[Path, bytes]:
    source_path = tmp_path / "source.safetensors"
    source_bytes = b"immutable source checkpoint\n"
    source_path.write_bytes(source_bytes)
    return source_path, source_bytes


def test_pinned_baseline_requires_success_before_advertising_support(tmp_path: Path) -> None:
    source_path, source_bytes = _source_file(tmp_path)
    destination = tmp_path / "published.bin"
    adapter = _adapter(_SUCCESS_CODE)

    assert adapter.capability_decision("compress") == (
        False,
        "no executable-version and artifact-bound successful result is retained",
    )
    result = adapter.run(_SOURCE, source_path, destination, operation="compress", seed=11)

    assert result.outcome is CompetitorOutcome.SUCCESS
    assert result.provenance.executable_version
    assert result.provenance.artifact_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert destination.read_bytes() == source_bytes
    assert source_path.read_bytes() == source_bytes
    assert adapter.capabilities == frozenset({"compress"})
    assert adapter.capability_decision("compress")[0] is True
    assert result.run_id.startswith("competitor_")


@pytest.mark.parametrize(
    ("code", "outcome"),
    (
        ("raise SystemExit(7)", CompetitorOutcome.CRASH),
        ("print('not-json')", CompetitorOutcome.MALFORMED_OUTPUT),
        (
            "import time; time.sleep(2)",
            CompetitorOutcome.TIMEOUT,
        ),
        ("print('x' * 1000)", CompetitorOutcome.BUDGET_EXHAUSTED),
    ),
)
def test_failure_states_are_distinct(
    tmp_path: Path,
    code: str,
    outcome: CompetitorOutcome,
) -> None:
    source_path, _ = _source_file(tmp_path)
    budget = CompetitorBudget(
        max_wall_seconds=0.15 if outcome is CompetitorOutcome.TIMEOUT else 3.0,
        max_stdout_bytes=64 if outcome is CompetitorOutcome.BUDGET_EXHAUSTED else 4096,
        max_stderr_bytes=4096,
        max_artifact_bytes=1024 * 1024,
        max_disk_bytes=1024 * 1024,
    )
    result = _adapter(code, budget=budget).run(
        _SOURCE,
        source_path,
        tmp_path / "published.bin",
        operation="compress",
        seed=23,
    )

    assert result.outcome is outcome
    assert result.outcome is not CompetitorOutcome.SUCCESS


def test_unsupported_format_is_explicit_and_does_not_execute(tmp_path: Path) -> None:
    source_path, _ = _source_file(tmp_path)
    adapter = _adapter(_SUCCESS_CODE)
    source = ModelSource(ModelFormat.GGUF, "fixture/source.gguf", "fixture-v1")

    result = adapter.run(
        source,
        source_path,
        tmp_path / "published.bin",
        operation="compress",
        seed=47,
    )

    assert result.outcome is CompetitorOutcome.UNSUPPORTED
    assert not (tmp_path / "published.bin").exists()


def test_source_mutation_is_rejected_and_not_published(tmp_path: Path) -> None:
    source_path, source_bytes = _source_file(tmp_path)
    code = (
        "import hashlib,json,pathlib,shutil,sys; "
        "s=pathlib.Path(sys.argv[1]); o=pathlib.Path(sys.argv[2]); "
        "s.write_bytes(b'mutated'); shutil.copyfile(s,o); b=o.read_bytes(); "
        "print(json.dumps(dict(status='success', artifact_path=str(o), "
        "artifact_sha256=hashlib.sha256(b).hexdigest(), artifact_size_bytes=len(b))))"
    )

    result = _adapter(code).run(
        _SOURCE,
        source_path,
        tmp_path / "published.bin",
        operation="compress",
        seed=11,
    )

    assert result.outcome is CompetitorOutcome.SOURCE_MUTATED
    assert source_path.read_bytes() != source_bytes
    assert not (tmp_path / "published.bin").exists()


def test_destination_inside_source_is_rejected(tmp_path: Path) -> None:
    source_path = tmp_path / "source-model"
    source_path.mkdir()
    (source_path / "weights.bin").write_bytes(b"immutable")

    with pytest.raises(CompetitorAdapterError, match="inside the source"):
        _adapter(_SUCCESS_CODE).run(
            _SOURCE,
            source_path,
            source_path / "nested" / "published.bin",
            operation="compress",
            seed=11,
        )
