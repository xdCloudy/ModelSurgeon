"""Coverage for bounded subprocess capture and environment filtering."""

from __future__ import annotations

import sys

import pytest

from modelsurgeon.utils.subprocesses import (
    run_bounded_process,
    sanitized_subprocess_environment,
)


def _run(script: str, **options: object):
    return run_bounded_process(
        [sys.executable, "-c", script],
        b"",
        timeout_seconds=options.pop("timeout_seconds", 3.0),
        stdout_limit=options.pop("stdout_limit", 16_384),
        stderr_limit=options.pop("stderr_limit", 16_384),
        terminate_grace_seconds=0.1,
        **options,
    )


def test_bounded_process_captures_stdout_and_stderr() -> None:
    result = _run("import sys; print('out'); print('err', file=sys.stderr)")

    assert result.returncode == 0
    assert result.stdout.decode().splitlines() == ["out"]
    assert result.stderr.decode().splitlines() == ["err"]
    assert not result.timed_out
    assert not result.output_exceeded


@pytest.mark.parametrize(
    ("stream", "limit_name", "expected_total_name"),
    [
        ("stdout", "stdout_limit", "stdout_total_bytes"),
        ("stderr", "stderr_limit", "stderr_total_bytes"),
    ],
)
def test_bounded_process_stops_when_a_stream_exceeds_its_limit(
    stream: str, limit_name: str, expected_total_name: str
) -> None:
    write = "sys.stdout.write" if stream == "stdout" else "sys.stderr.write"
    flush = "sys.stdout.flush" if stream == "stdout" else "sys.stderr.flush"
    result = _run(
        f"import sys;\nwhile True:\n {write}('x' * 4096); {flush}()",
        **{limit_name: 4096},
    )

    assert result.output_exceeded
    assert getattr(result, expected_total_name) > 4096
    assert len(getattr(result, stream)) <= 4096


def test_bounded_process_applies_a_combined_output_limit() -> None:
    result = _run(
        "import sys;\nwhile True:\n print('x' * 1024); print('y' * 1024, file=sys.stderr)",
        stdout_limit=16_384,
        stderr_limit=16_384,
        total_output_limit=4096,
    )

    assert result.output_exceeded
    assert result.stdout_total_bytes + result.stderr_total_bytes > 4096
    assert len(result.stdout) + len(result.stderr) <= 4096


def test_bounded_process_terminates_a_timed_out_child() -> None:
    result = _run("import time; time.sleep(30)", timeout_seconds=0.1)

    assert result.timed_out
    assert result.returncode != 0


def test_sanitized_environment_omits_ambient_secrets_and_keeps_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODELSURGEON_TEST_SECRET", "ambient-secret")

    environment = sanitized_subprocess_environment({"MODELSURGEON_TEST_MODE": "safe"})

    assert "MODELSURGEON_TEST_SECRET" not in environment
    assert environment["MODELSURGEON_TEST_MODE"] == "safe"


@pytest.mark.parametrize("override", [{"BAD=KEY": "value"}, {"BAD": "nul\0value"}, {1: "value"}])
def test_sanitized_environment_rejects_invalid_overrides(override: object) -> None:
    with pytest.raises(ValueError, match="valid strings"):
        sanitized_subprocess_environment(override)  # type: ignore[arg-type]
