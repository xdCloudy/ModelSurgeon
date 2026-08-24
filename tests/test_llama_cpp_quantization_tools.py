"""Tests for pinned external llama.cpp quantization tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import llama_cpp_tools as tools
from modelsurgeon.adapters.gguf.llama_cpp_tools import (
    LlamaCppQuantizationConfig,
    LlamaCppQuantizationRecipe,
    LlamaCppQuantizationToolError,
    discover_llama_cpp_quantizer,
    quantize_gguf_with_llama_cpp,
)
from modelsurgeon.evaluation.llama_cpp import (
    LlamaCppToolProvenance,
    LlamaCppValidationError,
    _BoundedCommandResult,
)


def _binaries(tmp_path: Path) -> tuple[Path, Path]:
    suffix = ".exe" if tools.os.name == "nt" else ""
    quantizer = tmp_path / f"llama-quantize{suffix}"
    cli = tmp_path / f"llama-cli{suffix}"
    quantizer.write_bytes(b"quantizer")
    cli.write_bytes(b"cli")
    return quantizer, cli


def _install_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, list[tuple[str, ...]]]:
    quantizer, cli = _binaries(tmp_path)
    monkeypatch.setattr(tools, "_resolve_executable", lambda value, revision: quantizer)
    provenance = LlamaCppToolProvenance(
        cli,
        "95b8e33e16bb9a60de780a70930ebf729db6a90a",
        "95b8e33e1",
        (str(cli), "--version"),
        "version: 0.2.0-dev (build 10597, commit 95b8e33e1)",
    )
    monkeypatch.setattr(tools, "_probe_tool", lambda config: provenance)
    calls: list[tuple[str, ...]] = []

    def fake_help(
        command: tuple[str, ...], **options: object
    ) -> _BoundedCommandResult:
        calls.append(command)
        return _BoundedCommandResult(
            1,
            False,
            (
                "allowed quantization types\n15 or Q4_K_M\n17 or Q5_K_M\n"
                "18 or Q6_K\n7 or Q8_0\n1 or F16\n32 or BF16"
            ),
            "",
            False,
            False,
        )

    monkeypatch.setattr(tools, "_run_bounded", fake_help)
    return quantizer, calls


def test_discovers_sibling_version_probe_and_supported_recipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quantizer, calls = _install_discovery(monkeypatch, tmp_path)

    tool = discover_llama_cpp_quantizer(
        LlamaCppQuantizationConfig(
            executable=quantizer,
            recipe=LlamaCppQuantizationRecipe.Q4_K_M,
        )
    )

    assert tool.quantizer_executable == quantizer
    assert tool.version_tool.reported_revision == "95b8e33e1"
    assert len(tool.quantizer_sha256) == 64
    assert calls == [(str(quantizer), "--help")]
    assert json.loads(json.dumps(tool.to_record()))["help_returncode"] == 1


def test_missing_tool_and_sibling_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        tools,
        "_resolve_executable",
        lambda value, revision: (_ for _ in ()).throw(
            LlamaCppValidationError("'llama-quantize' was not found on PATH")
        ),
    )
    with pytest.raises(LlamaCppQuantizationToolError, match="quantizer is unavailable"):
        discover_llama_cpp_quantizer()

    quantizer = tmp_path / "llama-quantize"
    quantizer.write_bytes(b"tool")
    monkeypatch.setattr(tools, "_resolve_executable", lambda value, revision: quantizer)
    with pytest.raises(LlamaCppQuantizationToolError, match="required sibling"):
        discover_llama_cpp_quantizer()


def test_unsupported_revision_is_rejected_before_help_or_quantization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quantizer, _ = _binaries(tmp_path)
    monkeypatch.setattr(tools, "_resolve_executable", lambda value, revision: quantizer)
    monkeypatch.setattr(
        tools,
        "_probe_tool",
        lambda config: (_ for _ in ()).throw(
            LlamaCppValidationError("unsupported llama.cpp revision deadbee")
        ),
    )
    monkeypatch.setattr(
        tools,
        "_run_bounded",
        lambda *args, **kwargs: pytest.fail("help or quantization must not execute"),
    )

    with pytest.raises(LlamaCppQuantizationToolError, match="revision validation"):
        discover_llama_cpp_quantizer()


def test_quantization_captures_invocation_and_publishes_validated_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quantizer, _ = _install_discovery(monkeypatch, tmp_path)
    source = tmp_path / "source.gguf"
    destination = tmp_path / "output.gguf"
    source.write_bytes(b"GGUF-source")
    calls: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...], **options: object
    ) -> _BoundedCommandResult:
        calls.append(command)
        if "--help" in command:
            return _BoundedCommandResult(
                1, False, "allowed quantization types\n15 or Q4_K_M", "", False, False
            )
        Path(command[-3]).write_bytes(b"GGUF-output")
        return _BoundedCommandResult(0, False, "quantized", "details", False, False)

    monkeypatch.setattr(tools, "_run_bounded", fake_run)
    monkeypatch.setattr(
        tools,
        "_validated_output",
        lambda path: (path.stat().st_size, "a" * 64),
    )
    config = LlamaCppQuantizationConfig(
        executable=quantizer,
        recipe=LlamaCppQuantizationRecipe.Q4_K_M,
        threads=4,
    )

    report = quantize_gguf_with_llama_cpp(source, destination, config=config)

    assert report.successful
    assert destination.read_bytes() == b"GGUF-output"
    assert report.command == calls[1]
    assert report.command[-2:] == ("Q4_K_M", "4")
    assert report.stdout == "quantized"
    assert report.stderr == "details"
    assert report.output_sha256 == "a" * 64


def test_failed_quantization_preserves_logs_and_removes_owned_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quantizer, _ = _install_discovery(monkeypatch, tmp_path)
    source = tmp_path / "source.gguf"
    destination = tmp_path / "output.gguf"
    source.write_bytes(b"GGUF-source")

    def fake_run(
        command: tuple[str, ...], **options: object
    ) -> _BoundedCommandResult:
        if "--help" in command:
            return _BoundedCommandResult(
                1, False, "allowed quantization types\n15 or Q4_K_M", "", False, False
            )
        Path(command[-3]).write_bytes(b"partial")
        return _BoundedCommandResult(
            2, False, "partial stdout", "failure detail", True, False
        )

    monkeypatch.setattr(tools, "_run_bounded", fake_run)
    report = quantize_gguf_with_llama_cpp(
        source,
        destination,
        config=LlamaCppQuantizationConfig(executable=quantizer),
    )

    assert not report.successful
    assert report.failure_reason == "llama-quantize exited with status 2"
    assert report.stdout == "partial stdout"
    assert report.stderr == "failure detail"
    assert report.stdout_truncated
    assert not destination.exists()
    assert not tuple(tmp_path.glob("*.partial"))


def test_existing_destination_and_requantization_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quantizer, _ = _install_discovery(monkeypatch, tmp_path)
    source = tmp_path / "source.gguf"
    destination = tmp_path / "output.gguf"
    source.write_bytes(b"GGUF-source")
    destination.write_bytes(b"owned")
    with pytest.raises(LlamaCppQuantizationToolError, match="will not be overwritten"):
        quantize_gguf_with_llama_cpp(source, destination)

    config = LlamaCppQuantizationConfig(executable=quantizer, allow_requantize=True)
    tool = discover_llama_cpp_quantizer(config)
    command = tools._quantization_command(
        tool, source, tmp_path / "partial.gguf", config
    )
    assert command[1] == "--allow-requantize"
