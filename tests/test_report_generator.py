from __future__ import annotations

import json

import pytest

from modelsurgeon.explain import (
    ReportFailure,
    ReportGenerationError,
    ReportInput,
    ReportLink,
    ReportPlot,
    ReportRedaction,
    generate_report,
)


def _input(timestamp: str = "2026-08-24T12:00:00Z") -> ReportInput:
    return ReportInput(
        "search",
        "search_abc",
        timestamp,
        {
            "model": "HuggingFaceTB/SmolLM2-135M",
            "checkpoint": "C:\\private\\models\\source",
            "token": "must-not-leak",
            "nested": {"output": "C:\\private\\runs\\one"},
        },
        (
            {"generation": 1, "parent": "checkpoint_source", "child": "checkpoint_one"},
        ),
        (("latency_seconds", 0.1), ("perplexity", None)),
        (ReportPlot("quality", "generation", "loss", ((0, 2), (1, 1.5))),),
        (ReportFailure("evaluation", "timeout", "bounded timeout", True),),
        {"cpu": "test CPU", "ram_bytes": 1024},
        (
            ReportLink("candidate", "checkpoint", "checkpoint_one"),
            ReportLink("source", "checkpoint", "checkpoint_source"),
        ),
        ReportRedaction(("C:\\private",)),
    )


def test_report_is_self_contained_deterministic_redacted_and_json_aligned() -> None:
    first = generate_report(_input())
    second = generate_report(_input())
    assert first == second
    assert json.loads(first.json_text) == first.record
    assert first.record["resolved_config"]["token"] == "<redacted>"  # type: ignore[index]
    serialized = first.json_text + first.html_text
    assert "must-not-leak" not in serialized
    assert "C:\\private" not in serialized
    assert first.record["resolved_config"]["checkpoint"] == "<redacted-path>/models/source"  # type: ignore[index]
    assert "&lt;redacted-path&gt;/models/source" in first.html_text
    assert "<svg" in first.html_text
    assert "https://" not in first.html_text
    assert "checkpoint_source" in first.html_text
    assert first.json_sha256 == second.json_sha256
    assert first.html_sha256 == second.html_sha256


def test_declared_timestamp_is_the_only_implicit_time_and_invalid_values_fail() -> None:
    first = generate_report(_input("one"))
    second = generate_report(_input("two"))
    assert first.json_sha256 != second.json_sha256
    with pytest.raises(ReportGenerationError, match="non-finite"):
        generate_report(
            ReportInput(
                "run",
                "run_bad",
                None,
                {"bad": float("nan")},
                (),
                (),
                (),
                (),
                {},
                (),
            )
        )
