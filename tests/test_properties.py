"""Tests for deterministic mutation property campaigns and shrink evidence."""

from __future__ import annotations

import pytest

from modelsurgeon.properties import (
    PROPERTY_OPERATIONS,
    WRITABLE_NATIVE_GGUF_CODECS,
    PropertyOutcome,
    PropertySuiteError,
    generate_property_cases,
    run_property_suite,
)


def test_campaign_is_deterministic_and_covers_writable_codecs_and_operations() -> None:
    first = generate_property_cases(seed=407, cases=2048)
    second = generate_property_cases(seed=407, cases=2048)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    targets = {case.target for case in first}
    assert {f"gguf_codec:{codec}" for codec in WRITABLE_NATIVE_GGUF_CODECS} <= targets
    assert {f"operation:{operation}" for operation in PROPERTY_OPERATIONS} <= targets
    assert len(first) == 2048


def test_passing_suite_checks_structural_reconciliation() -> None:
    cases = generate_property_cases(seed=407, cases=64)

    def evaluator(case: object) -> None:
        dimensions = case.dimensions  # type: ignore[attr-defined]
        assert all(value > 0 for value in dimensions)
        assert sum(dimensions) >= len(dimensions)

    report = run_property_suite(cases, evaluator)
    report.require_clean()
    assert report.outcome is PropertyOutcome.PASS


def test_failure_keeps_seed_and_shrunk_regression_case() -> None:
    case = generate_property_cases(seed=407, cases=1)[0]
    counter = {"calls": 0}

    def evaluator(candidate: object) -> None:
        counter["calls"] += 1
        if max(candidate.dimensions) > 1:  # type: ignore[attr-defined]
            raise AssertionError("injected rollback invariant failure")

    report = run_property_suite((case,), evaluator)
    assert report.outcome is PropertyOutcome.FAIL
    assert report.failures[0].case.seed == case.seed
    assert max(report.failures[0].shrunk_case.dimensions) < max(case.dimensions)
    assert all(
        shrunk <= original
        for shrunk, original in zip(
            report.failures[0].shrunk_case.dimensions, case.dimensions, strict=True
        )
    )
    assert counter["calls"] > 1
    with pytest.raises(PropertySuiteError, match="fail"):
        report.require_clean()


def test_unsupported_and_unknown_outcomes_are_retained() -> None:
    cases = generate_property_cases(seed=407, cases=2)

    def evaluator(case: object) -> None:
        if case is cases[0]:
            raise NotImplementedError("codec unavailable")
        raise TimeoutError("bounded property timeout")

    report = run_property_suite(cases, evaluator)
    assert report.outcome is PropertyOutcome.UNKNOWN
    assert [item.outcome for item in report.results] == [
        PropertyOutcome.UNSUPPORTED,
        PropertyOutcome.UNKNOWN,
    ]
