"""Tests for calibration identities and deterministic selection."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.datasets import (
    CalibrationContract,
    CalibrationSample,
    DatasetIdentity,
    DatasetTrust,
    PreprocessingIdentity,
    SelectionConfig,
    TokenizerIdentity,
)

DIGEST = "a" * 64


def _contract(seed: int = 7) -> CalibrationContract:
    return CalibrationContract(
        DatasetIdentity(
            "org/data", "commit-1", "validation", "apache-2.0", DatasetTrust.COMMUNITY,
            "reviewed card and maintainers", (("language", "en"),),
        ),
        PreprocessingIdentity("normalize", "2", DIGEST),
        TokenizerIdentity("org/tokenizer", "commit-2", "b" * 64),
        SelectionConfig(seed, 3),
    )


def _samples() -> tuple[CalibrationSample, ...]:
    return tuple(CalibrationSample(f"sample-{index}", f"{index:064x}") for index in range(6))


def test_same_contract_and_seed_select_same_samples_independent_of_input_order() -> None:
    samples = _samples()
    first = _contract().select(samples)
    second = _contract().select(tuple(reversed(samples)))

    assert first == second
    assert [sample.sample_id for sample in first] == ["sample-1", "sample-3", "sample-4"]
    record = _contract().to_record(first)
    assert record["dataset"]["license"] == "apache-2.0"  # type: ignore[index]
    assert record["dataset"]["trust"] == "community"  # type: ignore[index]
    assert json.loads(json.dumps(record)) == record


def test_seed_and_preprocessing_identity_affect_selection() -> None:
    samples = _samples()
    assert _contract(7).select(samples) != _contract(8).select(samples)


def test_duplicate_or_insufficient_candidates_fail_closed() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="unique"):
        _contract().select((samples[0], samples[0], samples[1]))
    with pytest.raises(ValueError, match="exceeds"):
        _contract().select(samples[:2])


def test_invalid_digests_and_selection_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        CalibrationSample("sample", "not-a-digest")
    with pytest.raises(ValueError, match="count positive"):
        SelectionConfig(0, 0)
