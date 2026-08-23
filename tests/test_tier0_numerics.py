"""Tests for deterministic Tier 0 NaN/Inf and output sanity validation."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.evaluation.numerics import (
    NumericSurfaceKind,
    SequenceNumericSurface,
    Tier0NumericsConfig,
    deterministic_sample_indices,
    validate_tier0_numerics,
)


def test_finite_parameter_activation_logits_and_loss_pass() -> None:
    surfaces = (
        SequenceNumericSurface(
            NumericSurfaceKind.PARAMETER,
            "model.layers.0.mlp",
            (1.0, 2.0),
        ),
        SequenceNumericSurface(
            NumericSurfaceKind.ACTIVATION,
            "model.layers.0.mlp.output",
            (0.0, 3.0),
        ),
        SequenceNumericSurface(NumericSurfaceKind.LOGITS, "logits", (-1.0, 1.0)),
        SequenceNumericSurface(NumericSurfaceKind.LOSS, "loss", (1.25,)),
    )
    result = validate_tier0_numerics(surfaces)
    assert result.passed
    assert result.failure is None
    assert result.inspected_surfaces == 4
    assert result.inspected_values == 7


@pytest.mark.parametrize(
    "kind,identity,value,expected",
    [
        (NumericSurfaceKind.PARAMETER, "model.layers.2.mlp", math.nan, "nan"),
        (NumericSurfaceKind.ACTIVATION, "model.layers.3.self_attn.output", math.inf, "+inf"),
        (NumericSurfaceKind.LOGITS, "logits", -math.inf, "-inf"),
        (NumericSurfaceKind.LOSS, "loss", math.nan, "nan"),
    ],
)
def test_first_nonfinite_report_identifies_surface_and_value_class(
    kind: NumericSurfaceKind,
    identity: str,
    value: float,
    expected: str,
) -> None:
    result = validate_tier0_numerics(
        (SequenceNumericSurface(kind, identity, (1.0, value, 2.0)),)
    )
    assert not result.passed
    assert result.failure is not None
    assert result.failure.kind is kind
    assert result.failure.identity == identity
    assert result.failure.sampled_index == 1
    assert result.failure.value_class == expected


def test_large_surface_sampling_is_deterministic_bounded_and_includes_endpoints() -> None:
    class LargeSurface:
        kind = NumericSurfaceKind.PARAMETER
        identity = "model.embed_tokens"

        def __init__(self) -> None:
            self.reads: list[int] = []

        def value_count(self) -> int:
            return 1_000_000

        def value_at(self, index: int) -> float:
            self.reads.append(index)
            return 1.0

    first = LargeSurface()
    second = LargeSurface()
    config = Tier0NumericsConfig(max_values_per_surface=17)
    assert validate_tier0_numerics((first,), config).passed
    assert validate_tier0_numerics((second,), config).passed
    assert first.reads == second.reads
    assert len(first.reads) == 17
    assert first.reads[0] == 0
    assert first.reads[-1] == 999_999
    assert first.reads == sorted(set(first.reads))


def test_first_affected_surface_stops_later_reads() -> None:
    good = SequenceNumericSurface(NumericSurfaceKind.PARAMETER, "first", (1.0, 2.0))
    bad = SequenceNumericSurface(NumericSurfaceKind.ACTIVATION, "second", (math.inf,))

    class NeverRead:
        kind = NumericSurfaceKind.LOGITS
        identity = "third"

        def value_count(self) -> int:
            return 1

        def value_at(self, index: int) -> float:
            del index
            raise AssertionError("later surface should not be inspected")

    result = validate_tier0_numerics((good, bad, NeverRead()))
    assert not result.passed
    assert result.failure is not None and result.failure.identity == "second"
    assert result.inspected_surfaces == 2


def test_sampling_and_surface_contracts_fail_early() -> None:
    with pytest.raises(ValueError, match="positive"):
        deterministic_sample_indices(0, 1)
    with pytest.raises(ValueError, match="positive"):
        Tier0NumericsConfig(0)
    with pytest.raises(ValueError, match="at least one"):
        validate_tier0_numerics(())
