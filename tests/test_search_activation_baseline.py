"""Pure ranking tests for the physical-search activation baseline."""

from __future__ import annotations

import math

from modelsurgeon.first_party_optimize_runtime import _activation_magnitude_order


def test_activation_magnitude_order_uses_lowest_finite_channel_signal() -> None:
    order = _activation_magnitude_order(
        ("channel-b", "channel-a", "channel-c", "missing"),
        {
            "channel-b": (
                {"name": "activation_abs_mean", "value": 0.2},
                {"name": "activation_abs_mean", "value": 0.4},
            ),
            "channel-a": ({"name": "activation_abs_mean", "value": 0.1},),
            "channel-c": (
                {"name": "activation_abs_mean", "value": math.nan},
                {"name": "activation_abs_mean", "value": True},
                {"name": "activation_rms", "value": 0.0},
            ),
        },
    )

    assert order == ("channel-a", "channel-b")
