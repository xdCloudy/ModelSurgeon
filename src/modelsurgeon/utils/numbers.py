"""Numeric validation helpers used by untrusted and public API boundaries."""

from __future__ import annotations

import math


def is_exact_int(value: object) -> bool:
    """Return whether a value is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: object) -> bool:
    """Return whether a value is a finite integer or float, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


__all__ = ["is_exact_int", "is_finite_number"]
