"""Strict JSON parsing helpers shared by persisted and external records."""

from __future__ import annotations

import json
import math
from json import JSONDecodeError, JSONDecoder
from typing import Any, NoReturn


def _reject_nonstandard_constant(token: str) -> NoReturn:
    raise JSONDecodeError(
        f"non-finite or non-standard numeric constant {token!r} is not permitted",
        token,
        0,
    )


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise JSONDecodeError(
            f"number {token!r} is outside the finite floating-point range",
            token,
            0,
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JSONDecodeError(f"duplicate object key {key!r}", key, 0)
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes | bytearray, **kwargs: Any) -> Any:
    """Parse unambiguous JSON while rejecting duplicate keys and non-finite numbers."""
    controlled = {"parse_constant", "parse_float", "object_pairs_hook", "cls"}.intersection(
        kwargs
    )
    if controlled:
        names = ", ".join(sorted(controlled))
        raise TypeError(f"strict_json_loads controls {names}")
    return json.loads(
        payload,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_keys,
        **kwargs,
    )


def strict_json_decoder(**kwargs: Any) -> JSONDecoder:
    """Create a decoder that rejects duplicate keys and non-finite numbers."""
    controlled = {"parse_constant", "parse_float", "object_pairs_hook", "cls"}.intersection(
        kwargs
    )
    if controlled:
        names = ", ".join(sorted(controlled))
        raise TypeError(f"strict_json_decoder controls {names}")
    return JSONDecoder(
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_keys,
        **kwargs,
    )
