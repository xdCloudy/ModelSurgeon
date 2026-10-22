"""Strict JSON parsing helpers shared by persisted and external records."""

from __future__ import annotations

import json
from json import JSONDecodeError, JSONDecoder
from typing import Any, NoReturn


def _reject_nonstandard_constant(token: str) -> NoReturn:
    raise JSONDecodeError(
        f"non-finite or non-standard numeric constant {token!r} is not permitted",
        token,
        0,
    )


def strict_json_loads(payload: str | bytes | bytearray, **kwargs: Any) -> Any:
    """Parse JSON while rejecting the non-standard NaN and Infinity tokens."""
    if "parse_constant" in kwargs:
        raise TypeError("strict_json_loads controls the parse_constant callback")
    return json.loads(payload, parse_constant=_reject_nonstandard_constant, **kwargs)


def strict_json_decoder(**kwargs: Any) -> JSONDecoder:
    """Create a decoder that rejects non-standard numeric constants."""
    if "parse_constant" in kwargs:
        raise TypeError("strict_json_decoder controls the parse_constant callback")
    return JSONDecoder(parse_constant=_reject_nonstandard_constant, **kwargs)
