"""Small, shared primitives for crossing untrusted conversational boundaries.

The provider and tool layers are intentionally capability-isolated rather than
privileged execution environments.  This module keeps that distinction
explicit: values crossing a boundary are copied, secret-bearing diagnostics
are scrubbed, and untrusted metadata is never promoted by this layer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum


class TrustZone(StrEnum):
    """The trust zone assigned to a value at a conversational boundary."""

    TRUSTED_ENGINE = "trusted_engine"
    UNTRUSTED_PROVIDER = "untrusted_provider"
    UNTRUSTED_TOOL = "untrusted_tool"


class IsolationFailure(RuntimeError):
    """Raised when a boundary cannot preserve trusted state."""


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization|bearer|credential|"
    r"password|secret|token)\b\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{4,}")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://[^/:\s]+:)[^@/\s]+@")
_PEM = re.compile(
    r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.DOTALL
)
_SECRET_KEY = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|bearer|credential|"
    r"password|secret|token)"
)


def redact_secret_text(value: str, *, secrets: Sequence[str] = ()) -> str:
    """Redact common credential forms and caller-supplied secret values."""

    result = value
    for secret in sorted(
        {item for item in secrets if isinstance(item, str) and len(item) >= 4},
        key=len,
        reverse=True,
    ):
        result = result.replace(secret, "<redacted>")
    result = _PEM.sub("<redacted>", result)
    result = _URL_CREDENTIALS.sub(r"\1<redacted>@", result)
    result = _SECRET_ASSIGNMENT.sub(r"\1<redacted>", result)
    return _BEARER.sub("Bearer <redacted>", result)


def redact_untrusted_value(value: object, *, secrets: Sequence[str] = ()) -> object:
    """Return a JSON-shaped value safe for diagnostics and retained raw data."""

    if isinstance(value, str):
        return redact_secret_text(value, secrets=secrets)
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if isinstance(key, str) and _SECRET_KEY.search(key)
                else redact_untrusted_value(child, secrets=secrets)
            )
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_untrusted_value(child, secrets=secrets) for child in value]
    return value


def copy_untrusted_json(value: object) -> object:
    """Copy JSON-compatible provider/tool input without sharing mutable state."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise IsolationFailure("untrusted metadata contains a non-text key")
        return {key: copy_untrusted_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [copy_untrusted_json(child) for child in value]
    if isinstance(value, tuple):
        return tuple(copy_untrusted_json(child) for child in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise IsolationFailure("untrusted metadata is not JSON-compatible")


__all__ = [
    "IsolationFailure",
    "TrustZone",
    "copy_untrusted_json",
    "redact_secret_text",
    "redact_untrusted_value",
]
