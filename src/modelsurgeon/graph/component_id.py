"""Stable, validated component identifiers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from functools import total_ordering
from typing import Any

_SIMPLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_INDEX = re.compile(r"^(0|[1-9][0-9]*)$")
_UPPER_HEX = frozenset("0123456789ABCDEF")
_UNRESERVED_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


def _validate_name(value: str) -> None:
    if not value:
        raise ValueError("component names cannot be empty")
    for character in value:
        if character == "\x00" or unicodedata.category(character) == "Cc":
            raise ValueError("component names cannot contain NUL or control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("component names must contain valid Unicode scalar values") from exc


def _encode_name(value: str) -> str:
    if _SIMPLE_NAME.fullmatch(value):
        return value
    encoded = "".join(
        chr(byte) if byte in _UNRESERVED_BYTES else f"%{byte:02X}"
        for byte in value.encode("utf-8")
    )
    return f"~{encoded}"


def _decode_escaped_name(raw: str) -> str:
    body = raw[1:]
    if not body:
        raise ValueError("escaped component name cannot be empty")

    decoded = bytearray()
    position = 0
    while position < len(body):
        character = body[position]
        if ord(character) in _UNRESERVED_BYTES:
            decoded.append(ord(character))
            position += 1
            continue
        if character != "%":
            raise ValueError(
                "escaped component names allow only ASCII letters, digits, '_', '-', "
                "and uppercase percent escapes"
            )
        if position + 2 >= len(body):
            raise ValueError("incomplete percent escape in component name")
        escape = body[position + 1 : position + 3]
        if any(digit not in _UPPER_HEX for digit in escape):
            raise ValueError("percent escapes must contain two uppercase hexadecimal digits")
        byte = int(escape, 16)
        if byte in _UNRESERVED_BYTES:
            raise ValueError("unreserved bytes must not be percent-encoded")
        decoded.append(byte)
        position += 3

    try:
        value = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("escaped component name is not valid UTF-8") from exc
    _validate_name(value)
    if _SIMPLE_NAME.fullmatch(value):
        raise ValueError(f"escaped component name {raw!r} has the simpler canonical form {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ComponentSegment:
    """One canonical name or non-negative index in a component path."""

    value: str | int

    def __post_init__(self) -> None:
        if isinstance(self.value, int):
            if self.value < 0:
                raise ValueError("component indices must be non-negative")
        else:
            _validate_name(self.value)

    def __str__(self) -> str:
        if isinstance(self.value, int):
            return str(self.value)
        return _encode_name(self.value)


@total_ordering
@dataclass(frozen=True, slots=True)
class ComponentId:
    """An immutable component path used across the full experiment lifecycle."""

    segments: tuple[ComponentSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a component ID cannot be empty")
        if self.segments[0].value != "model":
            raise ValueError("a component ID must use the canonical 'model' root")

    @classmethod
    def parse(cls, value: str) -> ComponentId:
        """Parse a strict dotted component ID into typed segments."""
        if not value or value.startswith(".") or value.endswith(".") or ".." in value:
            raise ValueError(f"invalid component ID: {value!r}")
        segments: list[ComponentSegment] = []
        for position, raw in enumerate(value.split(".")):
            try:
                if _INDEX.fullmatch(raw):
                    parsed: str | int = int(raw)
                elif raw.isdigit():
                    raise ValueError("numeric indices cannot contain leading zeros")
                elif _SIMPLE_NAME.fullmatch(raw):
                    parsed = raw
                elif raw.startswith("~"):
                    parsed = _decode_escaped_name(raw)
                else:
                    raise ValueError(f"invalid component segment syntax: {raw!r}")
                segments.append(ComponentSegment(parsed))
            except ValueError as exc:
                raise ValueError(f"invalid component segment {position} ({raw!r}): {exc}") from exc
        return cls(tuple(segments))

    @classmethod
    def from_module_name(cls, value: str) -> ComponentId:
        """Build an ID from a framework module name, using `model` for the root."""
        if not value:
            return cls.parse("model")
        if value == "model" or value.startswith("model."):
            return cls.parse(value)
        return cls.parse(f"model.{value}")

    @classmethod
    def from_json(cls, value: Any) -> ComponentId:
        """Parse the canonical JSON scalar representation."""
        if not isinstance(value, str):
            raise TypeError("a component ID JSON value must be a string")
        return cls.parse(value)

    def to_json(self) -> str:
        """Return the canonical JSON scalar representation."""
        return str(self)

    @property
    def parent(self) -> ComponentId | None:
        """Return the parent component, or None for a root ID."""
        if len(self.segments) == 1:
            return None
        return ComponentId(self.segments[:-1])

    def child(self, value: str | int) -> ComponentId:
        """Return a new ID with one validated child segment."""
        return ComponentId((*self.segments, ComponentSegment(value)))

    def __iter__(self) -> Iterator[ComponentSegment]:
        return iter(self.segments)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ComponentId):
            return NotImplemented
        return str(self).encode("utf-8") < str(other).encode("utf-8")

    def __str__(self) -> str:
        return ".".join(str(segment) for segment in self.segments)
