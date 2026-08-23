"""Stable, validated component identifiers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_INDEX = re.compile(r"^(0|[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True, order=True)
class ComponentSegment:
    """One canonical name or non-negative index in a component path."""

    value: str | int

    def __post_init__(self) -> None:
        if isinstance(self.value, int):
            if self.value < 0:
                raise ValueError("component indices must be non-negative")
        elif not _NAME.fullmatch(self.value):
            raise ValueError(f"invalid component segment: {self.value!r}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class ComponentId:
    """An immutable component path used across the full experiment lifecycle."""

    segments: tuple[ComponentSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a component ID cannot be empty")

    @classmethod
    def parse(cls, value: str) -> ComponentId:
        """Parse a strict dotted component ID into typed segments."""
        if not value or value.startswith(".") or value.endswith(".") or ".." in value:
            raise ValueError(f"invalid component ID: {value!r}")
        segments: list[ComponentSegment] = []
        for raw in value.split("."):
            parsed: str | int = int(raw) if _INDEX.fullmatch(raw) else raw
            segments.append(ComponentSegment(parsed))
        return cls(tuple(segments))

    @classmethod
    def from_module_name(cls, value: str) -> ComponentId:
        """Build an ID from a framework module name, using `model` for the root."""
        return cls.parse(value or "model")

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

    def __str__(self) -> str:
        return ".".join(str(segment) for segment in self.segments)
