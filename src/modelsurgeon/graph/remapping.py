"""Composable, fail-closed component identity remapping after surgery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.graph.component_id import ComponentId


class IdentityRemapError(ValueError):
    """Base error for invalid, incomplete, or ambiguous identity mappings."""


class UnknownSourceIdentityError(IdentityRemapError):
    """Raised instead of silently treating an unrecorded source as retained."""


class RemovedSourceIdentityError(IdentityRemapError):
    """Raised when callers try to resolve a component explicitly removed."""


class IdentityDisposition(StrEnum):
    RETAINED = "retained"
    REMOVED = "removed"
    RENUMBERED = "renumbered"
    SPLIT = "split"
    MERGED = "merged"
    SPLIT_MERGED = "split_merged"


@dataclass(frozen=True, slots=True)
class ComponentIdentityMapping:
    source: ComponentId
    targets: tuple[ComponentId, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.targets != tuple(sorted(set(self.targets))):
            raise IdentityRemapError("identity mapping targets must be unique and canonical")
        if not self.reason:
            raise IdentityRemapError("identity mappings require a reason")

    @property
    def removed(self) -> bool:
        return not self.targets

    def to_record(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "targets": [str(target) for target in self.targets],
            "removed": self.removed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ComponentIdentityRemap:
    """Complete explicit mapping for one surgery stage or composed sequence."""

    mappings: tuple[ComponentIdentityMapping, ...]

    def __post_init__(self) -> None:
        sources = tuple(mapping.source for mapping in self.mappings)
        if not sources or sources != tuple(sorted(sources)):
            raise IdentityRemapError("identity mapping sources must be non-empty and canonical")
        if len(sources) != len(set(sources)):
            raise IdentityRemapError("identity mapping sources must be unique")

    @classmethod
    def build(
        cls, mappings: tuple[ComponentIdentityMapping, ...]
    ) -> ComponentIdentityRemap:
        return cls(tuple(sorted(mappings, key=lambda item: item.source)))

    @classmethod
    def retained(
        cls, sources: tuple[ComponentId, ...], *, reason: str = "retained"
    ) -> ComponentIdentityRemap:
        return cls.build(
            tuple(ComponentIdentityMapping(source, (source,), reason) for source in sources)
        )

    def _by_source(self) -> dict[ComponentId, ComponentIdentityMapping]:
        return {mapping.source: mapping for mapping in self.mappings}

    def _target_counts(self) -> dict[ComponentId, int]:
        counts: dict[ComponentId, int] = {}
        for mapping in self.mappings:
            for target in mapping.targets:
                counts[target] = counts.get(target, 0) + 1
        return counts

    def mapping(self, source: ComponentId) -> ComponentIdentityMapping:
        try:
            return self._by_source()[source]
        except KeyError as error:
            raise UnknownSourceIdentityError(
                f"component {source} has no explicit post-surgery identity mapping"
            ) from error

    def resolve(self, source: ComponentId) -> tuple[ComponentId, ...]:
        mapping = self.mapping(source)
        if mapping.removed:
            raise RemovedSourceIdentityError(
                f"component {source} was removed and has no post-surgery identity"
            )
        return mapping.targets

    def disposition(self, source: ComponentId) -> IdentityDisposition:
        mapping = self.mapping(source)
        if mapping.removed:
            return IdentityDisposition.REMOVED
        merged = any(self._target_counts()[target] > 1 for target in mapping.targets)
        if len(mapping.targets) > 1:
            return (
                IdentityDisposition.SPLIT_MERGED if merged else IdentityDisposition.SPLIT
            )
        if merged:
            return IdentityDisposition.MERGED
        if mapping.targets == (source,):
            return IdentityDisposition.RETAINED
        return IdentityDisposition.RENUMBERED

    def compose(self, next_remap: ComponentIdentityRemap) -> ComponentIdentityRemap:
        """Compose old→middle with middle→new without implicit retained IDs."""

        next_by_source = next_remap._by_source()
        composed: list[ComponentIdentityMapping] = []
        for mapping in self.mappings:
            if mapping.removed:
                composed.append(mapping)
                continue
            targets: set[ComponentId] = set()
            reasons = [mapping.reason]
            for middle in mapping.targets:
                try:
                    following = next_by_source[middle]
                except KeyError as error:
                    raise UnknownSourceIdentityError(
                        f"sequential remap has no explicit mapping for intermediate {middle}"
                    ) from error
                targets.update(following.targets)
                reasons.append(following.reason)
            composed.append(
                ComponentIdentityMapping(
                    mapping.source,
                    tuple(sorted(targets)),
                    " -> ".join(dict.fromkeys(reasons)),
                )
            )
        return ComponentIdentityRemap.build(tuple(composed))

    def to_record(self) -> dict[str, object]:
        return {
            "mappings": [
                {
                    **mapping.to_record(),
                    "disposition": self.disposition(mapping.source).value,
                }
                for mapping in self.mappings
            ]
        }
