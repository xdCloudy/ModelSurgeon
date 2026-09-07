"""Versioned provider capability policy for contract conformance tests.

The matrix describes the contract boundary, not model quality.  A provider
may be unavailable or intentionally unsupported for a cell, but that state is
explicit and must never be interpreted as a successful capability claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.provider_kind import ProviderKind

PROVIDER_CONFORMANCE_SCHEMA_VERSION = 1


class ProviderConformanceCapability(StrEnum):
    STRUCTURED_OUTPUT = "structured_output"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    TOKEN_BUDGET = "token_budget"
    RESOURCE_BUDGET = "resource_budget"
    PROVENANCE = "provenance"


class ProviderConformanceStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class ProviderConformanceError(ValueError):
    """Raised when a conformance matrix is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class ProviderConformanceCell:
    """One explicit provider-kind and contract-capability expectation."""

    kind: ProviderKind
    capability: ProviderConformanceCapability
    status: ProviderConformanceStatus
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ProviderConformanceError("conformance cell rationale must be non-empty")
        if (
            self.status is ProviderConformanceStatus.UNSUPPORTED
            and "unsupported" not in self.rationale.lower()
        ):
            raise ProviderConformanceError(
                "unsupported conformance cells require an explicit unsupported rationale"
            )

    def to_record(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "capability": self.capability.value,
            "status": self.status.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ProviderConformanceMatrix:
    """Complete, deterministic policy for the provider conformance suite."""

    cells: tuple[ProviderConformanceCell, ...]
    schema_version: int = PROVIDER_CONFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected = set(product(ProviderKind, ProviderConformanceCapability))
        actual = {(cell.kind, cell.capability) for cell in self.cells}
        if len(actual) != len(self.cells) or actual != expected:
            raise ProviderConformanceError(
                "provider conformance matrix must contain every unique kind/capability cell"
            )
        if self.schema_version != PROVIDER_CONFORMANCE_SCHEMA_VERSION:
            raise ProviderConformanceError("unsupported provider conformance schema version")

    @property
    def matrix_id(self) -> str:
        encoded = canonical_identity_json(self.to_record()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_record(self) -> dict[str, object]:
        ordered = sorted(self.cells, key=lambda cell: (cell.kind.value, cell.capability.value))
        return {
            "schema_version": self.schema_version,
            "cells": [cell.to_record() for cell in ordered],
        }

    def markdown_table(self) -> str:
        """Render the public policy table without hiding unsupported cells."""

        indexed = {(cell.kind, cell.capability): cell for cell in self.cells}
        symbols = {
            ProviderConformanceStatus.SUPPORTED: "V",
            ProviderConformanceStatus.UNSUPPORTED: "—",
        }
        columns = tuple(ProviderKind)
        lines = [
            "| Capability | " + " | ".join(kind.value for kind in columns) + " |",
            "| --- | " + " | ".join("---" for _ in columns) + " |",
        ]
        for capability in ProviderConformanceCapability:
            values = [symbols[indexed[(kind, capability)].status] for kind in columns]
            lines.append(f"| {capability.value} | " + " | ".join(values) + " |")
        return "\n".join(lines)


def _cell(
    kind: ProviderKind,
    capability: ProviderConformanceCapability,
    status: ProviderConformanceStatus,
    rationale: str,
) -> ProviderConformanceCell:
    return ProviderConformanceCell(kind, capability, status, rationale)


def _build_matrix() -> ProviderConformanceMatrix:
    cells: list[ProviderConformanceCell] = []
    for kind in (
        ProviderKind.LOCAL,
        ProviderKind.COMPATIBLE_ENDPOINT,
        ProviderKind.HOSTED,
    ):
        for capability in ProviderConformanceCapability:
            cells.append(
                _cell(
                    kind,
                    capability,
                    ProviderConformanceStatus.SUPPORTED,
                    "deterministic fixture exercises the shared contract boundary",
                )
            )
    cells.extend(
        (
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.STRUCTURED_OUTPUT,
                ProviderConformanceStatus.UNSUPPORTED,
                "unsupported because no-LLM mode has no text-model capability",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.REFUSAL,
                ProviderConformanceStatus.SUPPORTED,
                "explicit unsupported result is retained as a refusal",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.TIMEOUT,
                ProviderConformanceStatus.UNSUPPORTED,
                "unsupported because no request is executed in no-LLM mode",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.CANCELLATION,
                ProviderConformanceStatus.SUPPORTED,
                "pre-cancelled requests return an explicit cancellation outcome",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.TOKEN_BUDGET,
                ProviderConformanceStatus.UNSUPPORTED,
                "unsupported because no-LLM mode advertises no request capability",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.RESOURCE_BUDGET,
                ProviderConformanceStatus.UNSUPPORTED,
                "unsupported because no-LLM mode advertises no request capability",
            ),
            _cell(
                ProviderKind.NONE,
                ProviderConformanceCapability.PROVENANCE,
                ProviderConformanceStatus.SUPPORTED,
                "explicit refusal retains deterministic provider provenance",
            ),
        )
    )
    return ProviderConformanceMatrix(tuple(cells))


PROVIDER_CONFORMANCE_MATRIX = _build_matrix()


__all__ = [
    "PROVIDER_CONFORMANCE_MATRIX",
    "PROVIDER_CONFORMANCE_SCHEMA_VERSION",
    "ProviderConformanceCapability",
    "ProviderConformanceCell",
    "ProviderConformanceError",
    "ProviderConformanceMatrix",
    "ProviderConformanceStatus",
]
