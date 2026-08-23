"""Deterministic, fail-closed model architecture family detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelFamily(StrEnum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    GEMMA = "gemma"


class ArchitectureDetectionError(ValueError):
    """Base error for absent or contradictory architecture evidence."""


class UnknownArchitectureError(ArchitectureDetectionError):
    """Raised when no explicit supported alias matches."""


class ConflictingArchitectureError(ArchitectureDetectionError):
    """Raised when explicit evidence points at more than one family."""


@dataclass(frozen=True, slots=True)
class ArchitectureEvidence:
    """Explicit architecture fields read from HF or GGUF metadata."""

    model_type: str | None = None
    architecture_names: tuple[str, ...] = ()
    gguf_architecture: str | None = None


@dataclass(frozen=True, slots=True)
class FamilySelection:
    """Selected family with the normalized evidence that justified it."""

    family: ModelFamily
    matched_evidence: tuple[str, ...]

    def to_record(self) -> dict[str, str | list[str]]:
        return {
            "family": self.family.value,
            "matched_evidence": list(self.matched_evidence),
        }


_MODEL_TYPE_ALIASES = {
    "llama": ModelFamily.LLAMA,
    "mistral": ModelFamily.MISTRAL,
    "qwen2": ModelFamily.QWEN,
    "qwen2_moe": ModelFamily.QWEN,
    "qwen3": ModelFamily.QWEN,
    "qwen3_moe": ModelFamily.QWEN,
    "gemma": ModelFamily.GEMMA,
    "gemma2": ModelFamily.GEMMA,
    "gemma3_text": ModelFamily.GEMMA,
}

_ARCHITECTURE_ALIASES = {
    "llamaforcausallm": ModelFamily.LLAMA,
    "mistralforcausallm": ModelFamily.MISTRAL,
    "qwen2forcausallm": ModelFamily.QWEN,
    "qwen2moeforcausallm": ModelFamily.QWEN,
    "qwen3forcausallm": ModelFamily.QWEN,
    "qwen3moeforcausallm": ModelFamily.QWEN,
    "gemmaforcausallm": ModelFamily.GEMMA,
    "gemma2forcausallm": ModelFamily.GEMMA,
    "gemma3forconditionalgeneration": ModelFamily.GEMMA,
    "gemma3fortextdecoding": ModelFamily.GEMMA,
}

_GGUF_ALIASES = {
    "llama": ModelFamily.LLAMA,
    "mistral": ModelFamily.MISTRAL,
    "qwen2": ModelFamily.QWEN,
    "qwen3": ModelFamily.QWEN,
    "gemma": ModelFamily.GEMMA,
    "gemma2": ModelFamily.GEMMA,
    "gemma3": ModelFamily.GEMMA,
}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def detect_model_family(evidence: ArchitectureEvidence) -> FamilySelection:
    """Select exactly one supported family from explicit metadata fields."""
    matches: list[tuple[str, ModelFamily]] = []
    if evidence.model_type:
        normalized = _normalize(evidence.model_type)
        family = _MODEL_TYPE_ALIASES.get(normalized)
        if family is not None:
            matches.append((f"model_type:{normalized}", family))
    for architecture_name in evidence.architecture_names:
        normalized = _normalize(architecture_name).replace("_", "")
        family = _ARCHITECTURE_ALIASES.get(normalized)
        if family is not None:
            matches.append((f"architecture:{normalized}", family))
    if evidence.gguf_architecture:
        normalized = _normalize(evidence.gguf_architecture)
        family = _GGUF_ALIASES.get(normalized)
        if family is not None:
            matches.append((f"gguf:{normalized}", family))

    families = {family for _, family in matches}
    if not families:
        supplied = {
            "model_type": evidence.model_type,
            "architecture_names": evidence.architecture_names,
            "gguf_architecture": evidence.gguf_architecture,
        }
        raise UnknownArchitectureError(
            f"no supported architecture alias matched explicit evidence: {supplied}"
        )
    if len(families) != 1:
        details = ", ".join(f"{source}={family.value}" for source, family in matches)
        raise ConflictingArchitectureError(f"architecture evidence conflicts: {details}")
    family = next(iter(families))
    return FamilySelection(
        family=family,
        matched_evidence=tuple(sorted(source for source, _ in matches)),
    )
