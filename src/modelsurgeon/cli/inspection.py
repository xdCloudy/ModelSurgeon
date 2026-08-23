"""Stable, framework-neutral records for the model inspection command."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from modelsurgeon.adapters import ArchitectureEvidence, FamilySelection, detect_model_family
from modelsurgeon.adapters.huggingface import (
    HuggingFaceDiscovery,
    HuggingFaceLoadProvenance,
    HuggingFaceLoadRequest,
    discover_huggingface_components,
    load_causal_lm,
)


@dataclass(frozen=True, slots=True)
class ModelInspection:
    """Loaded model identity and bounded discovery metadata."""

    provenance: HuggingFaceLoadProvenance
    family: FamilySelection
    discovery: HuggingFaceDiscovery

    def records(self) -> Iterator[dict[str, object]]:
        yield {
            "record_type": "model",
            "source": self.provenance.source,
            "requested_revision": self.provenance.requested_revision,
            "resolved_revision": self.provenance.resolved_revision,
            "family": self.family.family.value,
            "family_evidence": list(self.family.matched_evidence),
            **self.discovery.to_record(),
            "loader_options": self.provenance.to_record()["loader_options"],
        }
        for component in self.discovery.components():
            yield {"record_type": "component", **component.to_record()}


def _architecture_evidence(config: object) -> ArchitectureEvidence:
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str):
        model_type = None
    raw_architectures = getattr(config, "architectures", ())
    if not isinstance(raw_architectures, list | tuple):
        raw_architectures = ()
    architecture_names = tuple(
        architecture for architecture in raw_architectures if isinstance(architecture, str)
    )
    return ArchitectureEvidence(
        model_type=model_type,
        architecture_names=architecture_names,
    )


def inspect_huggingface_model(request: HuggingFaceLoadRequest) -> ModelInspection:
    """Load, resolve, and discover one model for stable CLI rendering."""
    loaded = load_causal_lm(request)
    config = getattr(loaded.model, "config", None)
    family = detect_model_family(_architecture_evidence(config))
    discovery = discover_huggingface_components(loaded.model, family.family)
    return ModelInspection(loaded.provenance, family, discovery)
