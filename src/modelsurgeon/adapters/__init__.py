"""Framework-neutral model format adapter contracts."""

from modelsurgeon.adapters.base import (
    AdapterCapability,
    AdapterIdentity,
    ComponentDescriptor,
    ModelAdapter,
    ModelFormat,
    ModelSession,
    ModelSource,
    MutationSupport,
    OpenOptions,
    TensorChunk,
    TensorDescriptor,
    UnsupportedCapabilityError,
    require_capability,
)
from modelsurgeon.adapters.family import (
    ArchitectureDetectionError,
    ArchitectureEvidence,
    ConflictingArchitectureError,
    FamilySelection,
    ModelFamily,
    UnknownArchitectureError,
    detect_model_family,
)

__all__ = [
    "AdapterCapability",
    "AdapterIdentity",
    "ArchitectureDetectionError",
    "ArchitectureEvidence",
    "ComponentDescriptor",
    "ConflictingArchitectureError",
    "FamilySelection",
    "ModelAdapter",
    "ModelFamily",
    "ModelFormat",
    "ModelSession",
    "ModelSource",
    "MutationSupport",
    "OpenOptions",
    "TensorChunk",
    "TensorDescriptor",
    "UnknownArchitectureError",
    "UnsupportedCapabilityError",
    "detect_model_family",
    "require_capability",
]

