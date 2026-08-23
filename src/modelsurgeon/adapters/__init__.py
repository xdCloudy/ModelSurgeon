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

__all__ = [
    "AdapterCapability",
    "AdapterIdentity",
    "ComponentDescriptor",
    "ModelAdapter",
    "ModelFormat",
    "ModelSession",
    "ModelSource",
    "MutationSupport",
    "OpenOptions",
    "TensorChunk",
    "TensorDescriptor",
    "UnsupportedCapabilityError",
    "require_capability",
]

