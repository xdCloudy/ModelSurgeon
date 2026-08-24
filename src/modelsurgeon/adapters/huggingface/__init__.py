"""Hugging Face model adapter."""

from modelsurgeon.adapters.huggingface.discovery import (
    HuggingFaceComponentKind,
    HuggingFaceDiscovery,
    HuggingFaceModelProvider,
    ParameterReconciliationError,
    TransformerShape,
    discover_huggingface_components,
)
from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDependencyError,
    HuggingFaceDType,
    HuggingFaceLoadError,
    HuggingFaceLoadProvenance,
    HuggingFaceLoadRequest,
    HuggingFaceLoadResult,
    HuggingFaceModelError,
    HuggingFaceRevisionError,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_attention import (
    HF_PHYSICAL_ATTENTION_SCHEMA_VERSION,
    HuggingFaceAttentionRemovalResult,
    HuggingFacePhysicalAttentionError,
    remove_huggingface_attention_heads,
)
from modelsurgeon.adapters.huggingface.physical_mlp import (
    HF_PHYSICAL_MLP_SCHEMA_VERSION,
    HuggingFaceMLPLayerResize,
    HuggingFaceMLPRemovalResult,
    HuggingFacePhysicalMLPError,
    remove_huggingface_mlp_channels,
)

__all__ = [
    "HF_PHYSICAL_ATTENTION_SCHEMA_VERSION",
    "HF_PHYSICAL_MLP_SCHEMA_VERSION",
    "HuggingFaceAttentionRemovalResult",
    "HuggingFaceComponentKind",
    "HuggingFaceDType",
    "HuggingFaceDependencyError",
    "HuggingFaceDiscovery",
    "HuggingFaceLoadError",
    "HuggingFaceLoadProvenance",
    "HuggingFaceLoadRequest",
    "HuggingFaceLoadResult",
    "HuggingFaceMLPLayerResize",
    "HuggingFaceMLPRemovalResult",
    "HuggingFaceModelError",
    "HuggingFaceModelProvider",
    "HuggingFacePhysicalAttentionError",
    "HuggingFacePhysicalMLPError",
    "HuggingFaceRevisionError",
    "ParameterReconciliationError",
    "TransformerShape",
    "discover_huggingface_components",
    "load_causal_lm",
    "remove_huggingface_attention_heads",
    "remove_huggingface_mlp_channels",
]
