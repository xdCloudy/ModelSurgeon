"""Hugging Face model adapter."""

from modelsurgeon.adapters.huggingface.block_influence import (
    BlockInfluenceScore,
    HuggingFaceBlockInfluenceError,
    measure_huggingface_block_influence,
)
from modelsurgeon.adapters.huggingface.discovery import (
    HuggingFaceComponentKind,
    HuggingFaceDiscovery,
    HuggingFaceModelProvider,
    ParameterReconciliationError,
    TransformerShape,
    discover_huggingface_components,
)
from modelsurgeon.adapters.huggingface.layer_sensitivity import (
    HuggingFaceLayerSensitivityError,
    LayerSensitivityScore,
    measure_huggingface_layer_sensitivity,
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
from modelsurgeon.adapters.huggingface.low_rank import (
    HF_LOW_RANK_SCHEMA_VERSION,
    HuggingFaceLowRankError,
    LowRankReplacement,
    LowRankReplacementReport,
    replace_huggingface_linears_low_rank,
)
from modelsurgeon.adapters.huggingface.physical_attention import (
    HF_PHYSICAL_ATTENTION_SCHEMA_VERSION,
    HuggingFaceAttentionRemovalResult,
    HuggingFacePhysicalAttentionError,
    remove_huggingface_attention_heads,
)
from modelsurgeon.adapters.huggingface.physical_layers import (
    HF_PHYSICAL_LAYER_SCHEMA_VERSION,
    HuggingFaceLayerRemovalResult,
    HuggingFacePhysicalLayerError,
    remove_huggingface_transformer_layers,
)
from modelsurgeon.adapters.huggingface.physical_mlp import (
    HF_PHYSICAL_MLP_SCHEMA_VERSION,
    HuggingFaceMLPLayerResize,
    HuggingFaceMLPRemovalResult,
    HuggingFacePhysicalMLPError,
    remove_huggingface_mlp_channels,
)

__all__ = [
    "HF_LOW_RANK_SCHEMA_VERSION",
    "HF_PHYSICAL_ATTENTION_SCHEMA_VERSION",
    "HF_PHYSICAL_LAYER_SCHEMA_VERSION",
    "HF_PHYSICAL_MLP_SCHEMA_VERSION",
    "BlockInfluenceScore",
    "HuggingFaceAttentionRemovalResult",
    "HuggingFaceBlockInfluenceError",
    "HuggingFaceComponentKind",
    "HuggingFaceDType",
    "HuggingFaceDependencyError",
    "HuggingFaceDiscovery",
    "HuggingFaceLayerRemovalResult",
    "HuggingFaceLayerSensitivityError",
    "HuggingFaceLoadError",
    "HuggingFaceLoadProvenance",
    "HuggingFaceLoadRequest",
    "HuggingFaceLoadResult",
    "HuggingFaceLowRankError",
    "HuggingFaceMLPLayerResize",
    "HuggingFaceMLPRemovalResult",
    "HuggingFaceModelError",
    "HuggingFaceModelProvider",
    "HuggingFacePhysicalAttentionError",
    "HuggingFacePhysicalLayerError",
    "HuggingFacePhysicalMLPError",
    "HuggingFaceRevisionError",
    "LayerSensitivityScore",
    "LowRankReplacement",
    "LowRankReplacementReport",
    "ParameterReconciliationError",
    "TransformerShape",
    "discover_huggingface_components",
    "load_causal_lm",
    "measure_huggingface_block_influence",
    "measure_huggingface_layer_sensitivity",
    "remove_huggingface_attention_heads",
    "remove_huggingface_mlp_channels",
    "remove_huggingface_transformer_layers",
    "replace_huggingface_linears_low_rank",
]
