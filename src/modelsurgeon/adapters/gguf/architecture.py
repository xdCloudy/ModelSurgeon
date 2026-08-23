"""Versioned GGUF architecture, tensor-axis, and mutation mapping contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.graph import ComponentId


class GGUFArchitectureError(ValueError):
    """Base error for unsupported or ambiguous GGUF architecture metadata."""


class UnknownGGUFArchitectureError(GGUFArchitectureError):
    """Raised when no explicit versioned architecture contract matches."""


class AmbiguousGGUFArchitectureError(GGUFArchitectureError):
    """Raised when metadata needs an explicit family selection."""


class UnknownTensorMappingError(GGUFArchitectureError):
    """Raised when a tensor name has no declared semantic mapping."""


class TensorRole(StrEnum):
    TOKEN_EMBEDDING = "token_embedding"
    OUTPUT = "output"
    OUTPUT_NORM = "output_norm"
    INPUT_NORM = "input_norm"
    POST_ATTENTION_NORM = "post_attention_norm"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_O = "attention_o"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"


class AxisSemantic(StrEnum):
    VOCABULARY = "vocabulary"
    INPUT_FEATURE = "input_feature"
    OUTPUT_FEATURE = "output_feature"
    HIDDEN_FEATURE = "hidden_feature"
    ATTENTION_HEAD = "attention_head"
    KV_HEAD = "kv_head"
    MLP_CHANNEL = "mlp_channel"


class CouplingKind(StrEnum):
    ATTENTION_HEAD = "attention_head"
    KV_HEAD = "kv_head"
    MLP_CHANNEL = "mlp_channel"


class MetadataSemantic(StrEnum):
    BLOCK_COUNT = "block_count"
    EMBEDDING_LENGTH = "embedding_length"
    FEED_FORWARD_LENGTH = "feed_forward_length"
    HEAD_COUNT = "attention.head_count"
    KV_HEAD_COUNT = "attention.head_count_kv"
    EXPERT_COUNT = "expert_count"
    EXPERT_USED_COUNT = "expert_used_count"


@dataclass(frozen=True, slots=True)
class TensorAxis:
    index: int
    semantic: AxisSemantic

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tensor axis index must be non-negative")


@dataclass(frozen=True, slots=True)
class TensorMapping:
    tensor_name: str
    component_id: ComponentId
    role: TensorRole
    axes: tuple[TensorAxis, ...]


@dataclass(frozen=True, slots=True)
class TensorAxisTarget:
    tensor_name: str
    role: TensorRole
    axis: int


@dataclass(frozen=True, slots=True)
class CouplingGroup:
    group_id: str
    kind: CouplingKind
    targets: tuple[TensorAxisTarget, ...]

    def __post_init__(self) -> None:
        if not self.group_id or len(self.targets) < 2:
            raise ValueError("coupling groups require an ID and at least two targets")


@dataclass(frozen=True, slots=True)
class TensorRule:
    pattern: str
    component_template: str
    role: TensorRole
    axes: tuple[TensorAxis, ...]

    def match(self, tensor_name: str) -> TensorMapping | None:
        matched = re.fullmatch(self.pattern, tensor_name)
        if matched is None:
            return None
        component = self.component_template.format(**matched.groupdict())
        return TensorMapping(tensor_name, ComponentId.parse(component), self.role, self.axes)


@dataclass(frozen=True, slots=True)
class ArchitectureAlias:
    architecture: str
    metadata_prefix: str


@dataclass(frozen=True, slots=True)
class GGUFArchitectureContract:
    family: ModelFamily
    version: int
    aliases: tuple[ArchitectureAlias, ...]
    tensor_rules: tuple[TensorRule, ...]

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.aliases or not self.tensor_rules:
            raise ValueError("architecture contracts require a version, aliases, and tensor rules")
        architecture_names = [alias.architecture for alias in self.aliases]
        if len(architecture_names) != len(set(architecture_names)):
            raise ValueError("architecture aliases must be unique within a contract")


@dataclass(frozen=True, slots=True)
class ResolvedGGUFArchitecture:
    contract: GGUFArchitectureContract
    architecture: str
    metadata_prefix: str

    def map_tensor(self, tensor_name: str) -> TensorMapping:
        for rule in self.contract.tensor_rules:
            mapping = rule.match(tensor_name)
            if mapping is not None:
                return mapping
        raise UnknownTensorMappingError(
            f"{self.architecture} contract has no mapping for tensor {tensor_name!r}"
        )

    def metadata_key(self, semantic: MetadataSemantic) -> str:
        return f"{self.metadata_prefix}.{semantic.value}"

    def metadata_updates(
        self,
        changes: tuple[tuple[MetadataSemantic, int], ...],
    ) -> tuple[tuple[str, int], ...]:
        semantics = [semantic for semantic, _ in changes]
        if len(semantics) != len(set(semantics)):
            raise GGUFArchitectureError("metadata update semantics must be unique")
        for semantic, value in changes:
            if value <= 0:
                raise GGUFArchitectureError(
                    f"metadata update {semantic.value} must be a positive integer"
                )
        return tuple(
            sorted((self.metadata_key(semantic), value) for semantic, value in changes)
        )

    def rename_tensor_blocks(
        self,
        tensor_name: str,
        layer_map: tuple[tuple[int, int | None], ...],
    ) -> str | None:
        """Rename one mapped block tensor; ``None`` means its block was removed."""
        mapping = dict(layer_map)
        if len(mapping) != len(layer_map):
            raise GGUFArchitectureError("layer remap sources must be unique")
        matched = re.fullmatch(r"blk\.(?P<layer>\d+)\.(?P<suffix>.+)", tensor_name)
        if matched is None:
            self.map_tensor(tensor_name)
            return tensor_name
        layer = int(matched.group("layer"))
        if layer not in mapping:
            raise GGUFArchitectureError(f"layer remap has no disposition for block {layer}")
        target = mapping[layer]
        if target is None:
            return None
        if target < 0:
            raise GGUFArchitectureError("renamed block indices must be non-negative")
        renamed = f"blk.{target}.{matched.group('suffix')}"
        self.map_tensor(renamed)
        return renamed

    def coupling_groups(self, layer: int) -> tuple[CouplingGroup, ...]:
        if layer < 0:
            raise GGUFArchitectureError("layer index must be non-negative")
        prefix = f"blk.{layer}"
        return (
            CouplingGroup(
                f"model.layers.{layer}:attention-head:v1",
                CouplingKind.ATTENTION_HEAD,
                (
                    TensorAxisTarget(f"{prefix}.attn_q.weight", TensorRole.ATTENTION_Q, 1),
                    TensorAxisTarget(f"{prefix}.attn_output.weight", TensorRole.ATTENTION_O, 0),
                ),
            ),
            CouplingGroup(
                f"model.layers.{layer}:kv-head:v1",
                CouplingKind.KV_HEAD,
                (
                    TensorAxisTarget(f"{prefix}.attn_k.weight", TensorRole.ATTENTION_K, 1),
                    TensorAxisTarget(f"{prefix}.attn_v.weight", TensorRole.ATTENTION_V, 1),
                ),
            ),
            CouplingGroup(
                f"model.layers.{layer}:mlp-channel:v1",
                CouplingKind.MLP_CHANNEL,
                (
                    TensorAxisTarget(f"{prefix}.ffn_gate.weight", TensorRole.MLP_GATE, 1),
                    TensorAxisTarget(f"{prefix}.ffn_up.weight", TensorRole.MLP_UP, 1),
                    TensorAxisTarget(f"{prefix}.ffn_down.weight", TensorRole.MLP_DOWN, 0),
                ),
            ),
        )


_COMMON_TENSOR_RULES = (
    TensorRule(
        r"token_embd\.weight",
        "model.embed_tokens",
        TensorRole.TOKEN_EMBEDDING,
        (TensorAxis(0, AxisSemantic.HIDDEN_FEATURE), TensorAxis(1, AxisSemantic.VOCABULARY)),
    ),
    TensorRule(
        r"output\.weight",
        "model.lm_head",
        TensorRole.OUTPUT,
        (TensorAxis(0, AxisSemantic.HIDDEN_FEATURE), TensorAxis(1, AxisSemantic.VOCABULARY)),
    ),
    TensorRule(
        r"output_norm\.weight",
        "model.norm",
        TensorRole.OUTPUT_NORM,
        (TensorAxis(0, AxisSemantic.HIDDEN_FEATURE),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_norm\.weight",
        "model.layers.{layer}.input_layernorm",
        TensorRole.INPUT_NORM,
        (TensorAxis(0, AxisSemantic.HIDDEN_FEATURE),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_norm\.weight",
        "model.layers.{layer}.post_attention_layernorm",
        TensorRole.POST_ATTENTION_NORM,
        (TensorAxis(0, AxisSemantic.HIDDEN_FEATURE),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_q\.weight",
        "model.layers.{layer}.self_attn.q_proj",
        TensorRole.ATTENTION_Q,
        (TensorAxis(0, AxisSemantic.INPUT_FEATURE), TensorAxis(1, AxisSemantic.ATTENTION_HEAD)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_q\.bias",
        "model.layers.{layer}.self_attn.q_proj",
        TensorRole.ATTENTION_Q,
        (TensorAxis(0, AxisSemantic.ATTENTION_HEAD),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_k\.weight",
        "model.layers.{layer}.self_attn.k_proj",
        TensorRole.ATTENTION_K,
        (TensorAxis(0, AxisSemantic.INPUT_FEATURE), TensorAxis(1, AxisSemantic.KV_HEAD)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_k\.bias",
        "model.layers.{layer}.self_attn.k_proj",
        TensorRole.ATTENTION_K,
        (TensorAxis(0, AxisSemantic.KV_HEAD),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_v\.weight",
        "model.layers.{layer}.self_attn.v_proj",
        TensorRole.ATTENTION_V,
        (TensorAxis(0, AxisSemantic.INPUT_FEATURE), TensorAxis(1, AxisSemantic.KV_HEAD)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_v\.bias",
        "model.layers.{layer}.self_attn.v_proj",
        TensorRole.ATTENTION_V,
        (TensorAxis(0, AxisSemantic.KV_HEAD),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_output\.weight",
        "model.layers.{layer}.self_attn.o_proj",
        TensorRole.ATTENTION_O,
        (TensorAxis(0, AxisSemantic.ATTENTION_HEAD), TensorAxis(1, AxisSemantic.OUTPUT_FEATURE)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.attn_output\.bias",
        "model.layers.{layer}.self_attn.o_proj",
        TensorRole.ATTENTION_O,
        (TensorAxis(0, AxisSemantic.OUTPUT_FEATURE),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_gate\.weight",
        "model.layers.{layer}.mlp.gate_proj",
        TensorRole.MLP_GATE,
        (TensorAxis(0, AxisSemantic.INPUT_FEATURE), TensorAxis(1, AxisSemantic.MLP_CHANNEL)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_gate\.bias",
        "model.layers.{layer}.mlp.gate_proj",
        TensorRole.MLP_GATE,
        (TensorAxis(0, AxisSemantic.MLP_CHANNEL),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_up\.weight",
        "model.layers.{layer}.mlp.up_proj",
        TensorRole.MLP_UP,
        (TensorAxis(0, AxisSemantic.INPUT_FEATURE), TensorAxis(1, AxisSemantic.MLP_CHANNEL)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_up\.bias",
        "model.layers.{layer}.mlp.up_proj",
        TensorRole.MLP_UP,
        (TensorAxis(0, AxisSemantic.MLP_CHANNEL),),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_down\.weight",
        "model.layers.{layer}.mlp.down_proj",
        TensorRole.MLP_DOWN,
        (TensorAxis(0, AxisSemantic.MLP_CHANNEL), TensorAxis(1, AxisSemantic.OUTPUT_FEATURE)),
    ),
    TensorRule(
        r"blk\.(?P<layer>\d+)\.ffn_down\.bias",
        "model.layers.{layer}.mlp.down_proj",
        TensorRole.MLP_DOWN,
        (TensorAxis(0, AxisSemantic.OUTPUT_FEATURE),),
    ),
)


GGUF_ARCHITECTURE_CONTRACTS = (
    GGUFArchitectureContract(
        ModelFamily.LLAMA,
        1,
        (ArchitectureAlias("llama", "llama"),),
        _COMMON_TENSOR_RULES,
    ),
    GGUFArchitectureContract(
        ModelFamily.MISTRAL,
        1,
        (ArchitectureAlias("mistral", "mistral"), ArchitectureAlias("llama", "llama")),
        _COMMON_TENSOR_RULES,
    ),
    GGUFArchitectureContract(
        ModelFamily.QWEN,
        1,
        (
            ArchitectureAlias("qwen2", "qwen2"),
            ArchitectureAlias("qwen2moe", "qwen2moe"),
            ArchitectureAlias("qwen3", "qwen3"),
            ArchitectureAlias("qwen3moe", "qwen3moe"),
        ),
        _COMMON_TENSOR_RULES,
    ),
    GGUFArchitectureContract(
        ModelFamily.GEMMA,
        1,
        (
            ArchitectureAlias("gemma", "gemma"),
            ArchitectureAlias("gemma2", "gemma2"),
            ArchitectureAlias("gemma3", "gemma3"),
        ),
        _COMMON_TENSOR_RULES,
    ),
)


def resolve_gguf_architecture(
    architecture: str,
    *,
    family: ModelFamily | None = None,
) -> ResolvedGGUFArchitecture:
    """Resolve explicit GGUF metadata, requiring a family when an alias is ambiguous."""
    normalized = architecture.strip().lower().replace("_", "").replace("-", "")
    matches: list[tuple[GGUFArchitectureContract, ArchitectureAlias]] = []
    for contract in GGUF_ARCHITECTURE_CONTRACTS:
        if family is not None and contract.family is not family:
            continue
        for alias in contract.aliases:
            alias_normalized = alias.architecture.replace("_", "").replace("-", "")
            if normalized == alias_normalized:
                matches.append((contract, alias))
    if not matches:
        family_detail = "" if family is None else f" for explicit family {family.value}"
        raise UnknownGGUFArchitectureError(
            f"unsupported GGUF general.architecture {architecture!r}{family_detail}"
        )
    if len(matches) != 1:
        families = ", ".join(sorted(match[0].family.value for match in matches))
        raise AmbiguousGGUFArchitectureError(
            f"GGUF architecture {architecture!r} matches multiple families ({families}); "
            "supply an explicit family"
        )
    contract, alias = matches[0]
    return ResolvedGGUFArchitecture(contract, alias.architecture, alias.metadata_prefix)
