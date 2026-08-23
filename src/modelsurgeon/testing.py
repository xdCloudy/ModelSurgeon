"""Deterministic, dependency-free transformer doubles for offline tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul

from modelsurgeon.adapters.family import ModelFamily


@dataclass(frozen=True, slots=True)
class TinyTransformerConfig:
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 12
    hidden_size: int = 8
    vocab_size: int = 32
    tie_word_embeddings: bool = False


@dataclass(frozen=True, slots=True)
class TinyModule:
    """Named module stand-in with a stable type label."""

    module_type: str


@dataclass(frozen=True, slots=True)
class TinyParameter:
    """Shape-only parameter whose scalar samples are generated on demand."""

    name: str
    shape: tuple[int, ...]
    seed: int

    def numel(self) -> int:
        return reduce(mul, self.shape, 1)

    def sample(self, index: int) -> float:
        """Return a reproducible pseudo-value without allocating a weight array."""
        if index < 0 or index >= self.numel():
            raise IndexError(f"parameter sample index {index} is outside 0..{self.numel() - 1}")
        payload = f"{self.seed}:{self.name}:{index}".encode()
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return raw / ((1 << 64) - 1) * 2.0 - 1.0


_FAMILY_CONFIG = {
    ModelFamily.LLAMA: ("llama", "LlamaForCausalLM"),
    ModelFamily.QWEN: ("qwen2", "Qwen2ForCausalLM"),
    ModelFamily.MISTRAL: ("mistral", "MistralForCausalLM"),
    ModelFamily.GEMMA: ("gemma2", "Gemma2ForCausalLM"),
}


class TinyTransformerDouble:
    """Minimal HF-shaped causal transformer with no framework dependency or payload."""

    def __init__(
        self,
        family: ModelFamily,
        *,
        seed: int = 1729,
        tie_word_embeddings: bool = False,
    ) -> None:
        model_type, architecture = _FAMILY_CONFIG[family]
        self.family = family
        self.seed = seed
        self.config = TinyTransformerConfig(
            model_type,
            (architecture,),
            tie_word_embeddings=tie_word_embeddings,
        )
        self._modules = self._build_modules()
        self._parameters = self._build_parameters()

    def _build_modules(self) -> tuple[tuple[str, TinyModule | TinyTransformerDouble], ...]:
        modules: list[tuple[str, TinyModule | TinyTransformerDouble]] = [
            ("", self),
            ("model", TinyModule("decoder")),
            ("model.embed_tokens", TinyModule("embedding")),
            ("model.layers", TinyModule("module_list")),
        ]
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            modules.extend(
                (
                    (prefix, TinyModule("decoder_layer")),
                    (f"{prefix}.self_attn", TinyModule("attention")),
                    (f"{prefix}.self_attn.q_proj", TinyModule("linear")),
                    (f"{prefix}.self_attn.k_proj", TinyModule("linear")),
                    (f"{prefix}.self_attn.v_proj", TinyModule("linear")),
                    (f"{prefix}.self_attn.o_proj", TinyModule("linear")),
                    (f"{prefix}.mlp", TinyModule("mlp")),
                    (f"{prefix}.mlp.gate_proj", TinyModule("linear")),
                    (f"{prefix}.mlp.up_proj", TinyModule("linear")),
                    (f"{prefix}.mlp.down_proj", TinyModule("linear")),
                    (f"{prefix}.input_layernorm", TinyModule("rms_norm")),
                    (f"{prefix}.post_attention_layernorm", TinyModule("rms_norm")),
                )
            )
        modules.extend(
            (
                ("model.norm", TinyModule("rms_norm")),
                ("lm_head", TinyModule("linear")),
            )
        )
        return tuple(modules)

    def _parameter(self, name: str, shape: tuple[int, ...]) -> TinyParameter:
        return TinyParameter(name, shape, self.seed)

    def _build_parameters(self) -> tuple[tuple[str, TinyParameter], ...]:
        hidden = self.config.hidden_size
        kv_width = hidden // self.config.num_attention_heads * self.config.num_key_value_heads
        intermediate = self.config.intermediate_size
        embed = self._parameter(
            "model.embed_tokens.weight",
            (self.config.vocab_size, hidden),
        )
        parameters: list[tuple[str, TinyParameter]] = [
            ("model.embed_tokens.weight", embed)
        ]
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            shapes = (
                (f"{prefix}.self_attn.q_proj.weight", (hidden, hidden)),
                (f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
                (f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
                (f"{prefix}.self_attn.o_proj.weight", (hidden, hidden)),
                (f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden)),
                (f"{prefix}.mlp.up_proj.weight", (intermediate, hidden)),
                (f"{prefix}.mlp.down_proj.weight", (hidden, intermediate)),
                (f"{prefix}.input_layernorm.weight", (hidden,)),
                (f"{prefix}.post_attention_layernorm.weight", (hidden,)),
            )
            parameters.extend((name, self._parameter(name, shape)) for name, shape in shapes)
        parameters.append(
            ("model.norm.weight", self._parameter("model.norm.weight", (hidden,)))
        )
        lm_head = (
            embed
            if self.config.tie_word_embeddings
            else self._parameter("lm_head.weight", (self.config.vocab_size, hidden))
        )
        parameters.append(("lm_head.weight", lm_head))
        return tuple(parameters)

    def named_modules(self) -> Iterable[tuple[str, object]]:
        return self._modules

    def named_parameters(self) -> Iterable[tuple[str, TinyParameter]]:
        return self._parameters

    def fingerprint(self) -> str:
        """Hash the complete generated structure and seed for fixture identity."""
        record = {
            "schema_version": 1,
            "family": self.family.value,
            "seed": self.seed,
            "config": asdict(self.config),
            "modules": [name for name, _ in self._modules],
            "parameters": [
                {"name": name, "shape": parameter.shape}
                for name, parameter in self._parameters
            ],
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def tiny_transformer(
    family: ModelFamily,
    *,
    seed: int = 1729,
    tie_word_embeddings: bool = False,
) -> TinyTransformerDouble:
    """Create one deterministic offline transformer double."""
    return TinyTransformerDouble(
        family,
        seed=seed,
        tie_word_embeddings=tie_word_embeddings,
    )
