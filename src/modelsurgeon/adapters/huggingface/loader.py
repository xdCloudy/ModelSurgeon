"""Safe, lazy Hugging Face loading boundary."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class HuggingFaceLoadRequest:
    """Immutable inputs controlling a causal language-model load."""

    model: str
    revision: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = "auto"


def load_causal_lm(request: HuggingFaceLoadRequest) -> Any:
    """Load a causal LM without importing heavyweight dependencies at package import."""
    try:
        auto_model = import_module("transformers").AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support is optional; install with `uv sync --extra hf`"
        ) from exc

    return auto_model.from_pretrained(
        request.model,
        revision=request.revision,
        trust_remote_code=request.trust_remote_code,
        device_map=request.device_map,
    )
