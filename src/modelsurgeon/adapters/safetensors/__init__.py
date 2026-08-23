"""Safetensors model adapter."""

from modelsurgeon.adapters.safetensors.index import (
    SafetensorEntry,
    SafetensorsIndexError,
    inspect_safetensors,
    inspect_safetensors_file,
)

__all__ = [
    "SafetensorEntry",
    "SafetensorsIndexError",
    "inspect_safetensors",
    "inspect_safetensors_file",
]

