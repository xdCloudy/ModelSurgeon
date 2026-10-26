"""Helpers for preserving per-module PyTorch training modes across repair work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def snapshot_module_training_modes(model: Any) -> dict[str, bool]:
    """Return the training flag for every named module in a model."""

    return {name: bool(module.training) for name, module in model.named_modules()}


def restore_module_training_modes(
    model: Any,
    modes: Mapping[str, bool],
    *,
    subtree_roots: Sequence[str] = (),
) -> None:
    """Restore named module modes, including new descendants of wrapped targets."""

    if "" in modes:
        model.train(modes[""])
    modules = dict(model.named_modules())
    for name, was_training in modes.items():
        module = modules.get(name)
        if module is not None:
            module.training = was_training
    for name in subtree_roots:
        module = modules.get(name)
        if name in modes and module is not None:
            was_training = modes[name]
            for descendant in module.modules():
                descendant.training = was_training
