"""Regression coverage for the documented public package boundary."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

_PUBLIC_NAMESPACES = (
    "modelsurgeon.adapters",
    "modelsurgeon.graph",
    "modelsurgeon.datasets",
    "modelsurgeon.features",
    "modelsurgeon.surgery",
    "modelsurgeon.evaluation",
    "modelsurgeon.experiments",
    "modelsurgeon.surgeon",
    "modelsurgeon.search",
    "modelsurgeon.explain",
)
_POLICY = Path("docs/api-compatibility.md")


@pytest.mark.parametrize("namespace", _PUBLIC_NAMESPACES)
def test_documented_public_namespaces_export_typed_symbols(namespace: str) -> None:
    module = import_module(namespace)
    exports = tuple(module.__all__)

    assert len(exports) == len(set(exports))
    assert exports
    for name in exports:
        assert hasattr(module, name)


def test_compatibility_policy_documents_every_supported_namespace() -> None:
    policy = _POLICY.read_text(encoding="utf-8")

    for namespace in _PUBLIC_NAMESPACES:
        assert f"`{namespace}`" in policy
