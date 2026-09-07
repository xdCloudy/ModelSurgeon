"""Shared provider selection enum with no conversational-package imports."""

from enum import StrEnum


class ProviderKind(StrEnum):
    LOCAL = "local"
    COMPATIBLE_ENDPOINT = "compatible_endpoint"
    HOSTED = "hosted"
    NONE = "none"


__all__ = ["ProviderKind"]
