"""Bounded reads for user-supplied CLI JSON manifests."""

from __future__ import annotations

from pathlib import Path

MAX_CLI_JSON_INPUT_BYTES = 64 * 1024 * 1024


class CliJsonInputError(ValueError):
    """Raised when a CLI JSON input exceeds its byte ceiling or is not UTF-8."""


def read_cli_json_text(path: Path) -> str:
    """Read one UTF-8 JSON input with a hard byte ceiling."""

    with path.open("rb") as stream:
        payload = stream.read(MAX_CLI_JSON_INPUT_BYTES + 1)
    if len(payload) > MAX_CLI_JSON_INPUT_BYTES:
        raise CliJsonInputError(
            f"JSON input exceeds {MAX_CLI_JSON_INPUT_BYTES} bytes"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CliJsonInputError("JSON input must be UTF-8") from error
