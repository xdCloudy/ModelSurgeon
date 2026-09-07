"""Generate release notes from a versioned or Unreleased changelog section."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_UNRELEASED = re.compile(r"^## Unreleased\s*$", re.MULTILINE)
_NEXT_SECTION = re.compile(r"^## .*$", re.MULTILINE)


class ReleaseNotesError(ValueError):
    """Raised when the changelog has no usable release section."""


def _section_body(changelog: str, match: re.Match[str], label: str) -> str:
    start = match.end()
    next_match = _NEXT_SECTION.search(changelog, start)
    end = next_match.start() if next_match is not None else len(changelog)
    body = changelog[start:end].strip()
    if not body:
        raise ReleaseNotesError(f"CHANGELOG.md has an empty {label} section")
    return body


def unreleased_body(changelog: str) -> str:
    """Return the body of the Unreleased section without another heading."""
    match = _UNRELEASED.search(changelog)
    if match is None:
        raise ReleaseNotesError("CHANGELOG.md has no ## Unreleased section")
    return _section_body(changelog, match, "## Unreleased")


def release_body(changelog: str, version: str) -> str:
    """Return the matching version section, falling back to Unreleased."""
    normalized = version.strip().removeprefix("v")
    if not normalized:
        raise ReleaseNotesError("release version cannot be empty")
    pattern = re.compile(rf"^## {re.escape(normalized)}(?:\s|$).*$", re.MULTILINE)
    match = pattern.search(changelog)
    if match is not None:
        return _section_body(changelog, match, f"## {normalized}")
    return unreleased_body(changelog)


def render_release_notes(changelog: str, version: str) -> str:
    """Render stable release notes without mutating CHANGELOG.md."""
    version = version.strip()
    if not version:
        raise ReleaseNotesError("release version cannot be empty")
    return f"# ModelSurgeon {version}\n\n{release_body(changelog, version)}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changelog", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render_release_notes(args.changelog.read_text(encoding="utf-8"), args.version)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
