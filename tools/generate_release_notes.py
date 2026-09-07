"""Generate release notes from the repository's Unreleased changelog section."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_UNRELEASED = re.compile(r"^## Unreleased\s*$", re.MULTILINE)
_NEXT_SECTION = re.compile(r"^## (?!Unreleased\b).*$", re.MULTILINE)


class ReleaseNotesError(ValueError):
    """Raised when the changelog has no usable Unreleased section."""


def unreleased_body(changelog: str) -> str:
    """Return the body of the Unreleased section without another heading."""
    match = _UNRELEASED.search(changelog)
    if match is None:
        raise ReleaseNotesError("CHANGELOG.md has no ## Unreleased section")
    start = match.end()
    next_match = _NEXT_SECTION.search(changelog, start)
    end = next_match.start() if next_match is not None else len(changelog)
    body = changelog[start:end].strip()
    if not body:
        raise ReleaseNotesError("CHANGELOG.md has an empty ## Unreleased section")
    return body


def render_release_notes(changelog: str, version: str) -> str:
    """Render stable release notes without mutating CHANGELOG.md."""
    version = version.strip()
    if not version:
        raise ReleaseNotesError("release version cannot be empty")
    return f"# ModelSurgeon {version}\n\n{unreleased_body(changelog)}\n"


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
