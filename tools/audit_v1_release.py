"""Audit the checked-in contract required for a reproducible v1.0 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath

from tools.generate_release_notes import ReleaseNotesError, release_body

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DOC = ROOT / "docs" / "release" / "v1.0-release-audit.md"
MANIFEST = ROOT / "docs" / "release" / "v1.0-reference-manifests.json"


class ReleaseAuditError(ValueError):
    """Raised when a required release contract is missing or inconsistent."""


def _version(project: Path) -> str:
    with project.open("rb") as handle:
        value = tomllib.load(handle)["project"]["version"]
    if not isinstance(value, str) or not value:
        raise ReleaseAuditError("pyproject.toml has no project version")
    return value


def audit_release(root: Path = ROOT, *, version: str) -> None:
    normalized = version.strip().removeprefix("v")
    if not normalized:
        raise ReleaseAuditError("release version cannot be empty")
    project_version = _version(root / "pyproject.toml")
    if project_version != normalized:
        raise ReleaseAuditError(
            f"project version {project_version!r} does not match release {normalized!r}"
        )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    try:
        release_body(changelog, version)
    except ReleaseNotesError as error:
        raise ReleaseAuditError(str(error)) from error

    audit_text = (root / AUDIT_DOC.relative_to(ROOT)).read_text(encoding="utf-8")
    required_phrases = (
        f"`v{normalized}`",
        "Supported model and format boundary",
        "Codec boundary",
        "Limitations that ship with v1.0",
        "Publication gate",
    )
    missing = [phrase for phrase in required_phrases if phrase not in audit_text]
    if missing:
        raise ReleaseAuditError(f"release audit is missing sections: {', '.join(missing)}")

    raw_manifest = json.loads((root / MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))
    if raw_manifest.get("release") != f"v{normalized}":
        raise ReleaseAuditError("reference manifest release does not match project version")
    entries = raw_manifest.get("manifests")
    if not isinstance(entries, list) or not entries:
        raise ReleaseAuditError("reference manifest has no entries")
    root_path = root.resolve()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReleaseAuditError("reference manifest contains an invalid entry")
        manifest_path = entry["path"]
        relative_path = PurePosixPath(manifest_path)
        windows_path = PureWindowsPath(manifest_path)
        if (
            not manifest_path
            or "\\" in manifest_path
            or relative_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in relative_path.parts
        ):
            raise ReleaseAuditError(
                "reference manifest path must be a safe repository-relative path: "
                f"{manifest_path!r}"
            )
        try:
            path = (root_path / Path(*relative_path.parts)).resolve()
        except (OSError, RuntimeError, ValueError) as error:
            raise ReleaseAuditError(
                f"reference manifest path cannot be resolved: {manifest_path!r}"
            ) from error
        if not path.is_relative_to(root_path):
            raise ReleaseAuditError(
                f"reference manifest path escapes repository root: {manifest_path!r}"
            )
        if not path.is_file():
            raise ReleaseAuditError(f"reference manifest path does not exist: {entry['path']}")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ReleaseAuditError(f"reference manifest has no valid SHA-256: {entry['path']}")
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        actual_hash = hashlib.sha256(canonical).hexdigest()
        if actual_hash != expected_hash:
            raise ReleaseAuditError(
                f"reference manifest hash mismatch for {entry['path']}: {actual_hash}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    audit_release(version=args.version)
    print(f"audited reproducible release contract for {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
