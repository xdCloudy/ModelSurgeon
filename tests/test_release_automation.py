"""Tests for release metadata, notes, and approval-gated workflow contracts."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from tools.generate_release_notes import ReleaseNotesError, render_release_notes, unreleased_body
from tools.verify_release_artifacts import ReleaseArtifactError, verify_release_artifacts

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def test_unreleased_notes_are_scoped_to_the_current_section() -> None:
    source = "## Unreleased\n\n- one\n- two\n\n## Next\n\n- do not include\n"
    assert unreleased_body(source) == "- one\n- two"
    assert render_release_notes(source, "v0.0.1") == "# ModelSurgeon v0.0.1\n\n- one\n- two\n"


def test_empty_unreleased_section_is_rejected() -> None:
    with pytest.raises(ReleaseNotesError, match="empty"):
        unreleased_body("## Unreleased\n\n## Next\n")


def _metadata() -> bytes:
    return (
        b"Metadata-Version: 2.3\n"
        b"Name: modelsurgeon\n"
        b"Version: 1.0.0\n"
        b"Requires-Python: >=3.12\n\n"
    )


def _write_test_dist(path: Path) -> None:
    path.mkdir()
    with zipfile.ZipFile(path / "modelsurgeon-1.0.0-py3-none-any.whl", "w") as archive:
        archive.writestr("modelsurgeon/__init__.py", "")
        archive.writestr("modelsurgeon-1.0.0.dist-info/METADATA", _metadata())
    with tarfile.open(path / "modelsurgeon-1.0.0.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("modelsurgeon-1.0.0/PKG-INFO")
        payload = _metadata()
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def test_release_artifact_verifier_checks_both_archive_types(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _write_test_dist(dist)

    assert verify_release_artifacts(dist, project=ROOT / "pyproject.toml", tag="v1.0.0") == (
        dist / "modelsurgeon-1.0.0-py3-none-any.whl",
        dist / "modelsurgeon-1.0.0.tar.gz",
    )


def test_release_artifact_verifier_rejects_a_tag_mismatch(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _write_test_dist(dist)

    with pytest.raises(ReleaseArtifactError, match="does not match"):
        verify_release_artifacts(dist, project=ROOT / "pyproject.toml", tag="v9.9.9")


def test_release_workflow_has_build_attest_and_approval_gates() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for fragment in (
        "release:",
        "types: [published]",
        "uv build --out-dir dist",
        "verify_release_artifacts.py",
        "generate_release_notes.py",
        "uv venv .release-venv --python 3.12",
        "actions/attest-build-provenance@v2",
        "attestations: write",
        "pypa/gh-action-pypi-publish@release/v1",
        "name: pypi",
        "needs: [build, attest]",
    ):
        assert fragment in text
