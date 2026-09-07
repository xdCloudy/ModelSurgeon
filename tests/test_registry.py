"""Offline local artifact registry contracts."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.registry import LocalArtifactRegistry, RegistryError, RegistryOutcome


def test_publish_verify_alias_tag_and_pagination_are_deterministic(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(tmp_path / "registry")
    first = registry.publish_bytes(b"one", kind="model", source=True)
    second = registry.publish_bytes(b"two", kind="surgeon", tags=("candidate",))
    registry.set_alias("current", second.digest)
    tagged = registry.tag(first.digest, "baseline")

    assert tagged.digest == first.digest
    assert registry.inspect("current").digest == second.digest
    assert registry.verify(first.digest).outcome is RegistryOutcome.VERIFIED
    page = registry.list(page_size=1)
    assert len(page.artifacts) == 1
    assert page.next_cursor is not None
    assert len(registry.list(cursor=page.next_cursor, page_size=1).artifacts) == 1


def test_gc_preserves_sources_leases_references_and_aliases(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(tmp_path / "registry")
    source = registry.publish_bytes(b"source", kind="model", source=True)
    leased = registry.publish_bytes(b"leased", kind="model")
    referenced = registry.publish_bytes(b"referenced", kind="evidence", externally_unresolved=True)
    garbage = registry.publish_bytes(b"garbage", kind="surgeon")
    registry.lease(leased.digest, "run-1")
    registry.add_reference(source.digest, referenced.digest)
    registry.set_alias("garbage-alias", garbage.digest)

    preview = registry.collect_garbage()
    assert preview.dry_run is True
    assert garbage.digest not in preview.candidates
    assert leased.digest in preview.protected
    assert registry.collect_garbage(dry_run=False).deleted == ()

    registry.release_lease("run-1")
    registry.remove_alias("garbage-alias")
    result = registry.collect_garbage(dry_run=False)
    assert garbage.digest in result.deleted
    with pytest.raises(RegistryError):
        registry.inspect(garbage.digest)


def test_export_import_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"round-trip", kind="surgeon", license_id="apache-2.0")
    bundle = tmp_path / "bundle.zip"
    source.export_bundle(artifact.digest, bundle, signing_key=b"test-key")

    target = LocalArtifactRegistry(tmp_path / "target")
    imported = target.import_bundle(
        bundle,
        signing_key=b"test-key",
        accepted_licenses=("apache-2.0",),
    )
    assert imported.digest == artifact.digest
    assert target.verify(imported.digest).outcome is RegistryOutcome.VERIFIED

    with ZipFile(bundle, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("payload") + b"tamper"
    tampered = tmp_path / "tampered.zip"
    with ZipFile(tampered, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload", payload)
    with pytest.raises(RegistryError, match="digest"):
        target.import_bundle(tampered, signing_key=b"test-key")


def test_unsigned_evidence_requires_explicit_negative_path(tmp_path: Path) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"evidence", kind="evidence")
    bundle = tmp_path / "evidence.zip"
    source.export_bundle(artifact.digest, bundle)
    target = LocalArtifactRegistry(tmp_path / "target")

    with pytest.raises(RegistryError, match="signature"):
        target.import_bundle(bundle)
    assert target.import_bundle(bundle, allow_unsigned=True).digest == artifact.digest


def test_registry_cli_emits_stable_json(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = LocalArtifactRegistry(root)
    artifact = registry.publish_bytes(b"cli", kind="model")
    result = CliRunner().invoke(
        app,
        ["registry", "verify", artifact.digest, "--root", str(root), "--json"],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["outcome"] == "verified"
