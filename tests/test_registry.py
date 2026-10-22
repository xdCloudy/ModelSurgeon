"""Offline local artifact registry contracts."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

import modelsurgeon.registry as registry_module
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


def test_export_streams_payload_without_reading_it_all_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"streamed payload", kind="model")
    bundle = tmp_path / "streamed.zip"

    def reject_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"export tried to read the complete file into memory: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    source.export_bundle(artifact.digest, bundle)

    with ZipFile(bundle, "r") as archive:
        assert archive.read("payload") == b"streamed payload"


def test_export_failure_preserves_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"streamed payload", kind="model")
    bundle = tmp_path / "bundle.zip"
    previous = b"previous complete bundle"
    bundle.write_bytes(previous)

    def fail_replace(source_path: str | bytes | os.PathLike[str], target_path: str | bytes) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(registry_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        source.export_bundle(artifact.digest, bundle, allow_overwrite=True)

    assert bundle.read_bytes() == previous
    assert list(tmp_path.glob(".bundle.zip.*.partial")) == []


def test_publish_file_streams_without_loading_the_complete_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "model.bin"
    source_path.write_bytes(b"model payload" * 1024)
    registry = LocalArtifactRegistry(tmp_path / "registry")

    def reject_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"file publication tried to read the complete file: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    artifact = registry.publish_file(source_path, kind="model", tags=("baseline",))

    assert artifact.size_bytes == len(b"model payload" * 1024)
    assert artifact.tags == ("baseline",)
    assert registry.verify(artifact.digest).outcome is RegistryOutcome.VERIFIED


def test_unsigned_evidence_requires_explicit_negative_path(tmp_path: Path) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"evidence", kind="evidence")
    bundle = tmp_path / "evidence.zip"
    source.export_bundle(artifact.digest, bundle)
    target = LocalArtifactRegistry(tmp_path / "target")

    with pytest.raises(RegistryError, match="signature"):
        target.import_bundle(bundle)
    assert target.import_bundle(bundle, allow_unsigned=True).digest == artifact.digest


def test_import_rejects_entry_count_before_opening_zipfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "too-many-members.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("payload", b"payload")
        archive.writestr("unexpected", b"extra")

    def unexpected_zip_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("ZIP entries should be rejected before ZipFile opens them")

    monkeypatch.setattr(registry_module, "ZipFile", unexpected_zip_open)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="exactly two members"):
        registry.import_bundle(bundle)


def test_import_enforces_configurable_payload_limit(tmp_path: Path) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"bounded payload", kind="model")
    bundle = tmp_path / "payload.zip"
    source.export_bundle(artifact.digest, bundle)
    target = LocalArtifactRegistry(tmp_path / "target")

    with pytest.raises(RegistryError, match="payload exceeds the import size limit"):
        target.import_bundle(bundle, max_payload_bytes=4)
    assert target.import_bundle(bundle, max_payload_bytes=1024).digest == artifact.digest


def test_import_accepts_zip64_directory_metadata(tmp_path: Path) -> None:
    source = LocalArtifactRegistry(tmp_path / "source")
    artifact = source.publish_bytes(b"zip64 payload", kind="model")
    bundle = tmp_path / "zip64.zip"
    source.export_bundle(artifact.digest, bundle)

    archive_bytes = bundle.read_bytes()
    eocd_offset = archive_bytes.rfind(b"PK\x05\x06")
    eocd = struct.unpack_from("<4s4H2LH", archive_bytes, eocd_offset)
    zip64_record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        eocd[3],
        eocd[4],
        eocd[5],
        eocd[6],
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, eocd_offset, 1)
    sentinel_eocd = struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    bundle.write_bytes(archive_bytes[:eocd_offset] + zip64_record + locator + sentinel_eocd)

    imported = LocalArtifactRegistry(tmp_path / "target").import_bundle(bundle)
    assert imported.digest == artifact.digest


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
