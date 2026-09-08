from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import ProviderConfig
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.setup import (
    SetupCheck,
    SetupOutcome,
    SetupReport,
    SetupRequest,
    SetupStatus,
    diagnose_setup,
    initialize_setup,
)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tiny_hf_models_v1.json"


def _codes(report: SetupReport) -> set[str]:
    return {check.code for check in report.checks}


def _check(report: SetupReport, category: str) -> SetupCheck:
    return next(check for check in report.checks if check.category == category)


def test_offline_init_creates_separated_layout_for_supported_fixture(tmp_path: Path) -> None:
    report = initialize_setup(
        SetupRequest(tmp_path / "data", fixture=FIXTURE, offline=True, min_free_bytes=1),
    )

    assert report.status is SetupStatus.READY
    assert report.outcome is SetupOutcome.SUPPORTED
    for name in ("Surgeon Tensors", "Text LLM", "Models", "Config"):
        assert (tmp_path / "data" / name).is_dir()
    assert (tmp_path / "data" / "Config" / "modelsurgeon.toml").is_file()
    assert (tmp_path / "data" / "README.md").is_file()
    assert "Text LLM/" in (tmp_path / "data" / "README.md").read_text(encoding="utf-8")


def test_diagnostics_does_not_create_missing_data_root(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    report = diagnose_setup(SetupRequest(root, offline=True, fixture=FIXTURE, min_free_bytes=1))

    assert not root.exists()
    assert report.status is SetupStatus.NEEDS_ATTENTION
    assert "missing_data_directory" in _codes(report)


def test_setup_reports_non_directory_data_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.write_text("not a directory", encoding="utf-8")

    report = diagnose_setup(SetupRequest(root, min_free_bytes=1))

    assert report.status is SetupStatus.FAILED
    assert "data_root_not_directory" in _codes(report)


def test_offline_setup_requires_a_local_fixture(tmp_path: Path) -> None:
    report = diagnose_setup(SetupRequest(tmp_path, offline=True, min_free_bytes=1))

    assert "offline_fixture_required" in _codes(report)
    assert report.outcome is SetupOutcome.UNSUPPORTED


def test_missing_fixture_is_retained_as_failed_evidence(tmp_path: Path) -> None:
    report = diagnose_setup(
        SetupRequest(tmp_path, fixture=tmp_path / "missing.json", min_free_bytes=1)
    )

    assert _check(report, "supported_fixture").code == "fixture_missing"
    assert report.outcome is SetupOutcome.FAILED


def test_disk_budget_and_permission_cells_are_explicit(tmp_path: Path, monkeypatch) -> None:
    import modelsurgeon.setup as setup_module

    report = diagnose_setup(SetupRequest(tmp_path, min_free_bytes=1 << 60))
    assert "disk_budget_exceeded" in _codes(report)

    monkeypatch.setattr(
        setup_module,
        "_probe_writable",
        lambda _: "data root is not writable: denied",
    )
    permission_report = diagnose_setup(SetupRequest(tmp_path, min_free_bytes=1))
    assert "permission_denied" in _codes(permission_report)


def test_provider_absence_is_not_silently_promoted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MODELSURGEON_SETUP_TEST_KEY", raising=False)
    config = ProviderConfig(
        kind=ProviderKind.COMPATIBLE_ENDPOINT,
        provider_id="endpoint",
        model_id="model",
        model_revision="revision-1",
        endpoint="https://provider.example/v1",
        api_key_env="MODELSURGEON_SETUP_TEST_KEY",
    )
    report = diagnose_setup(SetupRequest(tmp_path, provider=config, min_free_bytes=1))

    provider_check = _check(report, "provider")
    assert provider_check.code == "missing_api_key"
    assert provider_check.outcome is SetupOutcome.UNAVAILABLE


def test_offline_provider_selection_is_explicitly_unsupported(tmp_path: Path) -> None:
    config = ProviderConfig(
        kind=ProviderKind.LOCAL,
        provider_id="local",
        model_id="chat",
        model_revision="revision-1",
    )
    report = diagnose_setup(SetupRequest(tmp_path, provider=config, offline=True, min_free_bytes=1))

    provider_check = _check(report, "provider")
    assert provider_check.code == "offline_provider_disabled"
    assert provider_check.outcome is SetupOutcome.UNSUPPORTED


def test_absent_local_provider_runtime_is_explicitly_unsupported(tmp_path: Path) -> None:
    config = ProviderConfig(
        kind=ProviderKind.LOCAL,
        provider_id="local",
        model_id="chat",
        model_revision="revision-1",
    )
    report = diagnose_setup(SetupRequest(tmp_path, provider=config, min_free_bytes=1))

    provider_check = _check(report, "provider")
    assert provider_check.code == "adapter_unavailable"
    assert provider_check.outcome is SetupOutcome.UNSUPPORTED


def test_setup_report_is_canonical_and_redacted(tmp_path: Path) -> None:
    report = diagnose_setup(SetupRequest(tmp_path, min_free_bytes=1))
    payload = json.loads(report.canonical_json())

    assert payload["record_type"] == "modelsurgeon_setup_report"
    assert payload["schema_version"] == 1
    assert "secret" not in report.canonical_json().lower()


def test_setup_init_preserves_existing_manifest_without_force(tmp_path: Path) -> None:
    request = SetupRequest(tmp_path / "data", fixture=FIXTURE, offline=True, min_free_bytes=1)
    first = initialize_setup(request)
    second = initialize_setup(request)

    assert first.status is SetupStatus.READY
    assert second.status is SetupStatus.NEEDS_ATTENTION
    assert "configuration_exists" in _codes(second)


def test_setup_cli_offline_fixture_smoke(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "init",
            "--data-dir",
            str(tmp_path / "data"),
            "--fixture",
            str(FIXTURE),
            "--offline",
            "--min-free-gb",
            "0.000001",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["record_type"] == "modelsurgeon_setup_report"
    assert record["status"] == "ready"
    assert record["outcome"] == "supported"
