"""Local-first data layout, setup diagnostics, and first-run initialization.

Setup is intentionally non-downloading.  It creates user-owned directories,
records the selected local fixture, and reports whether optional provider
runtime pieces are available.  It does not install Python packages, fetch
models, or make a remote request.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from modelsurgeon.config import ProviderConfig
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.providers import provider_diagnostics

SETUP_SCHEMA_VERSION: Final = 1
DEFAULT_MIN_FREE_BYTES: Final = 1 << 30
LAYOUT_DIRECTORIES: Final = (
    Path("Surgeon Tensors"),
    Path("Text LLM"),
    Path("Models"),
    Path("Campaigns") / "active",
    Path("Campaigns") / "completed",
    Path("Evidence"),
    Path("Benchmarks"),
    Path("Artifacts"),
    Path("Cache"),
    Path("Logs"),
    Path("Config"),
)


class SetupOutcome(StrEnum):
    """Outcome vocabulary shared by setup checks and evidence."""

    SUPPORTED = "supported"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SetupStatus(StrEnum):
    """Overall setup state; individual checks retain their precise outcome."""

    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SetupCheck:
    """One redacted, actionable setup diagnostic."""

    category: str
    code: str
    outcome: SetupOutcome
    message: str
    details: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "category": self.category,
            "code": self.code,
            "outcome": self.outcome.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SetupRequest:
    """Inputs to setup diagnostics and initialization."""

    data_root: Path
    fixture: Path | None = None
    offline: bool = False
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    provider: ProviderConfig | None = None
    create_missing: bool = False

    def __post_init__(self) -> None:
        if self.min_free_bytes <= 0:
            raise ValueError("min_free_bytes must be positive")


@dataclass(frozen=True, slots=True)
class SetupReport:
    """Canonical first-run report suitable for a saved regression artifact."""

    status: SetupStatus
    request: SetupRequest
    checks: tuple[SetupCheck, ...]
    config_path: Path | None = None

    @property
    def outcome(self) -> SetupOutcome:
        if self.status is SetupStatus.READY:
            return SetupOutcome.SUPPORTED
        if any(check.outcome is SetupOutcome.FAILED for check in self.checks):
            return SetupOutcome.FAILED
        if any(check.outcome is SetupOutcome.UNSUPPORTED for check in self.checks):
            return SetupOutcome.UNSUPPORTED
        if any(check.outcome is SetupOutcome.UNAVAILABLE for check in self.checks):
            return SetupOutcome.UNAVAILABLE
        return SetupOutcome.UNKNOWN

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "modelsurgeon_setup_report",
            "schema_version": SETUP_SCHEMA_VERSION,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "data_root": str(self.request.data_root),
            "offline": self.request.offline,
            "min_free_bytes": self.request.min_free_bytes,
            "fixture": None if self.request.fixture is None else str(self.request.fixture),
            "config_path": None if self.config_path is None else str(self.config_path),
            "checks": [check.to_record() for check in self.checks],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def default_data_root(*, environ: Mapping[str, str] | None = None) -> Path:
    """Return a platform-appropriate user-owned mutable data root."""

    values = os.environ if environ is None else environ
    if sys.platform == "win32":
        base = values.get("LOCALAPPDATA")
        return (
            Path(base) / "ModelSurgeon"
            if base
            else Path.home() / "AppData" / "Local" / "ModelSurgeon"
        )
    base = values.get("XDG_DATA_HOME")
    return (
        Path(base) / "ModelSurgeon"
        if base
        else Path.home() / ".local" / "share" / "ModelSurgeon"
    )


def _check(
    category: str,
    code: str,
    outcome: SetupOutcome,
    message: str,
    **details: object,
) -> SetupCheck:
    return SetupCheck(category, code, outcome, message, details)


def _probe_writable(path: Path) -> str | None:
    """Return a redacted error if a directory cannot accept a temporary file."""

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".modelsurgeon-setup-", dir=path, delete=False
        ) as handle:
            handle.write(b"setup probe\n")
            probe = Path(handle.name)
        probe.unlink()
    except OSError as error:
        return f"data root is not writable: {error.strerror or error.__class__.__name__}"
    return None


def _runtime_check() -> SetupCheck:
    supported_platforms = {"win32", "linux", "darwin"}
    if sys.platform not in supported_platforms:
        return _check(
            "platform_runtime",
            "unsupported_platform",
            SetupOutcome.UNSUPPORTED,
            f"core setup has no declared support for platform {sys.platform!r}",
            platform=sys.platform,
            python=platform.python_version(),
        )
    if sys.version_info < (3, 12):  # noqa: UP036 - report the declared support boundary
        return _check(
            "platform_runtime",
            "unsupported_python",
            SetupOutcome.UNSUPPORTED,
            "ModelSurgeon requires Python 3.12 or newer",
            platform=sys.platform,
            python=platform.python_version(),
            required_python="3.12+",
        )
    return _check(
        "platform_runtime",
        "core_runtime_supported",
        SetupOutcome.SUPPORTED,
        "core CLI/Python runtime is supported; optional model runtimes are checked separately",
        platform=sys.platform,
        python=platform.python_version(),
    )


def _data_root_checks(request: SetupRequest) -> tuple[SetupCheck, SetupCheck]:
    root = request.data_root.expanduser()
    created = False
    if root.exists() and not root.is_dir():
        root_check = _check(
            "data_root",
            "data_root_not_directory",
            SetupOutcome.FAILED,
            "selected data root exists but is not a directory",
            path=str(root),
        )
        layout_check = _check(
            "layout",
            "layout_unavailable",
            SetupOutcome.UNAVAILABLE,
            "layout cannot be checked until the data root is a directory",
            path=str(root),
        )
        return root_check, layout_check
    if not root.exists():
        if request.create_missing:
            try:
                root.mkdir(parents=True, exist_ok=True)
                created = True
            except OSError as error:
                root_check = _check(
                    "data_root",
                    "data_root_create_failed",
                    SetupOutcome.FAILED,
                    f"data root could not be created: {error.strerror or error.__class__.__name__}",
                    path=str(root),
                )
                layout_check = _check(
                    "layout",
                    "layout_unavailable",
                    SetupOutcome.UNAVAILABLE,
                    "layout cannot be created until the data root is available",
                    path=str(root),
                )
                return root_check, layout_check
        else:
            root_check = _check(
                "data_root",
                "missing_data_directory",
                SetupOutcome.UNAVAILABLE,
                "data root is missing; run `modelsurgeon setup init` to create it",
                path=str(root),
            )
            layout_check = _check(
                "layout",
                "missing_layout_directories",
                SetupOutcome.UNAVAILABLE,
                "data directories are not present yet",
                missing=[str(path) for path in LAYOUT_DIRECTORIES],
            )
            return root_check, layout_check

    writable_error = _probe_writable(root)
    if writable_error is not None:
        root_check = _check(
            "data_root",
            "permission_denied",
            SetupOutcome.FAILED,
            writable_error,
            path=str(root),
        )
    else:
        root_check = _check(
            "data_root",
            "data_root_ready",
            SetupOutcome.SUPPORTED,
            "large mutable data has a writable user-owned location",
            path=str(root),
            created=created,
        )

    missing = [path for path in LAYOUT_DIRECTORIES if not (root / path).is_dir()]
    if missing and request.create_missing:
        try:
            for path in missing:
                (root / path).mkdir(parents=True, exist_ok=True)
            missing = [path for path in LAYOUT_DIRECTORIES if not (root / path).is_dir()]
        except OSError as error:
            return root_check, _check(
                "layout",
                "layout_create_failed",
                SetupOutcome.FAILED,
                    "one or more data directories could not be created: "
                    f"{error.strerror or error.__class__.__name__}",
                path=str(root),
            )
    if missing:
        return root_check, _check(
            "layout",
            "missing_layout_directories",
            SetupOutcome.UNAVAILABLE,
            "some first-run directories are missing; initialization will create them",
            missing=[str(path) for path in missing],
        )
    return root_check, _check(
        "layout",
        "layout_ready",
        SetupOutcome.SUPPORTED,
        "Surgeon Tensors, Text LLM, target Models, and run-output directories are separate",
        directories=[str(path) for path in LAYOUT_DIRECTORIES],
    )


def _disk_check(request: SetupRequest) -> SetupCheck:
    location = request.data_root.expanduser()
    probe = location if location.exists() else location.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as error:
        return _check(
            "disk_budget",
            "disk_usage_unavailable",
            SetupOutcome.UNKNOWN,
            f"free-space information is unavailable: {error.strerror or error.__class__.__name__}",
            path=str(probe),
            required_free_bytes=request.min_free_bytes,
        )
    if usage.free < request.min_free_bytes:
        return _check(
            "disk_budget",
            "disk_budget_exceeded",
            SetupOutcome.FAILED,
            "available free space is below the declared setup budget",
            path=str(probe),
            free_bytes=usage.free,
            required_free_bytes=request.min_free_bytes,
        )
    return _check(
        "disk_budget",
        "disk_budget_available",
        SetupOutcome.SUPPORTED,
        "free space meets the declared setup budget; model downloads are not performed",
        path=str(probe),
        free_bytes=usage.free,
        required_free_bytes=request.min_free_bytes,
    )


def _fixture_check(request: SetupRequest) -> SetupCheck:
    if request.fixture is None:
        if request.offline:
            return _check(
                "supported_fixture",
                "offline_fixture_required",
                SetupOutcome.UNSUPPORTED,
                "offline setup requires an existing local supported-fixture manifest",
            )
        return _check(
            "supported_fixture",
            "fixture_not_selected",
            SetupOutcome.UNKNOWN,
            "no local fixture was selected; setup does not download or validate target weights",
        )
    fixture = request.fixture.expanduser()
    if not fixture.is_file():
        return _check(
            "supported_fixture",
            "fixture_missing",
            SetupOutcome.FAILED,
            "selected supported-fixture manifest does not exist",
            path=str(fixture),
        )
    try:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return _check(
            "supported_fixture",
            "fixture_invalid",
            SetupOutcome.FAILED,
            f"supported-fixture manifest could not be read: {error}",
            path=str(fixture),
        )
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return _check(
            "supported_fixture",
            "fixture_schema_unsupported",
            SetupOutcome.UNSUPPORTED,
            "fixture manifest is not the supported schema version 1",
            path=str(fixture),
        )
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        return _check(
            "supported_fixture",
            "fixture_has_no_models",
            SetupOutcome.FAILED,
            "supported-fixture manifest contains no model entries",
            path=str(fixture),
        )
    families = sorted(
        {
            str(item["family"])
            for item in models
            if isinstance(item, Mapping) and "family" in item
        }
    )
    return _check(
        "supported_fixture",
        "fixture_manifest_supported",
        SetupOutcome.SUPPORTED,
        "existing local fixture manifest is recognized; no weights were downloaded",
        path=str(fixture),
        model_count=len(models),
        families=families,
    )


def _provider_check(request: SetupRequest) -> SetupCheck:
    config = request.provider or ProviderConfig()
    if request.offline and config.kind is not ProviderKind.NONE:
        return _check(
            "provider",
            "offline_provider_disabled",
            SetupOutcome.UNSUPPORTED,
            "offline mode permits only the explicit no-LLM path; no provider was contacted",
            kind=config.kind.value,
        )
    diagnostic = provider_diagnostics(config)
    if diagnostic.code == "no_llm":
        return _check(
            "provider",
            "no_llm",
            SetupOutcome.SUPPORTED,
            "no conversational provider is selected; direct CLI/Python paths remain available",
            kind=config.kind.value,
        )
    if diagnostic.code == "missing_api_key":
        return _check(
            "provider",
            diagnostic.code,
            SetupOutcome.UNAVAILABLE,
            diagnostic.message,
            kind=config.kind.value,
        )
    return _check(
        "provider",
        diagnostic.code,
        SetupOutcome.UNSUPPORTED,
        diagnostic.message,
        kind=config.kind.value,
        dependency=diagnostic.dependency,
    )


def diagnose_setup(request: SetupRequest) -> SetupReport:
    """Run non-mutating setup checks unless ``create_missing`` is requested."""

    checks = (
        _runtime_check(),
        *_data_root_checks(request),
        _disk_check(request),
        _fixture_check(request),
        _provider_check(request),
    )
    if any(check.outcome is SetupOutcome.FAILED for check in checks):
        status = SetupStatus.FAILED
    elif all(check.outcome is SetupOutcome.SUPPORTED for check in checks):
        status = SetupStatus.READY
    else:
        status = SetupStatus.NEEDS_ATTENTION
    return SetupReport(status, request, tuple(checks))


def _setup_config_text(request: SetupRequest, *, fixture: Path | None) -> str:
    fixture_value = "none" if fixture is None else str(fixture)
    provider_kind = (request.provider or ProviderConfig()).kind.value
    return (
        "# ModelSurgeon first-run setup manifest. This file contains no secrets.\n"
        f"schema_version = {SETUP_SCHEMA_VERSION}\n"
        f"data_root = {json.dumps(str(request.data_root.expanduser()))}\n"
        f"offline = {str(request.offline).lower()}\n"
        f"fixture_manifest = {json.dumps(fixture_value)}\n"
        f"min_free_bytes = {request.min_free_bytes}\n"
        "\n[provider]\n"
        f"kind = {json.dumps(provider_kind)}\n"
        "\n"
    )


def _user_readme() -> str:
    return """# ModelSurgeon data

This is the user-owned data root for ModelSurgeon.

- `Surgeon Tensors/` contains ModelSurgeon's learned Meta-Surgeon assets.
- `Text LLM/` contains an optional conversational text model only.
- `Models/` contains user-selected target models that ModelSurgeon inspects or edits.
- `Artifacts/`, `Evidence/`, `Campaigns/`, and `Benchmarks/` contain run outputs.
- `Config/modelsurgeon.toml` records this first-run layout and contains no secrets.

Setup does not download models or install a provider. The structured CLI and
Python APIs continue to work with the explicit no-LLM configuration.
"""


def initialize_setup(request: SetupRequest, *, force: bool = False) -> SetupReport:
    """Create the layout and a safe first-run manifest, then return diagnostics."""

    initialized = diagnose_setup(
        SetupRequest(
            data_root=request.data_root,
            fixture=request.fixture,
            offline=request.offline,
            min_free_bytes=request.min_free_bytes,
            provider=request.provider,
            create_missing=True,
        )
    )
    if initialized.status is SetupStatus.FAILED:
        return initialized
    config_path = request.data_root.expanduser() / "Config" / "modelsurgeon.toml"
    readme_path = request.data_root.expanduser() / "README.md"
    try:
        if config_path.exists() and not force:
            config_check = _check(
                "configuration",
                "configuration_exists",
                SetupOutcome.UNAVAILABLE,
                "first-run manifest already exists; pass --force to replace it",
                path=str(config_path),
            )
            checks = (*initialized.checks, config_check)
            return SetupReport(SetupStatus.NEEDS_ATTENTION, request, checks, config_path)
        config_path.write_text(
            _setup_config_text(request, fixture=request.fixture), encoding="utf-8", newline="\n"
        )
        if not readme_path.exists() or force:
            readme_path.write_text(_user_readme(), encoding="utf-8", newline="\n")
    except OSError as error:
        config_check = _check(
            "configuration",
            "configuration_write_failed",
            SetupOutcome.FAILED,
            "first-run files could not be written: "
            f"{error.strerror or error.__class__.__name__}",
            path=str(config_path),
        )
        return SetupReport(
            SetupStatus.FAILED, request, (*initialized.checks, config_check), config_path
        )
    config_check = _check(
        "configuration",
        "configuration_initialized",
        SetupOutcome.SUPPORTED,
        "first-run manifest and user README are present",
        path=str(config_path),
    )
    checks = (*initialized.checks, config_check)
    status = (
        SetupStatus.READY
        if all(check.outcome is SetupOutcome.SUPPORTED for check in checks)
        else SetupStatus.NEEDS_ATTENTION
    )
    return SetupReport(status, request, checks, config_path)


__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "LAYOUT_DIRECTORIES",
    "SETUP_SCHEMA_VERSION",
    "SetupCheck",
    "SetupOutcome",
    "SetupReport",
    "SetupRequest",
    "SetupStatus",
    "default_data_root",
    "diagnose_setup",
    "initialize_setup",
]
