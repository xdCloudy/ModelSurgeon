"""Capability-negotiated, fail-closed plugin contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import IO, Protocol

from modelsurgeon.experiments.identity import canonical_identity_json

PLUGIN_SCHEMA_VERSION = 1
_MAX_PLUGIN_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_PLUGIN_REQUEST_ITEMS = 100_000
_MAX_PLUGIN_REQUEST_DEPTH = 64


class PluginError(ValueError):
    """Raised when a plugin contract is malformed or unsafe to invoke."""


class PluginKind(StrEnum):
    EVALUATOR = "evaluator"
    RUNTIME = "runtime"
    TRANSFORMATION = "transformation"
    SEARCH_POLICY = "search_policy"
    OBJECTIVE = "objective"
    REGISTRY_PROVIDER = "registry_provider"


class PluginOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    CRASH = "crash"
    MALFORMED_OUTPUT = "malformed_output"


class PluginExecutionMode(StrEnum):
    SUBPROCESS = "subprocess"
    TRUSTED_IN_PROCESS = "trusted_in_process"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PluginError(f"{label} must be non-empty text")
    return value


def _canonical_tuple(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))) or any(not value.strip() for value in result):
        raise PluginError(f"{label} must be sorted, unique, and non-empty")
    return result


def _major(version: str) -> int:
    value = version.split(".", 1)[0]
    if not value.isdigit():
        raise PluginError("API versions must start with a numeric major")
    return int(value)


def _json_string_bytes(value: str) -> int:
    size = 2
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in {"\b", "\f", "\n", "\r", "\t"}:
            size += 2
        elif codepoint < 0x20:
            size += 6
        elif codepoint < 0x80:
            size += 1
        else:
            try:
                size += len(character.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise PluginError("plugin request contains invalid Unicode") from error
        if size > _MAX_PLUGIN_REQUEST_BYTES:
            raise PluginError("plugin request exceeds the 4 MiB serialized size limit")
    return size


def _bounded_json_copy(value: object, budget: list[int], depth: int = 0) -> object:
    """Validate and copy a JSON value while accounting its encoded size."""
    budget[1] += 1
    if budget[1] > _MAX_PLUGIN_REQUEST_ITEMS:
        raise PluginError("plugin request exceeds the JSON item limit")
    if depth > _MAX_PLUGIN_REQUEST_DEPTH:
        raise PluginError("plugin request exceeds the JSON nesting limit")

    if value is None:
        size = 4
        result: object = None
    elif isinstance(value, bool):
        size = 4 if value else 5
        result = value
    elif isinstance(value, int):
        if value.bit_length() > _MAX_PLUGIN_REQUEST_BYTES * 4:
            raise PluginError("plugin request exceeds the 4 MiB serialized size limit")
        try:
            size = len(str(value))
        except ValueError as error:
            raise PluginError("plugin integer exceeds the serialized size limit") from error
        result = value
    elif isinstance(value, float):
        try:
            size = len(json.dumps(value, allow_nan=False))
        except ValueError as error:
            raise PluginError("plugin request contains a non-finite number") from error
        result = value
    elif isinstance(value, str):
        size = _json_string_bytes(value)
        result = value
    elif isinstance(value, Mapping):
        if len(value) > _MAX_PLUGIN_REQUEST_ITEMS // 2:
            raise PluginError("plugin request exceeds the JSON item limit")
        size = 2
        item_count = 0
        copied: dict[str, object] = {}
        try:
            for key in value:
                if not isinstance(key, str) or not key:
                    raise PluginError("plugin request objects require non-empty string keys")
                budget[1] += 1
                if budget[1] > _MAX_PLUGIN_REQUEST_ITEMS:
                    raise PluginError("plugin request exceeds the JSON item limit")
                if item_count:
                    size += 1
                size += _json_string_bytes(key) + 1
                copied[key] = _bounded_json_copy(value[key], budget, depth + 1)
                item_count += 1
        except PluginError:
            raise
        except (KeyError, RuntimeError, ValueError) as error:
            raise PluginError("plugin request changed while being validated") from error
        result = copied
    elif isinstance(value, (list, tuple)):
        if len(value) > _MAX_PLUGIN_REQUEST_ITEMS:
            raise PluginError("plugin request exceeds the JSON item limit")
        size = 2
        item_count = 0
        copied_list: list[object] = []
        try:
            for item in value:
                if item_count:
                    size += 1
                copied_list.append(_bounded_json_copy(item, budget, depth + 1))
                item_count += 1
        except PluginError:
            raise
        except (RuntimeError, ValueError) as error:
            raise PluginError("plugin request changed while being validated") from error
        result = copied_list
    else:
        raise PluginError(
            f"plugin request value type {type(value).__name__} is not JSON-compatible"
        )

    budget[0] += size
    if budget[0] > _MAX_PLUGIN_REQUEST_BYTES:
        raise PluginError("plugin request exceeds the 4 MiB serialized size limit")
    return result


def _bounded_plugin_request_bytes(request: PluginRequest) -> bytes:
    record = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "request_id": request.request_id,
        "kind": request.kind.value,
        "required_capabilities": request.required_capabilities,
        "payload": request.payload,
        "source_artifact_digest": request.source_artifact_digest,
        "config_digest": request.config_digest,
        "seed": request.seed,
        "budget": request.budget.to_record(),
    }
    safe_record = _bounded_json_copy(record, [0, 0])
    encoded = canonical_identity_json(safe_record).encode("utf-8") + b"\n"
    if len(encoded) > _MAX_PLUGIN_REQUEST_BYTES:
        raise PluginError("plugin request exceeds the 4 MiB serialized size limit")
    return encoded


@dataclass(frozen=True, slots=True)
class PluginResourceBudget:
    max_wall_seconds: float = 60.0
    max_stdout_bytes: int = 1 << 20
    max_stderr_bytes: int = 1 << 20
    max_artifact_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        if self.max_wall_seconds <= 0:
            raise PluginError("plugin wall-time budget must be positive")
        if any(
            isinstance(value, bool) or value <= 0
            for value in (
                self.max_stdout_bytes,
                self.max_stderr_bytes,
                self.max_artifact_bytes,
            )
        ):
            raise PluginError("plugin byte budgets must be positive integers")

    def to_record(self) -> dict[str, object]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
        }


@dataclass(frozen=True, slots=True)
class PluginCapabilityCard:
    """Static dependency, trust, and capability declaration."""

    name: str
    kind: PluginKind
    plugin_version: str
    api_version: str
    capabilities: tuple[str, ...]
    license_id: str
    dependencies: tuple[str, ...] = ()
    trust_modes: tuple[PluginExecutionMode, ...] = (PluginExecutionMode.SUBPROCESS,)
    resource_budget: PluginResourceBudget = PluginResourceBudget()

    def __post_init__(self) -> None:
        _text(self.name, "plugin name")
        _text(self.plugin_version, "plugin version")
        _major(self.api_version)
        _canonical_tuple(self.capabilities, "plugin capabilities")
        _text(self.license_id, "plugin license")
        _canonical_tuple(self.dependencies, "plugin dependencies")
        if not self.trust_modes:
            raise PluginError("plugin must declare at least one trust mode")
        if len(set(self.trust_modes)) != len(self.trust_modes):
            raise PluginError("plugin trust modes must be unique")

    @property
    def plugin_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode()
        ).hexdigest()
        return f"plugin_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": PLUGIN_SCHEMA_VERSION,
            "name": self.name,
            "kind": self.kind.value,
            "plugin_version": self.plugin_version,
            "api_version": self.api_version,
            "capabilities": list(self.capabilities),
            "license_id": self.license_id,
            "dependencies": list(self.dependencies),
            "trust_modes": [item.value for item in self.trust_modes],
            "resource_budget": self.resource_budget.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["plugin_id"] = self.plugin_id
        return record


@dataclass(frozen=True, slots=True)
class PluginRequest:
    """Primitive-only request; source objects are represented by identities."""

    request_id: str
    kind: PluginKind
    required_capabilities: tuple[str, ...]
    payload: Mapping[str, object]
    source_artifact_digest: str | None
    config_digest: str
    seed: int
    budget: PluginResourceBudget

    def __post_init__(self) -> None:
        _text(self.request_id, "plugin request ID")
        _canonical_tuple(self.required_capabilities, "required capabilities")
        _text(self.config_digest, "plugin request config digest")
        if self.seed < 0:
            raise PluginError("plugin request seed cannot be negative")
        if self.source_artifact_digest is not None:
            _text(self.source_artifact_digest, "source artifact digest")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PLUGIN_SCHEMA_VERSION,
            "request_id": self.request_id,
            "kind": self.kind.value,
            "required_capabilities": list(self.required_capabilities),
            "payload": dict(self.payload),
            "source_artifact_digest": self.source_artifact_digest,
            "config_digest": self.config_digest,
            "seed": self.seed,
            "budget": self.budget.to_record(),
        }

    def canonical_json(self) -> str:
        return canonical_identity_json(self.to_record())


@dataclass(frozen=True, slots=True)
class PluginResult:
    plugin_id: str
    outcome: PluginOutcome
    reason: str
    payload: Mapping[str, object] = field(default_factory=dict)
    evidence: Mapping[str, object] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        _text(self.plugin_id, "plugin result ID")
        _text(self.reason, "plugin result reason")
        if self.wall_seconds < 0:
            raise PluginError("plugin result wall time cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PLUGIN_SCHEMA_VERSION,
            "plugin_id": self.plugin_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "payload": dict(self.payload),
            "evidence": dict(self.evidence),
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class PluginCompatibility:
    plugin_id: str
    outcome: PluginOutcome
    missing_capabilities: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "plugin_id": self.plugin_id,
            "outcome": self.outcome.value,
            "missing_capabilities": list(self.missing_capabilities),
            "missing_dependencies": list(self.missing_dependencies),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    """Metadata-only entry-point descriptor; it never imports plugin code."""

    name: str
    value: str
    group: str
    distribution: str | None = None
    distribution_version: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "group": self.group,
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
        }


def negotiate_plugin(
    card: PluginCapabilityCard,
    request: PluginRequest,
    *,
    available_dependencies: Iterable[str] = (),
    core_api_version: str = "1.0",
) -> PluginCompatibility:
    """Fail closed for kind, API major, capability, or dependency mismatches."""

    missing_capabilities = tuple(
        sorted(set(request.required_capabilities) - set(card.capabilities))
    )
    missing_dependencies = tuple(sorted(set(card.dependencies) - set(available_dependencies)))
    if card.kind is not request.kind:
        return PluginCompatibility(
            card.plugin_id, PluginOutcome.UNSUPPORTED, (), (), "plugin kind mismatch"
        )
    if _major(card.api_version) != _major(core_api_version):
        return PluginCompatibility(
            card.plugin_id,
            PluginOutcome.UNSUPPORTED,
            missing_capabilities,
            missing_dependencies,
            "plugin API major version is incompatible",
        )
    if missing_capabilities or missing_dependencies:
        return PluginCompatibility(
            card.plugin_id,
            PluginOutcome.UNSUPPORTED,
            missing_capabilities,
            missing_dependencies,
            "required capabilities or dependencies are unavailable",
        )
    return PluginCompatibility(card.plugin_id, PluginOutcome.SUPPORTED, (), (), "compatible")


class PluginImplementation(Protocol):
    def run(self, request: PluginRequest) -> PluginResult | Mapping[str, object]: ...


class PluginCatalog:
    """Explicitly allowlisted capability cards and metadata-only discovery."""

    def __init__(self, cards: Sequence[PluginCapabilityCard] = ()) -> None:
        names = [card.name for card in cards]
        if len(names) != len(set(names)):
            raise PluginError("plugin names must be unique")
        self.cards = tuple(sorted(cards, key=lambda card: card.name))

    @staticmethod
    def discover_entry_points(group: str = "modelsurgeon.plugins") -> tuple[PluginDescriptor, ...]:
        entries = importlib.metadata.entry_points()
        selected = entries.select(group=group)
        descriptors: list[PluginDescriptor] = []
        for entry in selected:
            distribution = entry.dist
            descriptors.append(
                PluginDescriptor(
                    entry.name,
                    entry.value,
                    group,
                    None if distribution is None else distribution.name,
                    None if distribution is None else distribution.version,
                )
            )
        names = [item.name for item in descriptors]
        if len(names) != len(set(names)):
            raise PluginError("discovered plugin names are duplicated")
        return tuple(sorted(descriptors, key=lambda item: item.name))

    def plan(
        self,
        request: PluginRequest,
        *,
        available_dependencies: Iterable[str] = (),
        allowlist: Iterable[str] = (),
    ) -> tuple[PluginCompatibility, ...]:
        allowed = set(allowlist)
        cards = [card for card in self.cards if not allowed or card.name in allowed]
        return tuple(
            negotiate_plugin(card, request, available_dependencies=available_dependencies)
            for card in cards
            if card.kind is request.kind
        )

    def card(self, name: str) -> PluginCapabilityCard:
        for card in self.cards:
            if card.name == name:
                return card
        raise PluginError(f"plugin is not allowlisted: {name}")


def _result_from_mapping(value: object, plugin_id: str) -> PluginResult:
    if not isinstance(value, Mapping):
        raise PluginError("plugin output must be a JSON object")
    payload = value.get("payload", {})
    evidence = value.get("evidence", {})
    if not isinstance(payload, Mapping) or not isinstance(evidence, Mapping):
        raise PluginError("plugin payload and evidence must be JSON objects")
    try:
        outcome = PluginOutcome(str(value["outcome"]))
        reason = _text(value["reason"], "plugin result reason")
        wall_seconds = float(value.get("wall_seconds", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise PluginError("plugin output is missing a valid outcome or reason") from error
    return PluginResult(plugin_id, outcome, reason, dict(payload), dict(evidence), wall_seconds)


def _terminate_plugin_process(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        killpg = getattr(os, "killpg", None)
        terminate_signal = getattr(signal, "SIGTERM", None)
        kill_signal = getattr(signal, "SIGKILL", terminate_signal)
        try:
            if killpg is None or terminate_signal is None:
                raise OSError("process-group termination is unavailable")
            killpg(process.pid, terminate_signal)
        except OSError:
            process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.25)
        try:
            if killpg is None or kill_signal is None:
                raise OSError("process-group termination is unavailable")
            killpg(process.pid, kill_signal)
        except OSError:
            if process.poll() is None:
                process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0.25)
    if process.poll() is None:
        process.kill()
        process.wait()


def _drain_plugin_stream(
    stream: IO[bytes],
    output: bytearray,
    byte_limit: int,
    exceeded: threading.Event,
) -> None:
    while chunk := stream.read(64 * 1024):
        remaining = byte_limit + 1 - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(output) > byte_limit:
            exceeded.set()


def invoke_in_process(
    card: PluginCapabilityCard,
    implementation: PluginImplementation,
    request: PluginRequest,
    *,
    trusted: bool = False,
    available_dependencies: Iterable[str] = (),
) -> PluginResult:
    """Invoke only an explicitly trusted implementation with primitive requests."""

    compatibility = negotiate_plugin(
        card, request, available_dependencies=available_dependencies
    )
    if compatibility.outcome is not PluginOutcome.SUPPORTED:
        return PluginResult(card.plugin_id, compatibility.outcome, compatibility.detail)
    if not trusted or PluginExecutionMode.TRUSTED_IN_PROCESS not in card.trust_modes:
        return PluginResult(
            card.plugin_id,
            PluginOutcome.UNSUPPORTED,
            "in-process execution requires explicit trusted capability",
        )
    started = time.monotonic()
    try:
        result = implementation.run(request)
        parsed = (
            result
            if isinstance(result, PluginResult)
            else _result_from_mapping(result, card.plugin_id)
        )
    except Exception as error:  # plugin failures are data at the boundary
        return PluginResult(
            card.plugin_id, PluginOutcome.FAILED, f"plugin raised {type(error).__name__}"
        )
    return PluginResult(
        parsed.plugin_id,
        parsed.outcome,
        parsed.reason,
        parsed.payload,
        parsed.evidence,
        time.monotonic() - started,
    )


def invoke_subprocess(
    card: PluginCapabilityCard,
    command: Sequence[str],
    request: PluginRequest,
    *,
    available_dependencies: Iterable[str] = (),
) -> PluginResult:
    """Invoke a plugin over canonical JSON with wall/output bounds."""

    compatibility = negotiate_plugin(
        card, request, available_dependencies=available_dependencies
    )
    if compatibility.outcome is not PluginOutcome.SUPPORTED:
        return PluginResult(card.plugin_id, compatibility.outcome, compatibility.detail)
    if PluginExecutionMode.SUBPROCESS not in card.trust_modes:
        return PluginResult(
            card.plugin_id, PluginOutcome.UNSUPPORTED, "subprocess mode is not declared"
        )
    if not command or any(not value.strip() for value in command):
        raise PluginError("subprocess plugin command must be non-empty")
    started = time.monotonic()
    stdout_limit = min(request.budget.max_stdout_bytes, card.resource_budget.max_stdout_bytes)
    stderr_limit = min(request.budget.max_stderr_bytes, card.resource_budget.max_stderr_bytes)
    timeout_seconds = min(request.budget.max_wall_seconds, card.resource_budget.max_wall_seconds)
    request_bytes = _bounded_plugin_request_bytes(request)
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            ),
        )
    except OSError as error:
        return PluginResult(
            card.plugin_id, PluginOutcome.CRASH, f"plugin process could not start: {error}"
        )

    stdin = process.stdin
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    assert stdin is not None
    assert stdout_stream is not None
    assert stderr_stream is not None
    stdout = bytearray()
    stderr = bytearray()
    output_exceeded = threading.Event()
    stdout_reader = threading.Thread(
        target=_drain_plugin_stream,
        args=(stdout_stream, stdout, stdout_limit, output_exceeded),
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=_drain_plugin_stream,
        args=(stderr_stream, stderr, stderr_limit, output_exceeded),
        daemon=True,
    )

    def write_request() -> None:
        try:
            stdin.write(request_bytes)
            stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    stdin_writer = threading.Thread(target=write_request, daemon=True)
    workers = (stdout_reader, stderr_reader, stdin_writer)
    for worker in workers:
        worker.start()

    deadline = started + timeout_seconds
    timed_out = False
    while True:
        if output_exceeded.is_set():
            _terminate_plugin_process(process)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_plugin_process(process)
            break
        if process.poll() is not None and not any(worker.is_alive() for worker in workers):
            break
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    for worker in workers:
        worker.join(timeout=0.5)
    for stream in (stdin, stdout_stream, stderr_stream):
        with suppress(OSError, ValueError):
            stream.close()

    elapsed = time.monotonic() - started
    if timed_out:
        return PluginResult(
            card.plugin_id,
            PluginOutcome.TIMEOUT,
            "plugin exceeded wall-time budget",
            wall_seconds=elapsed,
        )
    if output_exceeded.is_set():
        return PluginResult(
            card.plugin_id,
            PluginOutcome.MALFORMED_OUTPUT,
            "plugin output exceeded byte budget",
            wall_seconds=elapsed,
        )
    if process.returncode != 0:
        return PluginResult(
            card.plugin_id,
            PluginOutcome.CRASH,
            "plugin process exited unsuccessfully",
            wall_seconds=elapsed,
        )
    try:
        parsed = _result_from_mapping(json.loads(stdout), card.plugin_id)
    except (json.JSONDecodeError, UnicodeDecodeError, PluginError):
        return PluginResult(
            card.plugin_id,
            PluginOutcome.MALFORMED_OUTPUT,
            "plugin stdout was not a valid result",
            wall_seconds=elapsed,
        )
    return PluginResult(
        parsed.plugin_id,
        parsed.outcome,
        parsed.reason,
        parsed.payload,
        parsed.evidence,
        elapsed,
    )


__all__ = [
    "PLUGIN_SCHEMA_VERSION",
    "PluginCapabilityCard",
    "PluginCatalog",
    "PluginCompatibility",
    "PluginDescriptor",
    "PluginError",
    "PluginExecutionMode",
    "PluginImplementation",
    "PluginKind",
    "PluginOutcome",
    "PluginRequest",
    "PluginResourceBudget",
    "PluginResult",
    "invoke_in_process",
    "invoke_subprocess",
    "negotiate_plugin",
]
