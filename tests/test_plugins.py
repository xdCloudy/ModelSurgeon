"""Capability negotiation and fail-closed plugin boundary tests."""

from __future__ import annotations

import sys

import pytest

from modelsurgeon.plugins import (
    PluginCapabilityCard,
    PluginCatalog,
    PluginError,
    PluginExecutionMode,
    PluginKind,
    PluginOutcome,
    PluginRequest,
    PluginResourceBudget,
    invoke_in_process,
    invoke_subprocess,
    negotiate_plugin,
)


def _card(**overrides: object) -> PluginCapabilityCard:
    values: dict[str, object] = {
        "name": "fixture-evaluator",
        "kind": PluginKind.EVALUATOR,
        "plugin_version": "1.2.0",
        "api_version": "1.0",
        "capabilities": ("quality", "runtime"),
        "license_id": "apache-2.0",
        "dependencies": (),
        "trust_modes": (PluginExecutionMode.SUBPROCESS, PluginExecutionMode.TRUSTED_IN_PROCESS),
    }
    values.update(overrides)
    return PluginCapabilityCard(**values)  # type: ignore[arg-type]


def _request(**overrides: object) -> PluginRequest:
    values: dict[str, object] = {
        "request_id": "request-1",
        "kind": PluginKind.EVALUATOR,
        "required_capabilities": ("quality",),
        "payload": {"metric": "accuracy"},
        "source_artifact_digest": "sha256:" + "1" * 64,
        "config_digest": "sha256:" + "2" * 64,
        "seed": 7,
        "budget": PluginResourceBudget(max_wall_seconds=5),
    }
    values.update(overrides)
    return PluginRequest(**values)  # type: ignore[arg-type]


def test_negotiation_fails_closed_for_version_capability_and_dependencies() -> None:
    card = _card(dependencies=("optional-runtime",))
    request = _request(required_capabilities=("missing", "quality"))
    report = negotiate_plugin(card, request)
    assert report.outcome is PluginOutcome.UNSUPPORTED
    assert report.missing_capabilities == ("missing",)
    assert report.missing_dependencies == ("optional-runtime",)

    incompatible = negotiate_plugin(_card(api_version="2.0"), _request())
    assert incompatible.outcome is PluginOutcome.UNSUPPORTED


def test_catalog_rejects_duplicate_names_and_plans_without_importing_code() -> None:
    with pytest.raises(PluginError, match="unique"):
        PluginCatalog((_card(), _card()))
    catalog = PluginCatalog((_card(),))
    plan = catalog.plan(_request())
    assert plan[0].outcome is PluginOutcome.SUPPORTED
    assert catalog.cards[0].plugin_id.startswith("plugin_")


class _FakePlugin:
    def run(self, request: PluginRequest) -> dict[str, object]:
        return {
            "outcome": "supported",
            "reason": request.payload["metric"],
            "payload": {"value": 0.9},
            "evidence": {"source_artifact_digest": request.source_artifact_digest},
        }


def test_in_process_requires_explicit_trusted_capability_and_primitive_request() -> None:
    card = _card()
    request = _request()
    blocked = invoke_in_process(card, _FakePlugin(), request)
    assert blocked.outcome is PluginOutcome.UNSUPPORTED
    accepted = invoke_in_process(card, _FakePlugin(), request, trusted=True)
    assert accepted.outcome is PluginOutcome.SUPPORTED
    assert accepted.payload["value"] == 0.9


def test_subprocess_result_is_bounded_and_malformed_output_is_retained() -> None:
    card = _card(trust_modes=(PluginExecutionMode.SUBPROCESS,))
    request = _request()
    script = (
        "import json,sys; request=json.loads(sys.stdin.read()); "
        "print(json.dumps({'outcome':'supported','reason':request['request_id']}))"
    )
    result = invoke_subprocess(card, (sys.executable, "-c", script), request)
    assert result.outcome is PluginOutcome.SUPPORTED

    malformed = invoke_subprocess(
        card,
        (sys.executable, "-c", "print('not-json')"),
        request,
    )
    assert malformed.outcome is PluginOutcome.MALFORMED_OUTPUT


def test_plugin_output_budget_is_fail_closed() -> None:
    card = _card(resource_budget=PluginResourceBudget(max_stdout_bytes=4))
    request = _request()
    result = invoke_subprocess(
        card,
        (sys.executable, "-c", "print('12345')"),
        request,
    )
    assert result.outcome is PluginOutcome.MALFORMED_OUTPUT
