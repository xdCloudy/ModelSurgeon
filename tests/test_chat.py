"""Focused tests for the bounded chat command and session bootstrap."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from modelsurgeon.adapters.gguf import GGUFValueType
from modelsurgeon.cli.app import app
from modelsurgeon.conversation import (
    CancellationToken,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    InterpretIntentRequest,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderKind,
    ProviderLimits,
    ProviderModelIdentity,
    ProviderOutcome,
    ProviderResult,
    ProviderStreamEvent,
    SourceSpan,
    bootstrap_chat_session,
    result_from_raw_output,
)
from modelsurgeon.conversation.provider import ProviderRequest, TextModelProvider
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path, architecture: str = "llama") -> None:
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
    data.extend(_string("general.architecture"))
    data.extend(struct.pack("<I", GGUFValueType.STRING))
    data.extend(_string(architecture))
    data.extend(b"\0" * ((-len(data)) % 32))
    path.write_bytes(data)


def _intent(request: str) -> IntentRecord:
    span = SourceSpan("request-span", 0, len(request), request)
    fields = (
        IntentField(
            "constraint.quality",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.95,
                "unit": "ratio",
            },
            "ratio",
            0.99,
            (span.span_id,),
            True,
        ),
        IntentField(
            "objective.latency",
            {
                "kind": "preference",
                "metric": "latency",
                "direction": "minimize",
                "unit": "milliseconds",
                "normalization": "identity",
            },
            "milliseconds",
            0.99,
            (span.span_id,),
        ),
    )
    spec = ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    ).to_record()
    return IntentRecord(
        request,
        (span,),
        fields,
        (),
        (
            InterpretationStep(
                "normalize-request",
                "normalize",
                (span.span_id,),
                tuple(item.field_id for item in fields),
                0.99,
                ("evidence:request",),
            ),
        ),
        IntentProvenance(
            "request-revision-1",
            "fixture-provider",
            "fixture-provider-v1",
            "chat-test-v1",
            ("evidence:request",),
        ),
        IntentOutcome.EXECUTABLE,
        ("fixture interpretation",),
        spec,
    )


class _FixtureProvider(TextModelProvider):
    identity = ProviderModelIdentity("fixture-provider", "fixture-model", "fixture-model-v1")
    capability_card = ProviderCapabilityCard(
        identity,
        ProviderKind.LOCAL,
        "fixture-provider-v1",
        (ProviderCapability.INTERPRET_INTENT, ProviderCapability.STRUCTURED_OUTPUT),
        ProviderLimits(2048, 1024, 3072),
        ("modelsurgeon.provider-output:1",),
    )

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def call(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> ProviderResult:
        cancellation.raise_if_cancelled()
        assert isinstance(request, InterpretIntentRequest)
        return result_from_raw_output(
            self,
            request,
            {
                "operation": "interpret_intent",
                "intent": _intent(request.original_request).to_record(),
            },
        )

    def stream(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> Any:
        if False:
            yield ProviderStreamEvent(request.request_id, 0, "")
        return

    def cancel(self, request_id: str) -> bool:
        return False


def test_chat_bootstrap_rejects_missing_and_unsupported_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist") as missing:
        bootstrap_chat_session(tmp_path / "missing.gguf")
    assert missing.value.code == "model_missing"

    unsupported = tmp_path / "model.safetensors"
    unsupported.write_bytes(b"fixture")
    with pytest.raises(ValueError, match=r"only \.gguf") as wrong_format:
        bootstrap_chat_session(unsupported)
    assert wrong_format.value.code == "unsupported_model_format"


def test_chat_bootstrap_is_deterministic_and_validates_before_interaction(
    tmp_path: Path,
) -> None:
    model = tmp_path / "provider.gguf"
    _gguf(model)
    providers: list[_FixtureProvider] = []

    def factory(_: object) -> TextModelProvider:
        provider = _FixtureProvider()
        providers.append(provider)
        return provider

    first = bootstrap_chat_session(
        model,
        runtime_revision="fixture-runtime-v1",
        provider_factory=factory,  # type: ignore[arg-type]
    )
    second = bootstrap_chat_session(
        model,
        runtime_revision="fixture-runtime-v1",
        provider_factory=factory,  # type: ignore[arg-type]
    )
    assert first.bootstrap.session_id == second.bootstrap.session_id
    assert first.bootstrap.model_revision is not None
    assert first.bootstrap.runtime_revision == "fixture-runtime-v1"
    assert providers[0].started
    turn = first.interpret("retain quality and reduce latency")
    assert turn.outcome == "executable"
    assert turn.policy_decision is not None
    assert turn.to_record()["execution"] == "not_requested"
    first.close()
    second.close()
    assert all(provider.closed for provider in providers)


def test_chat_preserves_provider_absence_and_cancellation() -> None:
    session = bootstrap_chat_session(Path("unused"), provider_kind=ProviderKind.NONE)
    unsupported = session.interpret("interpret this")
    assert unsupported.provider_result.outcome is ProviderOutcome.UNSUPPORTED
    assert unsupported.policy_decision is None

    token = CancellationToken()
    token.cancel()
    cancelled = session.interpret("interpret this", cancellation=token)
    assert cancelled.provider_result.outcome is ProviderOutcome.CANCELLED
    session.close()


def test_chat_command_offline_no_provider_smoke_and_invalid_path() -> None:
    runner = CliRunner()
    smoke = runner.invoke(
        app,
        ["chat", "unused.gguf", "--provider", "none", "--request", "hello", "--json"],
        color=False,
    )
    assert smoke.exit_code == 0, smoke.output
    records = [json.loads(line) for line in smoke.stdout.splitlines()]
    assert records[0]["record_type"] == "chat_session_bootstrap"
    assert records[1]["outcome"] == "unsupported"
    assert records[1]["provider_result"]["outcome"] == "unsupported"

    invalid = runner.invoke(app, ["chat", "missing.gguf", "--json"], color=False)
    assert invalid.exit_code == 2
    error = json.loads(invalid.stdout)
    assert error["record_type"] == "error"
    assert "does not exist" in error["message"]
