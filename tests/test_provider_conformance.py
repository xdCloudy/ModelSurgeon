"""Shared provider capability and failure conformance tests."""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from modelsurgeon.adapters.gguf import GGUFValueType
from modelsurgeon.conversation import (
    PROVIDER_CONFORMANCE_MATRIX,
    CancellationToken,
    CompatibleEndpointProvider,
    EndpointConfig,
    EndpointHttpRequest,
    EndpointHttpResponse,
    ExplanationProviderOutput,
    ExplanationRequest,
    HostedProviderAdapter,
    LocalGGUFProvider,
    LocalGGUFProviderConfig,
    NullTextModelProvider,
    ProviderBudget,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderConformanceCapability,
    ProviderFailureCode,
    ProviderKind,
    ProviderLimits,
    ProviderModelIdentity,
    ProviderOutcome,
    ProviderProvenance,
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    RetryPolicy,
    invoke_provider,
    request_digest,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path) -> None:
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
    data.extend(_string("general.architecture"))
    data.extend(struct.pack("<I", GGUFValueType.STRING))
    data.extend(_string("llama"))
    data.extend(b"\0" * ((-len(data)) % 32))
    path.write_bytes(data)


class _Runtime:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def tokenize(self, prompt: bytes, *, add_bos: bool) -> list[int]:
        return list(range(max(1, (len(prompt) + 3) // 4)))

    def create_chat_completion(
        self, *, messages: list[dict[str, str]], **_: object
    ) -> dict[str, Any]:
        if self.mode == "slow":
            time.sleep(0.05)
        prompt = messages[0]["content"]
        encoded = prompt.split("REQUEST_JSON_BEGIN\n", 1)[1].split(
            "\nREQUEST_JSON_END", 1
        )[0]
        request = json.loads(encoded)
        if self.mode == "refusal":
            content: object = {"operation": request["operation"], "execute": "delete"}
        else:
            content = {
                "operation": request["operation"],
                "text": "measured evidence only",
                "evidence_refs": [],
            }
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    def close(self) -> None:
        return None


@dataclass
class _Transport:
    config: EndpointConfig
    mode: str
    requests: list[EndpointHttpRequest] = field(default_factory=list)

    def request(self, request: EndpointHttpRequest) -> EndpointHttpResponse:
        self.requests.append(request)
        if request.method == "GET":
            return _response(
                {
                    "object": "model",
                    "id": self.config.model_id,
                    "modelsurgeon": {
                        "protocol_version": 1,
                        "provider_id": self.config.provider_id,
                        "model_id": self.config.model_id,
                        "model_revision": self.config.model_revision,
                        "capabilities": [
                            "clarify_intent",
                            "explain_evidence",
                            "interpret_intent",
                            "structured_output",
                        ],
                        "limits": self.config.limits.to_record(),
                        "structured_output_schemas": ["modelsurgeon.provider_output.v1"],
                    },
                }
            )
        if self.mode == "slow":
            time.sleep(0.05)
        content = (
            {"operation": "explain_evidence", "execute": "delete"}
            if self.mode == "refusal"
            else {
                "operation": "explain_evidence",
                "text": "measured evidence only",
                "evidence_refs": [],
            }
        )
        return _response(
            {
                "id": "fixture-response",
                "object": "chat.completion",
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


def _response(payload: object) -> EndpointHttpResponse:
    return EndpointHttpResponse(200, {}, json.dumps(payload).encode())


def _config(kind: ProviderKind) -> EndpointConfig:
    return EndpointConfig(
        "fixture-endpoint",
        "https://provider.example/v1",
        "fixture-provider",
        "fixture-model",
        "revision-1",
        kind,
        limits=ProviderLimits(4096, 1024, 8192, max_wall_seconds=1.0),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.0),
    )


def _provider(kind: ProviderKind, tmp_path: Path, mode: str):
    if kind is ProviderKind.NONE:
        return NullTextModelProvider()
    if kind is ProviderKind.LOCAL:
        path = tmp_path / "fixture.gguf"
        _gguf(path)
        return LocalGGUFProvider(
            LocalGGUFProviderConfig(
                path,
                "fixture-revision-1",
                "fixture-runtime-1",
                max_input_tokens=512,
                max_output_tokens=128,
                max_context_tokens=640,
                max_memory_bytes=8 * 1024 * 1024,
                max_wall_seconds=1.0,
                n_batch=1,
            ),
            runtime_factory=lambda **_: _Runtime(mode),
            runtime_version="fixture-runtime-1",
        )
    config = _config(kind)
    provider_type = (
        CompatibleEndpointProvider
        if kind is ProviderKind.COMPATIBLE_ENDPOINT
        else HostedProviderAdapter
    )
    provider = provider_type(config, transport=_Transport(config, mode))
    provider.start()
    return provider


def _request(request_id: str = "conformance-1", budget: ProviderBudget | None = None):
    return ExplanationRequest(
        request_id,
        ({"record_type": "evidence", "value": 1},),
        budget=budget or ProviderBudget(max_wall_seconds=0.5),
    )


@pytest.mark.parametrize("kind", tuple(ProviderKind))
def test_matrix_is_complete_and_docs_are_generated(kind: ProviderKind) -> None:
    cells = [cell for cell in PROVIDER_CONFORMANCE_MATRIX.cells if cell.kind is kind]
    assert {cell.capability for cell in cells} == set(ProviderConformanceCapability)
    documentation = (
        Path(__file__).parents[1] / "docs" / "testing" / "provider-conformance.md"
    ).read_text(encoding="utf-8")
    generated = documentation.split("<!-- BEGIN GENERATED MATRIX -->\n", 1)[1].split(
        "\n<!-- END GENERATED MATRIX -->", 1
    )[0]
    assert generated == PROVIDER_CONFORMANCE_MATRIX.markdown_table()


@pytest.mark.parametrize("kind", tuple(ProviderKind))
def test_structured_output_and_provenance_are_shared_contracts(
    kind: ProviderKind, tmp_path: Path
) -> None:
    provider = _provider(kind, tmp_path, "success")
    request = _request()
    result = invoke_provider(provider, request)
    if kind is ProviderKind.NONE:
        assert result.outcome is ProviderOutcome.UNSUPPORTED
    else:
        assert result.outcome is ProviderOutcome.SUPPORTED
        assert result.output is not None
        assert result.provenance.response_digest is not None
    assert result.provenance.provider_id == provider.identity.provider_id
    assert result.provenance.request_digest == request_digest(request)
    assert result.canonical_json() == result.canonical_json()


@pytest.mark.parametrize("kind", tuple(ProviderKind))
def test_refusals_are_retained_and_never_supported(kind: ProviderKind, tmp_path: Path) -> None:
    result = invoke_provider(_provider(kind, tmp_path, "refusal"), _request("refusal"))
    assert result.outcome is not ProviderOutcome.SUPPORTED
    assert result.output is None
    assert result.failure is not None
    assert result.failure.request_id == result.request_id
    expected = (
        ProviderOutcome.UNSUPPORTED
        if kind is ProviderKind.NONE
        else ProviderOutcome.MALFORMED_OUTPUT
    )
    assert result.outcome is expected


@pytest.mark.parametrize("kind", tuple(ProviderKind))
def test_timeout_and_cancellation_are_explicit(kind: ProviderKind, tmp_path: Path) -> None:
    request = _request("timeout", ProviderBudget(max_wall_seconds=0.01))
    result = invoke_provider(_provider(kind, tmp_path, "slow"), request)
    if kind is ProviderKind.NONE:
        assert result.outcome is ProviderOutcome.UNSUPPORTED
    else:
        assert result.outcome is ProviderOutcome.TIMEOUT
        assert result.failure is not None
        assert result.failure.code is ProviderFailureCode.TIMEOUT

    token = CancellationToken()
    token.cancel()
    result = invoke_provider(
        _provider(kind, tmp_path, "success"), _request("cancel"), cancellation=token
    )
    assert result.outcome is ProviderOutcome.CANCELLED
    assert result.failure is not None
    assert result.failure.code is ProviderFailureCode.CANCELLED


@pytest.mark.parametrize(
    ("kind", "budget"),
    [(kind, ProviderBudget(max_input_tokens=100_000)) for kind in ProviderKind]
    + [(kind, ProviderBudget(max_memory_bytes=1 << 40)) for kind in ProviderKind],
)
def test_token_and_resource_budgets_fail_closed(
    kind: ProviderKind, budget: ProviderBudget, tmp_path: Path
) -> None:
    result = invoke_provider(_provider(kind, tmp_path, "success"), _request("budget", budget))
    assert result.outcome is ProviderOutcome.UNSUPPORTED
    assert result.output is None
    assert result.failure is not None
    if kind is not ProviderKind.NONE:
        assert result.failure.code is ProviderFailureCode.LIMIT_EXCEEDED


def test_provider_result_provenance_is_revalidated_at_shared_boundary() -> None:
    class BadProvider:
        identity = ProviderModelIdentity("bad-provider", "bad-model", "revision-1")
        capability_card = ProviderCapabilityCard(
            identity,
            ProviderKind.LOCAL,
            "bad-provider-v1",
            (ProviderCapability.EXPLAIN_EVIDENCE, ProviderCapability.STRUCTURED_OUTPUT),
            ProviderLimits(4096, 1024, 8192, max_wall_seconds=1.0),
        )

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def call(
            self, request: ProviderRequest, *, cancellation: CancellationToken
        ) -> ProviderResult:
            output = ExplanationProviderOutput("untrusted")
            return ProviderResult(
                request.request_id,
                request.operation,
                self.identity,
                ProviderOutcome.SUPPORTED,
                ProviderProvenance(
                    self.identity.provider_id,
                    self.capability_card.provider_revision,
                    "sha256:" + "0" * 64,
                    "sha256:" + "1" * 64,
                ),
                output=output,
            )

        def stream(
            self, request: ProviderRequest, *, cancellation: CancellationToken
        ) -> Iterator[ProviderStreamEvent]:
            if False:
                yield ProviderStreamEvent(request.request_id, 0, "never")

        def cancel(self, request_id: str) -> bool:
            return False

    result = invoke_provider(BadProvider(), _request("bad-result"))
    assert result.outcome is ProviderOutcome.MALFORMED_OUTPUT
    assert result.failure is not None
    assert result.failure.code is ProviderFailureCode.PROTOCOL
