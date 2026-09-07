"""Tests for the bounded offline local GGUF provider adapter."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.gguf import GGUFValueType
from modelsurgeon.conversation import (
    CancellationToken,
    ExplanationProviderOutput,
    ExplanationRequest,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    InterpretIntentRequest,
    LocalGGUFProvider,
    LocalGGUFProviderConfig,
    ProviderBudget,
    ProviderFailureCode,
    ProviderOutcome,
    SourceSpan,
    invoke_provider,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path, *, architecture_type: GGUFValueType = GGUFValueType.STRING) -> None:
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
    data.extend(_string("general.architecture"))
    data.extend(struct.pack("<I", architecture_type))
    if architecture_type is GGUFValueType.STRING:
        data.extend(_string("llama"))
    else:
        data.extend(struct.pack("<I", 1))
    data.extend(b"\0" * ((-len(data)) % 32))
    path.write_bytes(data)


def _intent(request: str) -> IntentRecord:
    span_id = "request-span"
    return IntentRecord(
        request,
        (SourceSpan(span_id, 0, len(request), request),),
        (IntentField("objective.metric", "latency", None, 0.99, (span_id,)),),
        (),
        (
            InterpretationStep(
                "normalize-request",
                "normalize",
                (span_id,),
                ("objective.metric",),
                0.99,
                ("evidence:request",),
            ),
        ),
        IntentProvenance(
            "request-revision-1",
            "local-gguf",
            "local-gguf-provider-v1",
            "local-provider-test",
            ("evidence:request",),
        ),
        IntentOutcome.EXECUTABLE,
        ("request normalized",),
        {"contract_id": "objective_contract_example", "schema_version": 1},
    )


class _Runtime:
    def __init__(self, *, mode: str = "interpret") -> None:
        self.mode = mode
        self.closed = False

    def tokenize(self, prompt: bytes, *, add_bos: bool) -> list[int]:
        return list(range(max(1, (len(prompt) + 3) // 4)))

    def create_chat_completion(
        self, *, messages: list[dict[str, str]], **_: object
    ) -> dict[str, Any]:
        prompt = messages[0]["content"]
        encoded = prompt.split("REQUEST_JSON_BEGIN\n", 1)[1].split("\nREQUEST_JSON_END", 1)[0]
        request = json.loads(encoded)
        if self.mode == "malformed":
            content = "not json"
        elif request["operation"] == "interpret_intent":
            content = json.dumps(
                {
                    "operation": "interpret_intent",
                    "intent": _intent(request["original_request"]).to_record(),
                }
            )
        else:
            content = json.dumps(
                {
                    "operation": request["operation"],
                    "text": "evidence explained",
                    "evidence_refs": [],
                }
            )
        return {"choices": [{"message": {"content": content}}]}

    def close(self) -> None:
        self.closed = True


def _provider(
    path: Path, *, mode: str = "interpret", factory: object | None = None
) -> LocalGGUFProvider:
    runtime_factory = factory or (lambda **_: _Runtime(mode=mode))
    return LocalGGUFProvider(
        LocalGGUFProviderConfig(
            path,
            "fixture-revision-1",
            "fixture-runtime-1",
            max_input_tokens=512,
            max_output_tokens=128,
            max_context_tokens=640,
            max_memory_bytes=8 * 1024 * 1024,
            n_batch=1,
        ),
        runtime_factory=runtime_factory,  # type: ignore[arg-type]
        runtime_version="fixture-runtime-1",
    )


def test_local_provider_uses_common_interpretation_and_explanation_contract(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    provider = _provider(path)

    interpreted = invoke_provider(provider, InterpretIntentRequest("request-1", "reduce latency"))
    assert interpreted.outcome is ProviderOutcome.SUPPORTED
    assert interpreted.output is not None
    assert interpreted.provenance.model_path == str(path.resolve())
    assert interpreted.provenance.runtime_revision == "fixture-runtime-1"
    assert interpreted.provenance.configuration_digest is not None

    explained = invoke_provider(
        provider,
        ExplanationRequest("request-2", ({"record_type": "evidence", "value": 1},)),
    )
    assert explained.outcome is ProviderOutcome.SUPPORTED
    assert isinstance(explained.output, ExplanationProviderOutput)


def test_local_provider_rejects_missing_and_malformed_models(tmp_path: Path) -> None:
    missing = _provider(tmp_path / "missing.gguf")
    result = invoke_provider(missing, InterpretIntentRequest("request-3", "explain"))
    assert result.outcome is ProviderOutcome.FAILED
    assert result.failure is not None
    assert result.failure.code is ProviderFailureCode.UNAVAILABLE

    malformed = tmp_path / "malformed.gguf"
    _gguf(malformed, architecture_type=GGUFValueType.UINT32)
    result = invoke_provider(_provider(malformed), InterpretIntentRequest("request-4", "explain"))
    assert result.outcome is ProviderOutcome.UNSUPPORTED
    assert result.failure is not None
    assert result.failure.code is ProviderFailureCode.INVALID_MODEL


def test_local_provider_retains_malformed_output_and_resource_failures(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    malformed = invoke_provider(
        _provider(path, mode="malformed"), InterpretIntentRequest("request-5", "explain")
    )
    assert malformed.outcome is ProviderOutcome.MALFORMED_OUTPUT

    def oom_factory(**_: object) -> object:
        raise MemoryError("allocation failed")

    oom = invoke_provider(
        _provider(path, factory=oom_factory), InterpretIntentRequest("request-6", "explain")
    )
    assert oom.outcome is ProviderOutcome.FAILED
    assert oom.failure is not None
    assert oom.failure.code is ProviderFailureCode.RESOURCE_EXHAUSTED


def test_local_provider_enforces_context_and_cancellation(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    provider = _provider(path)
    too_small = invoke_provider(
        provider,
        InterpretIntentRequest(
            "request-7", "explain", budget=ProviderBudget(max_input_tokens=1)
        ),
    )
    assert too_small.outcome is ProviderOutcome.UNSUPPORTED
    assert too_small.failure is not None
    assert too_small.failure.code is ProviderFailureCode.LIMIT_EXCEEDED

    token = CancellationToken()
    token.cancel()
    cancelled = invoke_provider(
        provider, InterpretIntentRequest("request-8", "explain"), cancellation=token
    )
    assert cancelled.outcome is ProviderOutcome.CANCELLED


def test_local_provider_timeout_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)

    class SlowRuntime(_Runtime):
        def create_chat_completion(self, **kwargs: object) -> dict[str, Any]:
            time.sleep(0.05)
            return super().create_chat_completion(**kwargs)  # type: ignore[arg-type]

    provider = LocalGGUFProvider(
        LocalGGUFProviderConfig(
            path,
            "fixture-revision-1",
            "fixture-runtime-1",
            max_input_tokens=512,
            max_output_tokens=128,
            max_context_tokens=640,
            max_memory_bytes=8 * 1024 * 1024,
            max_wall_seconds=1,
            n_batch=1,
        ),
        runtime_factory=lambda **_: SlowRuntime(),
        runtime_version="fixture-runtime-1",
    )
    result = invoke_provider(
        provider,
        InterpretIntentRequest("request-9", "explain", budget=ProviderBudget(0.01)),
    )
    assert result.outcome is ProviderOutcome.TIMEOUT
    assert result.failure is not None
    assert result.failure.code is ProviderFailureCode.TIMEOUT
