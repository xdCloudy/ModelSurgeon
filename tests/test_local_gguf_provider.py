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
    LlamaCliRuntime,
    LocalGGUFProvider,
    LocalGGUFProviderConfig,
    ProviderBudget,
    ProviderFailureCode,
    ProviderOutcome,
    SourceSpan,
    invoke_provider,
)
from modelsurgeon.search import compile_intent_record
from modelsurgeon.search.spec_preview import build_spec_preview


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
            if self.mode == "compact":
                content = json.dumps(
                    {
                        "banner": "ignored by the trusted normalizer",
                        "operation": "interpret_intent",
                        "decision": {
                            "quality_retention_ratio": None,
                            "max_vram_bytes": None,
                            "max_ram_bytes": None,
                            "optimize_latency": True,
                            "clarification_required": False,
                        },
                    }
                )
            else:
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


def test_local_provider_compact_intent_is_source_grounded(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)

    result = invoke_provider(
        _provider(path, mode="compact"),
        InterpretIntentRequest(
            "request-compact",
            "Make this model faster, with no more than 1% quality loss.",
        ),
    )

    assert result.outcome is ProviderOutcome.SUPPORTED
    assert result.output is not None
    intent = result.output.intent.to_record()
    assert intent["outcome"] == "executable"
    fields = {field["field_id"]: field for field in intent["fields"]}
    assert fields["constraint.quality"]["value"]["threshold"] == 0.99
    assert fields["objective.latency"]["value"]["direction"] == "minimize"


def test_local_provider_accepts_goal_style_allow_no_more_than_quality_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)

    result = invoke_provider(
        _provider(path, mode="compact"),
        InterpretIntentRequest(
            "request-goal-style",
            "Make this model fit in 6 GB of VRAM, prioritize generation speed, "
            "preserve coding ability, and allow no more than 2% quality loss.",
        ),
    )

    assert result.outcome is ProviderOutcome.SUPPORTED
    assert result.output is not None
    intent = result.output.intent.to_record()
    assert intent["outcome"] == "clarification_required"
    fields = {field["field_id"]: field for field in intent["fields"]}
    assert fields["constraint.quality"]["value"]["threshold"] == 0.98
    assert fields["constraint.peak_vram"]["value"]["threshold"] == 6 * 1024**3
    assert "ambiguity-quality-threshold" not in {
        item["ambiguity_id"] for item in intent["ambiguities"]
    }
    assert "ambiguity-task-quality" in {
        item["ambiguity_id"] for item in intent["ambiguities"]
    }


def test_local_provider_carries_explicit_coding_benchmark_into_spec(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    dataset = tmp_path / "coding.jsonl"
    dataset.write_text(
        '{"id":"one","prompt":"write a function","reference":"return 1"}\n',
        encoding="utf-8",
    )

    result = invoke_provider(
        _provider(path, mode="compact"),
        InterpretIntentRequest(
            "request-task-quality",
            f'Make this model faster, preserve coding ability, use coding benchmark: "{dataset}", '
            "and allow no more than 2% quality loss.",
        ),
    )

    assert result.outcome is ProviderOutcome.SUPPORTED
    assert result.output is not None
    intent = result.output.intent.to_record()
    assert intent["outcome"] == "executable"
    assert intent["ambiguities"] == []
    spec = intent["emitted_spec"]
    assert isinstance(spec, dict)
    task_quality = spec["task_quality"]
    assert isinstance(task_quality, dict)
    assert task_quality["method"] == "code_exact_match"
    assert task_quality["dataset"] == str(dataset.resolve())
    compilation = compile_intent_record(result.output.intent)
    assert compilation.executable
    preview = build_spec_preview(result.output.intent, compilation=compilation)
    assert preview.spec is not None
    assert preview.spec["task_quality"] == task_quality


def test_llama_cli_runtime_uses_bounded_argument_list(tmp_path: Path, monkeypatch: Any) -> None:
    executable = tmp_path / "llama-cli.exe"
    executable.write_bytes(b"runtime")
    model = tmp_path / "model.gguf"
    _gguf(model)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> Any:
        calls.append(command)
        return type("Completed", (), {"returncode": 0, "stdout": "banner {\"ok\":true}"})()

    monkeypatch.setattr("modelsurgeon.conversation.local_gguf.subprocess.run", fake_run)
    runtime = LlamaCliRuntime(
        executable=executable,
        model_path=str(model),
        n_ctx=640,
        n_batch=1,
        n_gpu_layers=0,
        verbose=False,
        max_wall_seconds=2,
    )

    result = runtime.create_chat_completion(
        messages=[{"role": "user", "content": "return JSON"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert result["choices"][0]["message"]["content"] == 'banner {"ok":true}'
    assert calls and calls[0][0] == str(executable.resolve())
    assert "--single-turn" in calls[0]
    assert "user: return JSON" in calls[0]


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
