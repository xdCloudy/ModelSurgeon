# Experimental chat session bootstrap

The session bootstrap includes the versioned [chat inspection context](chat-inspection-context.md).
It carries direct model, hardware, memory, runtime, and capability records to
the compiler while preserving explicit unknown and unsupported outcomes.

Status: implemented as the bounded v2.3 entry slice. Interpretation remains
separate from execution, and the optional [chat optimize execution adapter](chat-execution-adapter.md)
hands only confirmed specs to the stable planner/orchestrator APIs.

## Command contract

```text
modelsurgeon chat <provider-model-path> [options]
```

The initial supported provider is the local GGUF text-model adapter:

```bash
uv run modelsurgeon chat ./provider.gguf --request "retain quality and reduce latency"
```

The command validates the path, `.gguf` container, explicit supported
architecture metadata, content SHA-256, and the pinned `llama-cpp-python`
revision before starting the provider. It does not download a model or use a
network fallback. `--provider none` is an explicit no-provider mode useful for
offline contract checks; interpretation then returns `unsupported`.

To request a stable plan for a separate target model, add `--target-model`,
`--target-revision`, and `--preview-plan`. Add `--execute`, `--state`, an
explicit `--approval-id`, and every required `--approve` value only after the
plan has been reviewed. `--resume` resumes the matching durable campaign.

Each session has a deterministic `chat_session_<sha256>` identity and a hard
turn budget (8 by default, maximum 64). Each request has bounded input,
output, and wall-time budgets. `--json` emits one canonical bootstrap record
followed by one retained turn record per request.

## Authority and outcomes

The provider is invoked only through `invoke_provider()`. A supported provider
response must decode to the canonical `IntentRecord`, then passes through the
existing intent compiler and policy evaluator. The validated interpretation is
shown in the turn record before any execution hand-off could be considered.
When the optional execution flags are absent, records state
`execution: not_requested`. With them, the execution record contains only
typed planner/orchestrator outcomes and engine-owned provenance.

Provider and policy outcomes remain explicit: `supported`, `unsupported`,
`failed`, `unknown`, `timeout`, `cancelled`, `malformed_output`,
`clarification_required`, and `refused` are not collapsed into success.
Invalid paths, unsupported formats, unknown architectures, unavailable
runtimes, malformed provider output, and cancellation fail closed.

Provider identity, capability card, model revision, runtime revision,
architecture, request
identity, request/response provenance, interpretation, policy diagnostics, and
negative outcomes remain in the canonical records. The provider model is the
control-plane client; it is not the target model and cannot select tensors,
approve work, mutate artifacts, or claim live optimization evidence.
