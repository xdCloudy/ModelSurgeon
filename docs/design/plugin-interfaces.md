# Capability-negotiated plugin interfaces v1

`modelsurgeon.plugins` defines the extension boundary for evaluator, runtime,
transformation, search-policy, objective, and registry-provider plugins.

## Discovery and compatibility

Entry-point discovery uses `importlib.metadata` metadata only. Discovery returns
descriptors and does not import plugin modules or execute arbitrary code. Core
loading is an explicit allowlist operation. A `PluginCapabilityCard` declares
kind, plugin/API versions, capabilities, license, dependencies, trust modes,
and resource bounds.

`negotiate_plugin()` fails closed on kind mismatch, API-major skew, missing
capabilities, and unavailable dependencies. Unknown capabilities are not
inferred from names or silently enabled.

## Invocation boundary

`PluginRequest` contains primitive JSON-compatible payloads and source/config
identities. It never carries a live model, mutable source object, or local
filesystem handle. In-process invocation requires both an explicit caller
trust decision and the plugin's `trusted_in_process` declaration. Otherwise
the subprocess boundary is the default.

Subprocess invocation sends canonical JSON over stdin and enforces wall-time,
stdout, stderr, and artifact budgets. Timeout, process crash, malformed output,
unsupported capability, and ordinary failure remain distinct outcomes. A
plugin cannot turn malformed output into a supported result.

```python
report = negotiate_plugin(card, request)
result = invoke_subprocess(card, command, request)
```

No marketplace or arbitrary sandbox is implied by this contract; deployments
must choose their allowlist, dependency inventory, and trust policy explicitly.
