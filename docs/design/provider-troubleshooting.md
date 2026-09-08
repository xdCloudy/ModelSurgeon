# Provider diagnostics and troubleshooting

Provider diagnostics are a read-only control-plane report. They do not load a
target model, change an optimization policy, rewrite canonical evidence, or
contact a remote endpoint. The same record is available from Python and the
CLI:

```python
from modelsurgeon.providers import provider_diagnostics

diagnostic = provider_diagnostics(settings, offline=True)
print(diagnostic.canonical_json())
```

```text
modelsurgeon provider diagnostics --json --offline
```

## Read the status and state together

`status` is the coarse outcome used by existing v2.2 callers. `state` and the
per-capability cells explain what is actually known:

| State | Meaning | Safe interpretation |
| --- | --- | --- |
| `configured` | A selection or credential reference was supplied | Configuration exists; it is not runtime evidence. |
| `detected` | A local file/runtime or provider capability card was observed | The bounded adapter boundary was found. |
| `measured` | A bounded probe exercised the capability | Only the probed capability and request budget are covered. |
| `unsupported` | The boundary explicitly does not advertise the capability | Do not guess or fall back silently. |
| `unknown` | The required evidence was not available | Keep the cell unresolved; do not treat it as supported. |

Common codes include:

- `no_llm`: explicit no-provider mode; direct CLI/Python optimization remains available.
- `local_model_missing`: the configured local GGUF file is absent.
- `adapter_unavailable`: an optional local runtime or required credential reference is unavailable.
- `missing_api_key`: the named environment variable has no value.
- `offline_provider_disabled`: offline mode refuses remote selection without contacting it.
- `remote_capabilities_unknown`: a remote provider is configured but has not supplied a bounded capability card.
- `missing_capability`: a measured/provider card omitted a requested capability.

Errors and diagnostic records redact credential-shaped values. API keys are
never serialized; only the environment-variable name may be used for
configuration. Local model paths are represented as presence, not as a path,
in exported diagnostic records.

## Compatibility limits

The local adapter currently recognizes GGUF text models and its capability
card is bounded to interpretation, clarification, explanation, and structured
output. Streaming is an explicit unsupported cell for that adapter. Remote
adapters require a capability probe before their operation cells become
detected; a configured endpoint is not evidence of remote support. No-LLM is
the compatibility-safe choice for direct automation and clean environments.
