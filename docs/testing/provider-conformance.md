# Provider capability and failure conformance

Status: implemented for the v2.2 replaceable text-model provider boundary.

This suite checks the typed provider contract, not natural-language quality or
optimization correctness. Local GGUF, compatible endpoint, and hosted cells
use deterministic fixtures. The no-LLM provider is tested as a first-class
mode with explicit unsupported cells; unsupported and failed outcomes remain
retained records rather than being counted as successes.

## Capability matrix

`V` means the shared harness has a deterministic test for the behavior. `—`
means the provider mode explicitly does not claim that capability.

<!-- BEGIN GENERATED MATRIX -->
| Capability | local | compatible_endpoint | hosted | none |
| --- | --- | --- | --- | --- |
| structured_output | V | V | V | — |
| refusal | V | V | V | V |
| timeout | V | V | V | — |
| cancellation | V | V | V | V |
| token_budget | V | V | V | — |
| resource_budget | V | V | V | — |
| provenance | V | V | V | V |
<!-- END GENERATED MATRIX -->

The no-LLM refusal, cancellation, and provenance cells cover the boundary
behavior: `NullTextModelProvider` returns a typed unsupported result, a
pre-cancelled request returns `CANCELLED`, and both retain deterministic
request/provider provenance. It does not execute a text-model request, so
structured output, timeout execution, and provider budget enforcement are
explicitly unsupported.

## Contract assertions

The shared tests verify that every adapter:

- validates structured output before a supported result can be emitted;
- retains malformed output, refusal, timeout, cancellation, and budget
  failures with request identity and safe diagnostics;
- enforces advertised token and memory limits before provider work proceeds;
- preserves deterministic request and response provenance; and
- cannot return an executable-looking output that bypasses the common decoder,
  even when an implementation manually constructs a result.

Run the offline matrix with:

```text
uv run --locked --extra dev pytest tests/test_provider_conformance.py
```

The fixtures use a tiny metadata-only GGUF container, an in-memory endpoint
transport, and a deterministic runtime double. They are protocol evidence
only; no hosted service, credential, model-quality score, or optimization
measurement is implied. The exact local runtime, repository revision, budget
values, and retained outcomes are emitted by the test log when the suite is
run.
