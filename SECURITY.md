# Security Policy

## Supported versions

Until v1.0, only the latest `main` revision receives security fixes.

## Reporting

Please use GitHub private vulnerability reporting when available. Do not disclose exploitable model-loading, artifact-path, deserialization, or checkpoint-overwrite issues publicly before maintainers have assessed them.

## Security model

Model and dataset inputs are untrusted. Remote model code is disabled by default. ModelSurgeon must not overwrite source checkpoints by default, must use atomic output publication, and must constrain artifact paths to the configured run directory. Generated models can behave unexpectedly and require evaluation before use.

## Conversational control-plane boundary

The v2.9 conversational security gate is a bounded, local, typed control-plane
contract; it is not a general security certification. Consequential actions
require scoped approval for the exact plan and material diff. Prompts, provider
text/metadata, and tool output are untrusted and cannot override hard
constraints, approvals, evidence, transactions, or source-model immutability.
Unknown, contradictory, malformed, stale, and isolation-failure states fail
closed, and negative or unresolved observations remain retained.

The repository does not claim operating-system containment for hostile
in-process providers or handlers, live hosted-provider or credential security,
complete future prompt-injection resistance, distributed campaign recovery,
optimizer optimality, or universal deployment/model-family support. Do not put
credentials in prompts, fixtures, evidence bundles, diagnostics, or chat
history. See the [v2.9 security boundary](docs/release/v2.9-conversational-security-boundary.md)
and [residual-risk record](docs/research/v2.9-conversational-security-release-v1.json).
The v3.0 product release preserves these limits and audits them through the
[v3.0 release manifest](docs/research/v3.0-product-release-v1.json); it is not
a general security certification.

The core package does not depend on `diskcache` and does not import it. The
optional local GGUF adapter requires a separately managed `llama-cpp-python`
runtime; current releases of that runtime pull unmaintained `diskcache`, which
uses pickle serialization by default and has no patched release. Do not expose
that runtime's cache directory to untrusted writers, and do not treat the
optional local provider as production-safe until its dependency graph is
replaced or patched.

