# Fail-closed competitor adapter contract

`modelsurgeon.adapters.competitor` is the v1.1 boundary for external
compression methods. It accepts only argument-array subprocess commands and
keeps competitor implementation details outside the ModelSurgeon process.

## Capability and provenance

`SubprocessCompetitorAdapter.capabilities` is empty until an operation has a
successful run with both an executable version and artifact-bound evidence.
`capability_decision()` therefore cannot turn a declared command into a support
claim. The success record binds the emitted JSON to the exact transactional
staging path, verifies the digest and byte count, and records the published
destination, source revision, method revision/license, command, seed,
configuration digest, selected environment, logs, and measured resource facts.

The source checkpoint is hashed before and after execution. Any change is a
`source_mutated` terminal result and the staged artifact is discarded. The
existing atomic checkpoint publisher rejects a destination that is the source,
already exists, or is inside a source directory.

## Distinct fail-closed states

The result schema keeps `unsupported`, `timeout`, `crash`, `malformed_output`,
`budget_exhausted`, `interrupted`, `source_mutated`, and
`publication_failed` separate from `success`. A failed command never becomes a
support claim, and a malformed or unbound artifact never becomes a published
benchmark result. Logs and telemetry remain attached to every attempted state.

The contract enforces wall-time, stdout/stderr, artifact, and disk ceilings at
the subprocess boundary. RAM/VRAM-specific claims remain outside this adapter
until a host-specific telemetry provider is supplied; the adapter does not
silently report those resources as measured.

## Conformance evidence

`tests/test_competitor_adapter.py` runs a pinned open-source Python 3.12.13
copy baseline against a temporary source checkpoint and exercises each required
failure state. The test source is read-only from the adapter's perspective and
the successful output is published outside the source tree.
