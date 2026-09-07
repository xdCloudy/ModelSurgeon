# Deployable architecture state

`modelsurgeon.search.deployable_state` is the v1.4 state boundary for search
and distance calculations. It contains no live framework model objects.

## State identity

`DeployableArchitectureState` records model family/revision, depth, contiguous
per-layer attention and MLP widths, query/KV heads, hidden and embedding sizes,
low-rank factors, component sparsity, quantization, placement, artifact
lineage, and ordered mutation IDs. A state ID is the SHA-256 of the canonical
versioned record, so quantization, placement, and mutation order cannot be
silently deduplicated.

Every axis is known, predicted, unknown, or unsupported. Missing values must
carry an explicit non-known status. Artifact records distinguish complete
physical output (with digest, size, and physical outcome evidence) from
predicted, not-yet-materialized, unsupported, and unknown output.

## Distance

`architecture_distance()` returns a versioned decomposition across family,
architecture axes, quantization, placement, and mutation order. Numeric axes
use bounded normalized absolute differences; categorical context changes are
visible as a unit distance. The result is symmetric and finite when both
states are fully comparable. Unknown or unsupported axes yield an explicit
unknown distance rather than an extrapolated score.

## Compatibility

`deployable_state_from_record()` recomputes and verifies the state ID. Unknown
schema versions fail closed. The migration hook accepts only the minimal v0
additive shape and still subjects the result to current validation. The public
surface is exported from `modelsurgeon.search` and is intended for persisted
search candidates, lineage records, and visualization-safe summaries.
