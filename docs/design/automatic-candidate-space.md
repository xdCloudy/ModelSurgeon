# Automatic capability and candidate-space planning

`build_auto_candidate_space` composes pinned model discovery, target hardware,
the architecture compatibility matrix, objective revisions, plugin capability
names, and the existing bounded `ArchitectureCandidateSpace` generator. It is
a dry-run boundary: it does not download a source, execute a runtime, mutate a
model, or publish an artifact.

## Identity and bounds

The request records model ID/revision/base state, hardware profile/revision,
objective and tool revisions, compatibility matrix ID, discovered axis
domains, plugin names, and an unsigned 64-bit seed. The resulting plan ID
hashes that canonical record and the exclusions. Candidate count is capped at
100,000; evaluation count is capped by the request; artifact bytes are bounded
by the target disk envelope and a conservative two-copy source estimate.

## Fail-closed capability decisions

Unknown architecture or runtime revision produces `unknown` with no candidate
space. Unsupported compatibility cells or missing required plugin names
produce `unsupported` with explicit keys and reasons. A coupling rule that
references an undiscovered axis, or a source larger than the target disk
envelope, produces `failed`. No refusal is converted into an empty successful
space, and no unsupported capability is inferred from a neighboring family,
codec, or runtime.

Supported requests retain a canonical candidate space with hardware storage
rules and coupling constraints. Its existing generator reports rejected
cells, supports bounded pages, and preserves deterministic candidate IDs.
