# Cumulative Hugging Face physical surgery

`run_huggingface_cumulative_sequence` composes physical MLP-width, attention-
head, transformer-layer, and low-rank edit callbacks against the current
reloaded model. Each accepted stage has its own content-addressed child path
and must pass all of these gates:

1. Apply the edit to a deep-copied last accepted model.
2. Publish through the caller's atomic safetensors publisher.
3. Reload the published child.
4. Run a bounded generation smoke check.
5. Snapshot parameter count, tensor count, shapes, dtypes, and storage bytes.
6. Append a typed `PhysicalArtifactOutcome` to the cumulative lineage.

The `PhysicalOutcomeSequence` check verifies that parameter and storage deltas
are cumulative rather than double-counted. The sequence ID hashes the ordered
edit identities, so order changes are explicit identities even when two
orders happen to produce equivalent shapes. The first accepted model remains
the committed state if a later stage fails. The failed stage's uniquely named
child is removed and no partial child is reported as an accepted artifact.

The publisher, reloader, and generation smoke are explicit interfaces because
the repository supports multiple Hugging Face model families. Production
callers should connect the publisher to the existing atomic safetensors
checkpoint writer and use revision-pinned model loaders and deterministic
generation inputs.
