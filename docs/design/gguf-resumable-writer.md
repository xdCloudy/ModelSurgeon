# Resumable GGUF output construction

Resumable output uses stable private staging and manifest paths beside the destination.
The manifest binds the stage to SHA-256 identities for both immutable input and exact
output layout. It records only fsynced tensor boundaries, the committed prefix checksum,
and each completed tensor checksum.

On resume, the writer verifies schema, input and plan identities, completed tensor names
and offsets, stage length, and the committed prefix digest. Bytes after the last
committed tensor are truncated, so a power loss during a tensor replays only that tensor;
completed sources are not read again. Every subsequent tensor is fsynced before its
manifest checkpoint is atomically replaced.

No resume path publishes a partial model. Final parser validation and atomic no-replace
publication use the same contract as one-shot output. Changed inputs, layouts, or
committed bytes fail closed and preserve evidence for inspection. Explicit discard
removes only the three private resume artifacts associated with the exact destination.
