# Cumulative physical artifact outcome schema

The physical outcome contract is framework-neutral. Each immutable outcome
records source and parent lineage, architecture counts, tensor shape/byte
deltas, a complete identity-remap record, artifact publication evidence, and
optional deployment measurements. Safetensors and GGUF-specific facts live in
the details object while the core keeps the same identity and reconciliation
rules.

An outcome is complete only when its format schema, SHA-256 digest, byte size,
reload check, and generation smoke are present. Incomplete or unsupported
publication cannot carry a digest or size. Unknown schema versions and
partially populated imports fail closed.

PhysicalOutcomeSequence composes stage deltas once. Each child must name the
preceding outcome, and its cumulative parameter/storage deltas must equal the
sum of stage deltas through that child. This prevents cumulative mixed surgery
from double-counting earlier changes. The checked-in fixtures round-trip one
HF/Safetensors result and one native GGUF result through the typed importer.
