# Atomic safetensors checkpoints

Issue #123 publishes physically modified tensor state as a new immutable checkpoint directory.
The writer accepts canonical contiguous CPU tensors, a JSON configuration, and explicit maximum
tensor and shard byte sizes. Tensor names are sorted, then packed deterministically without
splitting any tensor. One shard uses `model.safetensors`; multiple shards use Hugging Face's
numbered naming and a sorted `model.safetensors.index.json` weight map.

All shards, the optional index, and `config.json` are written into a unique sibling staging
directory. Files are flushed and synchronized, then each staged tensor is inspected without
materialization and its payload is SHA-256 hashed in 1 MiB reads. Names, shapes, dtypes, byte
sizes, shard assignment, and checksums must match the pre-write plan before the directory is
renamed into visibility. An existing destination is never overwritten.

Source shard checksums are captured before staging and checked again immediately before publish.
Unsupported devices, non-contiguous tensors, invalid names/dtypes, oversized tensors, malformed
source checkpoints, serialization failures, and verification disagreement all leave the
destination absent and remove the owned staging directory.
