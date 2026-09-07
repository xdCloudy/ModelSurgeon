# Numerical and artifact stability contract

`modelsurgeon.evaluation.platform_stability` records whether a reference run
and a candidate run remain semantically comparable across Windows, WSL, Linux,
CPU/CUDA, dtypes, runtimes, thread counts, and pinned upstream revisions.

Each comparison retains exact platform/runtime/dtype/thread/accelerator,
upstream revision, configuration digest, and optional artifact digest/size.
The result is one of supported, unsupported, failed, or unknown. Unsupported
hardware and unavailable probes remain cells in the matrix; they are not
silently converted into passing metrics.

The immutable tolerance registry is version 1. Every metric definition carries
its unit, expected floating variation, semantic-drift bound, definition
revision, and changelog marker. A change to a metric definition therefore
requires an explicit registry revision and a changelog marker. Comparisons
classify values as within tolerance, expected variation, semantic drift,
artifact drift, or unknown. Semantic and artifact drift fail closed.

Persisted records use schema version 1. `migrate_platform_stability_record`
supports the additive v0-to-v1 migration and rejects unknown versions without
rewriting the original evidence. Record IDs are deterministic hashes of the
complete comparison evidence.

The checked-in tests are a tiny deterministic CI matrix. Scheduled jobs may
add licensed HF/GGUF/reference-hardware cells using the same record contract.

