# Kernel shape and offload microbenchmarks

`modelsurgeon.evaluation.microbenchmarks` stores isolated transformer-relevant kernel
evidence bound to a v1.3 hardware profile. A partition is content-addressed by profile and
normalized context IDs, exact shape/dtype/codec/batch/context/offload settings, runtime and
tool revisions, and seed.

## Preregistered shapes

The protocol crosses tensor-core and SIMD dimensions, cache-line and attention-head
boundaries, KV-cache context boundaries, GGUF block boundaries, and zero/partial offload
layer splits. It does not extrapolate from one shape to another and does not claim an
end-to-end model speedup from isolated measurements.

## Measurement and validity

Each shape has two warmups and at least ten timed observations. Every observation carries a
hardware context ID and an optional environment fingerprint. Context changes or conflicting
available fingerprints reject the result. Metrics retain median, p95, dispersion, and a
conservative empirical confidence envelope from the observed repetitions for prefill and
decode latency, throughput, bandwidth, RAM, and VRAM.

Cells whose relative dispersion exceeds the preregistered threshold are retained as
`unstable` and excluded from eligible predictor results. Unsupported, failed, and unknown
cells are retained with reasons. The partition validity report exposes only measured,
stable result IDs as eligible evidence.
