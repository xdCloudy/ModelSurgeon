# Cumulative native GGUF surgery

`run_gguf_cumulative_sequence` is the lineage coordinator for native GGUF
executors. Each executor receives the last accepted child and a unique output
path, then returns a typed measurement containing the rediscovered architecture,
encoded payload hashes, changed tensor names, reload/generation evidence, and
peak working-memory and scratch-disk usage.

Before a child is accepted, the coordinator verifies that every payload not
declared changed is byte-identical to the previous child, both resource limits
are satisfied, and the output is a non-empty regular file. The resulting child
is represented as a complete `PhysicalArtifactOutcome`, and
`PhysicalOutcomeSequence` verifies cumulative parameter and storage deltas.

Native MLP, attention-head, layer, and low-rank executors remain responsible
for live codec planning, metadata remapping, bounded streaming, and pinned
llama.cpp reload/generation checks. This coordinator deliberately does not
infer support for a family or quantization profile that its executor did not
declare.

If a stage fails, its exact child path is removed and the prior accepted child
remains the source for any caller-managed resume. No failed or partial child is
included in the accepted lineage.
