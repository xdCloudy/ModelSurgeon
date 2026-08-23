# Selective GGUF requantization

Changed float blocks are encoded with the exact destination codec declared by
the validated GGUF binding. The destination defaults to the source type, but an
explicit different native codec is supported when its recomputed output size is
already reflected in the physical plan.

Encoding streams complete destination blocks under encoded-size, validation-value,
and combined working-memory ceilings. Every payload is structurally revalidated,
decoded once into a bounded validation buffer, and compared with its changed
float input. Chunk-level mean, squared, maximum, and L2 error metrics are emitted;
optional maximum-error and mean-squared-error ceilings reject a payload before it
is yielded or recorded.

Destination block ranges must be ordered, non-overlapping, within the new tensor
shape, and exact multiples of the selected codec. Audit reports record output
block/byte ranges, payload totals, error summaries, and observed peak working
bytes. Unchanged regions remain outside this encoder and retain their direct-copy
path.
