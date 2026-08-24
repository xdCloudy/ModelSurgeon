# Bounded logit-distillation repair v1

Distillation repairs a mutated candidate toward either precomputed immutable
baseline logits or logits captured by inference from a distinct teacher. Captured
logits are detached, converted to contiguous FP32 CPU tensors, cloned, shape- and
finite-validated, byte-bounded, and content-hashed before candidate mutation.

Teacher and candidate tokenizers are represented by exact signatures containing
the effective vocabulary size and digest plus core special-token IDs. Any mismatch
fails before teacher capture or candidate mutation. Candidate parameters are an
exact canonical name set with a preflight trainable-parameter ceiling.

The objective is temperature-scaled teacher-to-candidate KL mixed with optional
supervised loss; the two configured weights must sum to one. Each step evaluates
the complete repair set, and only the best state that improves upon the no-repair
candidate is retained. Wall-budget exhaustion, no improvement, or exceptions
restore exact candidate snapshots and produce no child checkpoint.

The result records teacher source and logit hash, tokenizer signature, temperature,
loss mix, histories, selected parameter count, token rows, teacher-logit bytes,
teacher-capture/training/total time, memory peaks, and immutable source/parent/output
lineage.
