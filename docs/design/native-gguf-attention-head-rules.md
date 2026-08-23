# Native GGUF Attention-Head Rules

Issue #151 resolves which Llama and dense Qwen MHA, GQA, and MQA head removals
can be represented without silently changing query-to-KV ownership.

The decision is pinned to llama.cpp revision
`c060ca974c773c7c3d17fd1b66dc9d312bc292c0`. Its loader defaults key/value head
length to embedding width divided by query-head count, but accepts explicit
`attention.key_length` and `attention.value_length`. Its Llama tensor loader
constructs Q and O widths from query-head count times head length, and K/V widths
from KV-head count times head length. The rule resolver therefore preserves the
original key/value head lengths explicitly while reducing head counts and tensor
axes. See the upstream
[hyperparameter loader](https://github.com/ggml-org/llama.cpp/blob/c060ca974c773c7c3d17fd1b66dc9d312bc292c0/src/llama-model.cpp)
and [Llama tensor declarations](https://github.com/ggml-org/llama.cpp/blob/c060ca974c773c7c3d17fd1b66dc9d312bc292c0/src/models/llama.cpp).

## Safe grouping rule

Query heads are partitioned into contiguous groups according to the source
`query_heads / kv_heads` ratio. A request is representable when:

- one or more complete KV groups may be removed, including their K/V head and
  every associated query head; and
- every retained KV group removes the identical set of local query-head offsets.

This covers MHA naturally: each query head is its own KV group, so Q/K/V/O all
remove matching heads. GQA may remove complete groups, an equal local pattern
from all retained groups, or both. MQA has one KV group, so it may remove query
heads but can never remove its sole K/V head.

Uneven local GQA patterns are rejected because compacting Q would otherwise map
retained query heads to different KV heads. Empty, duplicate, out-of-range, and
all-head requests also fail before payload access.

## Tensor and codec rules

Head removal is model-wide because the supported GGUF metadata uses scalar head
counts. The resolver emits Q/K/V/O rules for every layer:

- Q axis 1 removes `key_length` values per query head;
- K axis 1 removes `key_length` values per removed KV head;
- V axis 1 removes `value_length` values per removed KV head;
- O axis 0 removes `value_length` values per query head.

Outer-axis edits copy retained encoded head slices. O-axis edits use direct block
copy only when every touched codec block is removed completely; otherwise they
require selective repacking. Every destination shape must remain representable by
its exact tensor codec. The output must explicitly write query-head count,
KV-head count, key length, and value length metadata.
