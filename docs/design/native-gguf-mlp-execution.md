# Native GGUF MLP Execution

Issue #150 turns a validated native MLP channel-removal plan into a new GGUF
without materializing a full floating-point model. The executor is deliberately
limited to the coupled Llama and dense Qwen gate, up, and down projections
accepted by the planner.

## Execution contract

The executor requires an open read-only `MemoryMappedGGUF`, a
`NativeGGUFMLPRemovalPlan`, an exact-codec registry, a distinct destination, and
a successful conservative disk preflight. Before reading changed payloads it
checks that source dimensions, source codec identities, destination dimensions,
and encoded sizes still agree with the plan.

The source container's GGUF version, byte order, alignment, metadata value
types, tensor order, tensor names, and original GGML type IDs are retained. Only
the architecture feed-forward-width metadata value and the three planned MLP
tensor shapes/payloads change.

## Low-memory paths

Each planned axis selects one of three paths:

- whole-slice copy drops complete gate/up rows and copies retained encoded blocks;
- direct-block copy removes complete aligned blocks from each down-projection row;
- contiguous-axis repack decodes one down-projection row, removes the requested
  values, and immediately requantizes that row with the exact original codec.

All input reads and output chunks obey explicit encoded-byte ceilings. The
repack path retains at most one old row, one filtered row, and the bounded codec
validation workspace. It checks their combined working estimate against
`max_row_working_bytes`. It never constructs a complete dequantized tensor or
model.

## Transaction and audit behavior

Output uses tensor-boundary resumable staging. A failure leaves no published
destination; the exact staging pair can either resume with the same input and
plan or be explicitly discarded. Publication occurs only after the staged GGUF
passes structural validation and is synchronized to disk.

The result contains the output discovery graph, output and per-tensor SHA-256
digests, source hashes for every untouched tensor, per-row requantization error
summaries, and observed peak row working bytes. After publication the executor
also verifies that every untouched tensor is byte-identical and that the output
graph's feed-forward width and coupled tensor shapes match the plan.

Aligned removals never enter a float codec path. Arbitrary block-representable
removals requantize only down-projection rows whose contiguous axis must be
repacked; gate/up retained rows remain encoded copies.
