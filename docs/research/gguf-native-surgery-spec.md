# Native GGUF surgery specification map

Status: accepted bounded spike for #17  
Decision date: 2026-08-23

## Outcome

Native quantized GGUF surgery is feasible without loading a complete model or converting it to a Hugging Face checkpoint. ModelSurgeon v1.0 will treat the container as an indexed, aligned byte stream; decode, edit, and re-encode only bounded codec blocks; update architecture metadata and coupled tensors transactionally; and preserve every unmodified byte range where the format permits.

GGUF is the container. `ggml_type` is the per-tensor storage codec. Labels such as `Q4_K_M` are whole-file quantization recipes that select a mixture of tensor codecs; they are not interchangeable tensor types. The implementation must never infer a codec from a filename or substitute one K/IQ layout for another.

## Pinned upstream sources

All conclusions and vector expectations are pinned rather than referring to a moving branch:

| Authority | Revision | Use |
|---|---|---|
| [ggml GGUF specification](https://github.com/ggml-org/ggml/blob/d99724f2b141fed9e1a9f402213506c081433465/docs/gguf.md) | `d99724f2b141fed9e1a9f402213506c081433465` | container layout, metadata and version history |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/95b8e33e16bb9a60de780a70930ebf729db6a90a) | `95b8e33e16bb9a60de780a70930ebf729db6a90a` | reference runtime revision |
| [`gguf.h`](https://github.com/ggml-org/llama.cpp/blob/95b8e33e16bb9a60de780a70930ebf729db6a90a/ggml/include/gguf.h) | blob `b3a1e1230a06110a0a5f04469d849ac2000b467c` | C container API, v3/default alignment constants |
| [`ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/95b8e33e16bb9a60de780a70930ebf729db6a90a/ggml/src/ggml-common.h) | blob `83f9118da84a6a61967a5c8a04af9893130a4e95` | normative block structures and static sizes |
| [`ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/95b8e33e16bb9a60de780a70930ebf729db6a90a/ggml/src/ggml-quants.c) | blob `1ebc50a763f16db909de37090da38cc8c0fdde94` | normative scalar reference encode/decode behavior |
| [`gguf-py` quant codecs](https://github.com/ggml-org/llama.cpp/blob/95b8e33e16bb9a60de780a70930ebf729db6a90a/gguf-py/gguf/quants.py) | blob `80966b6ef1518a45b86745d94eb70d05c3c5490f` | Python cross-check and bit-exact Q8_0 claim |
| [antirez/gguf-tools](https://github.com/antirez/gguf-tools/tree/fdfafbed766db0a1e9019b07994cd88f133d1aab) | `fdfafbed766db0a1e9019b07994cd88f133d1aab` | independent C reader design comparison |

The executable fixture uses `gguf==0.19.0` and `tests/fixtures/gguf_spec_v1.json`. The test creates a v3 Q8_0 file and requires official llama.cpp `gguf-py` and a deliberately independent Python-stdlib structural reader to agree on all metadata, tensor shape/type, alignment, offsets, and byte size.

## Container revisions and byte layout

| Version | Decision |
|---|---|
| v1 | Initial format. Reject for mutation in v1.0; it is outside current upstream reader support and uses legacy count widths. |
| v2 | Counts/lengths moved from 32 to 64 bits. Read and mutate after the mmap parser and writer pass pinned vectors. It is little-endian. |
| v3 | Adds byte-swapped/big-endian files. This is the primary write target. Readers must detect the swapped version word; writers preserve source endianness unless an explicit migration is recorded. |

The ordered structure is magic `GGUF`, version, tensor count, metadata count, typed key/value records, tensor descriptors, padding to the global alignment, and tensor bytes. v2/v3 strings and array lengths use `uint64`; tensor dimension count and type tags use `uint32`; dimensions and relative data offsets use `uint64`. Tensor offsets are relative to the aligned tensor-data base, not the file start.

Metadata and descriptor fields are packed sequentially without incidental alignment. The tensor-data base and every tensor's relative offset are aligned. Missing `general.alignment` means 32. For safe mutation, #124 must accept only bounded, non-zero power-of-two alignments that are multiples of eight; validate every addition/multiplication before mapping; reject overlaps, out-of-file ranges, excessive dimensions, duplicate keys/names, and tensor byte counts inconsistent with shape and codec.

Unknown typed metadata is preserved losslessly, including order and exact value type. `general.architecture` selects #19's explicit architecture adapter. Quantized files require `general.quantization_version`. A structural edit updates all affected architecture counts, split metadata, tensor names/shapes, and tokenizer/output metadata declared by that adapter; no global key is guessed.

## Codec families

Block size is the number of logical values; type size is the encoded bytes per block. A quantized tensor's fastest-varying storage dimension must be divisible by its codec block size. Partial blocks are rejected rather than padded silently.

| Family | Types | Block/type bytes | Native-surgery decision and owner |
|---|---|---|---|
| F32 | F32 | `1/4` | Raw IEEE binary32 values. Implement endian-aware direct copy/edit in #134; unchanged bytes remain untouched. |
| F16 | F16 | `1/2` | IEEE binary16 values. Implement endian-aware conversion in #134 and test rounding, signed zero, infinities, and NaNs against upstream. |
| BF16 | BF16 | `1/2` | Bfloat16 values with different mantissa width from F16. Implement as a distinct #134 codec; never route through the F16 layout merely because both occupy two bytes. |
| Q8 | Q8_0 | `32/34` | One fp16 delta plus 32 signed quants; implement from the scalar upstream reference in #135. Do not confuse with Q8_1 or internal Q8_K. |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | `256/84`, `256/110`, `256/144`, `256/176`, `256/210`, `256/292` | Each layout has distinct scale/min/high-bit packing. Contract in #18; codecs in #136–#139. Q8_K is primarily an intermediate dot-product layout and is not a `Q8_0` substitute. |
| IQ | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M | `256/66`, `256/74`, `256/98`, `256/50`, `32/18`, `256/110`, `256/82`, `256/136`, `256/56` | Codebook/sign-table and sometimes importance-data dependent. #140 measures IQ2 and IQ4 samples and selects the finite v1 matrix; #141 implements only selected types. Every other IQ type fails before mutation. IQ support remains a core v1 deliverable, but support claims are evidence-gated rather than guessed. |

`Q4_K_S`, `Q4_K_M`, `Q5_K_S`, and `Q5_K_M` describe file-level tensor-selection recipes. The tensor descriptor still stores Q4_K, Q5_K, Q6_K, or another concrete `ggml_type`. ModelSurgeon records the concrete codec for every edited tensor and the original whole-file recipe as provenance.

The normative encoder is the scalar/reference implementation at the pinned llama.cpp revision. Optimized CPU/GPU kernels may be used only after #133 proves decoded values, encoded bytes where deterministic, error bounds, and malformed-block behavior against pinned vectors. A round trip is not assumed byte-identical for lossy re-encoding; unchanged blocks are copied byte-for-byte, while changed blocks record error metrics.

## Low-memory surgery execution contract

1. #124 maps the container read-only and builds a bounded descriptor index without touching payload pages.
2. #19 resolves architecture, axis semantics, coupled groups, renames, and metadata updates explicitly.
3. #18 selects the exact codec and validates axis/block divisibility before allocation.
4. The surgery engine reads at most a configured chunk of complete blocks, decodes into a bounded scratch buffer, applies the mutation, re-encodes, and immediately writes to a staged output.
5. #130 writes a new header/descriptors, streams unchanged ranges from the source, inserts zero alignment padding, fsyncs, validates with independent readers/runtime checks, and atomically publishes. Inputs are never overwritten.

Peak resident memory must be `O(metadata index + configured block chunk)`, never `O(model size)`. Tensor payloads are not collected into Python lists or full NumPy arrays. Cancellation removes or quarantines the incomplete staged file; it never promotes it.

## Decisions and routed follow-ups

- Container parsing, bounds, mmap behavior, endian detection and sparse 40 GB proof: #124 and #125.
- Exact codec identity, supported axes, block divisibility and error-estimation interface: #18.
- Architecture metadata/tensor naming and coupled edit groups: #19.
- Dense, Q8_0, K-quant and IQ codecs: #134–#141. No new issues are needed.
- Cross-language/upstream conformance vectors: #133; end-to-end codec/architecture compatibility: #161.
- Transactional streaming output, padding, metadata rewrite and atomic publication: #130.
- Optional external llama.cpp quantization tool discovery and revision capture: #160. External requantization is a fallback/validation path, not native surgery.
- Direct quantized feature extraction without full decode: #46.

There are no unresolved design questions in this spike. New upstream types or format versions are unsupported until their exact layout, reference codec, vectors, resource bound, and roadmap owner are added explicitly.
