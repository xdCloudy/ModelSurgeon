# GGUF quantization codec contract

Status: accepted v1 contract

Every GGUF tensor resolves to one exact `GGMLQuantizationType` and `CodecLayout`. A layout fixes its family, logical block size, encoded type size, ordered byte fields, and layout version. The registry resolves only exact types: Q4_K cannot satisfy Q5_K, Q8_K cannot satisfy Q8_0, and a whole-file recipe such as Q4_K_M is never treated as a tensor codec.

`QUANT_LAYOUTS` pins F32, F16, BF16, Q8_0, Q2_K through Q8_K, and the prioritized IQ layouts to the #17 source revision. Ordered `BlockField` records expose every scale, minimum, high-bit, sign, codebook-index, quant, and block-sum byte range. Codec implementations must match this record exactly before registration.

GGML dimensions use the contiguous storage dimension at index zero. `plan_axis_edit` rejects empty/non-positive shapes, partial contiguous blocks, invalid axes, and encoded sizes beyond the signed 64-bit range. `plan_supported_axes` returns the complete per-axis report. It includes row/tensor byte counts and the supported edit rule:

- axis zero edits operate on complete codec blocks and have index granularity equal to the block size;
- higher-axis edits operate on whole encoded slices and have index granularity one;
- dense scalar layouts have block size one and therefore allow element-granular axis-zero edits.

`validate_tensor_alignment` keeps codec planning tied to the container contract: tensor offsets must be non-negative multiples of a power-of-two alignment between 8 bytes and the v1 safety cap of 1 MiB. The default remains 32 when GGUF metadata omits it.

The `QuantizationCodec` protocol validates encoded blocks and decodes/encodes caller-provided bounded buffers with explicit byte order. Operations report blocks, logical elements, and bytes processed. Implementations cannot return a full tensor implicitly. `QuantizationError` provides common finite-sample mean absolute, mean squared, maximum absolute, and reference-L2 metrics; codec-specific reports may extend persistence records later without changing these definitions.

Malformed/partial blocks, unsupported byte order, shape divisibility failure, missing exact codecs, and identity/layout disagreement fail before mutation. Codec implementations and upstream vectors remain in #133–#141; this issue defines their shared boundary without pre-claiming support.
