# Matched Requantization Controls

Structural measurements on quantized models conflate two effects: damage from
changing the model graph and damage from decoding and requantizing affected
weights. Issue #156 defines a matched no-surgery control so evaluation can
attribute those effects separately.

## Control construction

The control consumes the same validated quantized mutation plan used by surgery.
For every tensor it replaces the destination shape with the original shape while
retaining the plan's selective repack spans. It then streams those spans through
the same selective decoder and exact-codec requantizer. No tensor shape, metadata,
or component identity changes.

The current control deliberately rejects codec substitutions. Using the identical
source and destination codec keeps source and output block coordinates comparable
and prevents a codec-family change from becoming an unreported experimental
variable. Native codecs are deterministic; a required unsigned 64-bit seed is
still recorded for experiment identity and future stochastic codec contracts.

After the stream completes, adjacent chunks are normalized into canonical codec
block ranges. The control fails unless decoded and encoded ranges match exactly.
Reports include those ranges, seed, byte and block counts, quantization errors,
and decoder/encoder peak working-memory observations.

## Delta attribution

For a scalar metric with baseline `B`, matched control `C`, and surgery result
`S`, reports expose:

- requantization delta: `C - B`;
- surgery delta: `S - C`;
- combined delta: `S - B`.

The first two terms reconcile to the combined delta. All measurements must be
finite and share the same named metric; dataset, tokenizer, context, and runtime
matching remain responsibilities of the evaluation runner issues that consume
this contract.
