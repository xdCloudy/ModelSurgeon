# External llama.cpp quantization tooling

ModelSurgeon discovers or accepts an explicit `llama-quantize` executable; it does
not bundle llama.cpp binaries. The resolved quantizer must have a sibling
`llama-cli` in the same directory. Because `llama-quantize` has no version command
at the pinned upstream revision, ModelSurgeon uses that sibling as the binary-set
revision probe and requires commit
`95b8e33e16bb9a60de780a70930ebf729db6a90a`.

Discovery captures the resolved quantizer path and SHA-256, the sibling version
command/output, and the quantizer help command/output. The requested recipe must
appear in the tool's own advertised quantization types. Missing executables, an
incomplete binary directory, version drift, an unexpected help interface, and an
unsupported recipe all fail before a model command executes.

Quantization records the exact invocation, source identity, bounded stdout/stderr,
timeout/return state, and output identity. The external tool writes to a unique
partial file beside the requested destination. ModelSurgeon structurally validates
and hashes that GGUF, then publishes it with a no-overwrite hard link and removes
the owned partial name. Existing destinations are never replaced. Failed commands
retain their bounded logs and remove only the partial path created for that attempt.

The supported external recipes are `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, and
`BF16`. External recipe support is distinct from native edit-codec support.
Q4_K_M, for example, is a mixed recipe. The GGUF index recognizes the pinned
Q4_0/Q4_1/Q5_0/Q5_1 storage layouts so those tensors can be bounded and copied
byte-for-byte, but they are excluded from `QUANT_LAYOUTS`: ModelSurgeon makes no
native decode, encode, or surgery claim for those legacy formats.
