# Runtime export capability matrix v1

`modelsurgeon.adapters.runtime_exports` is the format-neutral planning boundary
for exporting structural model states. It does not claim universal runtime
support. Every family × runtime × architecture-state cell is retained with a
verified, experimental, unsupported, failed, or unknown outcome.

## Evidence contract

An export request carries source artifact identity, converter revision, target
runtime revision, exact command, license, and resolved configuration digest.
Verified and experimental cells require retained evidence plus all provenance
fields. Missing tokenizer/generation/config metadata is `unknown`, not a
successful export. A capability matrix is complete over its declared finite
families, runtimes, and state traits.

The state traits distinguish uniform width, asymmetric width, low-rank factors,
quantization, and mixed codecs. The planner returns `unsupported` for target
runtimes without a verified representation for asymmetric, low-rank, or
mixed-codec states; it never silently densifies or approximates them.

The current bounded claims are verified Transformers metadata/save-reload
contracts, experimental pinned llama.cpp paths for Llama and Qwen, explicit
unsupported llama.cpp paths for other families, and unknown MLX/ONNX/vLLM
family reload paths until target-runtime evidence exists.
