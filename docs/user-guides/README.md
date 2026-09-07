# End-to-end user guides

These guides describe the two supported ModelSurgeon workflows from installation through validation and recovery:

- [Hugging Face and safetensors](huggingface.md): inspect a pinned model, run a bounded first experiment, and publish a physically edited checkpoint.
- [Native GGUF](native-gguf.md): validate a pinned quantized model, perform a bounded Q4_K_M edit, resume an interrupted write, and validate the result with llama.cpp.
- [Stable Python API and notebook](python-api.md): build a deterministic read-only plan without hidden downloads, source overwrite, or credentials.
- [Consumer troubleshooting](troubleshooting.md): interpret supported, unknown, unsupported, and failed outcomes and recover safely.
- [Hardware and quality profiles](profiles.md): choose the bounded CPU, low-VRAM, and quality envelopes used by the planner.

Both guides are for the current v1.0.0 evidence-bounded research release surface. They deliberately use new output paths, immutable source revisions, bounded workloads, and fail-closed validation. They do not imply that every model family, quantization, or external runtime is supported; check the [compatibility matrix](../architecture-compatibility.md) and retain exact evidence for the model and tool versions you use.

