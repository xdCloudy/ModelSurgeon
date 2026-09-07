# End-to-end user guides

These guides describe the two supported ModelSurgeon workflows from installation through validation and recovery:

- [Hugging Face and safetensors](huggingface.md): inspect a pinned model, run a bounded first experiment, and publish a physically edited checkpoint.
- [Native GGUF](native-gguf.md): validate a pinned quantized model, perform a bounded Q4_K_M edit, resume an interrupted write, and validate the result with llama.cpp.

Both guides are for the current pre-alpha release surface. They deliberately use new output paths, immutable source revisions, bounded workloads, and fail-closed validation. They do not imply that every model family, quantization, or external runtime is supported; check the [compatibility matrix](../architecture-compatibility.md) and retain exact evidence for the model and tool versions you use.

