# GGUF architecture mapping contract

Status: accepted v1 contract

A resolved GGUF architecture contract is a versioned, framework-neutral mapping from explicit `general.architecture` metadata and a selected family to canonical components, storage-axis semantics, mutation coupling groups, metadata keys, and tensor renames. It contains no reader, tensor payload, or model object.

Llama, Qwen 2/3 and MoE spellings, Mistral, and Gemma 1/2/3 have explicit aliases. Current Mistral converters can emit the `llama` container architecture and key prefix, so `llama` without a family is intentionally ambiguous; callers must supply the already selected Mistral or Llama family and persist that evidence. Unknown architecture values and family/alias mismatches fail closed.

Common dense transformer rules map token/output embeddings, output/input/post-attention norms, Q/K/V/O attention projections, and gate/up/down MLP projections. GGML storage dimension zero is the contiguous input dimension for Q/K/V and gate/up, while their output dimension carries attention/KV-head or MLP-channel semantics. O and down projections consume those coupled structures on dimension zero. The contract exposes separate attention-head, KV-head, and MLP-channel groups for each block.

Metadata changes resolve to the selected architecture prefix for block, embedding, feed-forward, head, KV-head, and expert counts. Block renaming requires an explicit disposition for every referenced source index, preserves known non-block tensor names, validates the renamed tensor against the same contract, and uses `None` only for an explicitly removed block. Writers must check the complete renamed set for collisions before emitting #130's output.

Unmapped tensors remain safe for byte-preserving copy but cannot participate in a mutation until a versioned rule and fixture are added. MoE expert tensor axes require their dedicated downstream architecture fixtures before support is claimed; the alias alone does not silently apply dense MLP mutation rules to expert tensors.
