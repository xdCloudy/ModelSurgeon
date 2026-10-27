# Representative model, format, and operation compatibility

Assessment: 2026-10-27 · weighted verified-support coverage: **48.5%** across 10 declared representative cells.

This is a curated, evidence-bounded set, not a Cartesian claim over every model x format x quantization x operation. Architecture, checkpoint format, quantization, and operation are explicit fields. `experimental`, `research_only`, `unknown`, and `unsupported` are not supported claims.

| Cell | Architecture / model scope | Format | Quantization | Operation | Status | Evidence | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hf-llama-mlp` | Llama / named tiny/pretrained checkpoints | Hugging Face/safetensors | FP16/BF16 | MLP channel surgery | Verified | [first-party-hf-acceptance-evidence-v3.json](../research/first-party-hf-acceptance-evidence-v3.json) | Exact tested checkpoints/configurations only. |
| `hf-mistral-mlp` | Mistral / named tiny/pretrained checkpoints | Hugging Face/safetensors | FP16/BF16 | MLP channel surgery | Verified | [first-party-hf-acceptance-evidence-v3.json](../research/first-party-hf-acceptance-evidence-v3.json) | Not all Mistral variants. |
| `hf-smollm-attn-layer` | SmolLM/SmolLM2 / named 135M checkpoints | Hugging Face/safetensors | FP16/FP32 | attention head and layer surgery | Experimental | [first-party-hf-acceptance-evidence-v4.json](../research/first-party-hf-acceptance-evidence-v4.json), [first-party-hf-acceptance-evidence-v5.json](../research/first-party-hf-acceptance-evidence-v5.json) | Some strict-gate failures retained. |
| `hf-qwen-mlp` | Qwen2/Qwen2.5 / named 0.5B checkpoints | Hugging Face/safetensors | FP16/FP32 | MLP surgery | Verified | [first-party-hf-acceptance-evidence-v3.json](../research/first-party-hf-acceptance-evidence-v3.json), [first-party-hf-chat-task-quality-acceptance-evidence-v1.json](../research/first-party-hf-chat-task-quality-acceptance-evidence-v1.json) | Small named checkpoint/task scope. |
| `gguf-llama-q4-mlp` | Llama / named Q4_K_M proof checkpoint | GGUF | Q4_K_M | bounded MLP channel surgery | Verified | [v0.7-native-q4-k-m-mlp-proof.md](../research/v0.7-native-q4-k-m-mlp-proof.md) | Pinned runtime/geometry and exact operation only. |
| `gguf-qwen-dense` | Dense Qwen / named dense Qwen target | GGUF | verified codecs only | bounded MLP optimizer path | Experimental | [first-party-native-gguf-campaign-v1.json](../research/first-party-native-gguf-campaign-v1.json) | Physical children failed quality gate. |
| `gguf-smollm-q4` | SmolLM2 / named 135M fixture | GGUF | Q4_K_M | MLP surgery | Experimental | [first-party-native-gguf-campaign-v1.json](../research/first-party-native-gguf-campaign-v1.json) | All measured candidates failed quality gate. |
| `gguf-qwen35-moe` | Qwen3.5 MoE / specific MoE checkpoint | GGUF | Q4_K_M | expert selection/slicing | Research only | [v1.1-structured-baseline-evidence-v1.json](../research/v1.1-structured-baseline-evidence-v1.json) | PRs #736/#738 are unmerged; not main support. |
| `hf-gemma` | Gemma / no representative physical checkpoint | Hugging Face/safetensors | unknown | structural surgery | Unknown | [architecture-compatibility.md](../architecture-compatibility.md) | Architecture recognition is not physical support. |
| `multimodal` | Multimodal / no representative bounded cell | HF or GGUF | unknown | cross-modal surgery | Unsupported | [goal-reality-gap-matrix.md](../research/goal-reality-gap-matrix.md) | No multimodal surgery claim. |

## Rules for expanding coverage

Add a representative cell only when an exact architecture/configuration/operation boundary has a concrete test or experiment plan. Promote it to verified only with physical artifact validation, reload/runtime evidence, and quality/resource measurements appropriate to the operation. Keep adjacent but untested combinations unknown or unsupported; never infer them from a family name.
