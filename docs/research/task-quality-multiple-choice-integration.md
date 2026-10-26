# Multiple-choice task-quality integration

## Technology acquisition record

| Field | Evidence |
| --- | --- |
| Source | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), commit `d6de81643928d653435c431bae19945d41d32520`; `lm_eval/api/model.py` (`TemplateLM._encode_pair` and `loglikelihood`) and `docs/new_task_guide.md` (multiple-choice output). |
| Component | Teacher-forced causal continuation likelihood and finite-choice accuracy. |
| Licence | MIT at the reviewed repository revision. No source code was copied or imported; no new dependency or third-party notice is required. |
| Current ModelSurgeon equivalent | `code_exact_match_v2` generated-code evaluation, which cannot score classification, knowledge, or other fixed-choice tasks. Perplexity alone is not a task-specific accuracy metric. |
| Technical advantages | Compares each answer using the model's probability for the full continuation; avoids sampling variance from generated answers; stores all per-choice scores and ties to the dataset digest. Tokenization is performed on the concatenated prompt and choice, then split by the separately tokenized prompt length to preserve causal token boundaries. |
| Hardware requirements | Existing optional Hugging Face dependencies (PyTorch, Transformers, tokenizer). CPU is supported; CUDA uses the model's current input device. Each example-choice pair makes one forward pass. |
| Supported architectures | Causal Hugging Face models already supported by `HuggingFaceMLPProofRuntime`; no new model-family adapter is introduced. Encoder-only and native GGUF task-quality evaluation remain unsupported by this method. |
| Integration approach | **REIMPLEMENT** the documented likelihood-scoring principle in the existing ModelSurgeon task-quality and physical measurement path. Preserve the existing code-exact-match schema and gate. No lm-evaluation-harness runtime or task code is vendored. |
| Expected benefit | Hypothesis: user-supplied finite-choice tasks provide a more direct quality guard than perplexity for some workloads. Success requires a pinned real-model campaign where baseline and physical candidates are scored identically and the declared retention gate rejects known regressions. This implementation alone claims no quality or optimization improvement. |
| Risk | Results depend on prompt and choice formatting. Summed sequence log-likelihood favors shorter choices in some task designs; this evaluator does not implement length-normalized accuracy. The current evaluator has numerical unit tests but no retained real-model optimization campaign. |
| Validation | Synthetic logits verify the selected option and summed score path; malformed datasets, empty-baseline gating, config limits, and first-party gate integration are tested. A pinned real-model campaign is required before treating this path as experimentally validated. |
| Priority | P1: extends the task-specific quality guard while the higher-priority physical optimizer and Meta-Surgeon improvements continue. |

## Boundaries

The local dataset is capped at 16 MiB and 4,096 examples. Each row permits at most
16 choices and 16,384 characters per prompt/choice. The evaluator caps each
prompt-choice sequence at 8,192 tokens (4,096 by default), all choices combined
at 16,384 model forwards, and total scored input at one million tokens. Budgets
are checked before the first model forward. The runtime records the dataset
revision, split, digest, correct and predicted indices, and every choice's
summed conditional log-likelihood. Perplexity and any configured task score
remain separate hard gates.
