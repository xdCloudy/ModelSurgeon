# EvalPlus pruning study

This optional study compares a pinned Hugging Face source checkpoint with a physically
MLP-pruned copy using one greedy completion for every HumanEval+ task. EvalPlus handles
code generation and execution; ModelSurgeon performs the structural edit through its
existing Hugging Face MLP surgery adapter. No EvalPlus source is copied into ModelSurgeon.

## Pinned sources

- EvalPlus: [`e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`](https://github.com/evalplus/evalplus/tree/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2), Apache-2.0.
- HumanEval+: `v0.1.10`, release commit `200defce9e3429d28ca215b6dd061c0f7f31c18b`, Apache-2.0.
- Model: [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/tree/ea3f2471cf1b1f0db85067f1ef93848e38e88c25), revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, Apache-2.0.

EvalPlus supplies generation, sanitization, and pass@1 evaluation. The independent
ModelSurgeon selector assigns each gated MLP channel a global score by summing, across
layers, the product of the gate-row L2 norm, up-row L2 norm, and down-column L2 norm.
The default removes the 256 lowest-scoring channel indices from every layer, then saves
and reloads the narrowed checkpoint before evaluation.

## Run

Use Python 3.12 and a CUDA-enabled PyTorch environment for generation. Install the
optional pinned evaluator with `uv sync --locked --extra hf --extra evalplus`. Build the
Linux evaluator image from a small context containing only the project metadata and this
Dockerfile:

```powershell
$context = Join-Path $env:TEMP ("modelsurgeon-evalplus-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $context | Out-Null
Copy-Item pyproject.toml, uv.lock -Destination $context
Copy-Item tools/evalplus/Dockerfile -Destination (Join-Path $context 'Dockerfile')
docker build --platform linux/amd64 --tag modelsurgeon-evalplus:e5d0ed0 $context
```

Then run:

```powershell
python tools/run_evalplus_hf_pruning_study.py `
  --output reports/benchmarks/qwen25_coder_05b_humanevalplus_mlp_pruning.json
```

The runner refuses to overwrite a report. It records source revisions, dataset and
artifact digests, parameter counts, generated-sample digests, per-task pass status, and
source-relative pass@1 retention. It does not publish generated code. The evaluator
container has networking disabled, a read-only root and input mount, no Linux
capabilities, a non-root uid, a 3 GiB memory limit, and a 2 CPU limit. Generated code is
removed from the temporary workspace after the study.

EvalPlus's HF decoder is explicitly put in base-prompt mode because HumanEval is a
function-completion task and the pinned EvalPlus revision does not provide an instruction
prefix for this Qwen chat template. Decoding is greedy, one sample per task, capped at 256
new tokens. The upstream decoder currently emits a Transformers attention-mask warning
for its single-prompt path; treat that as a protocol limitation when interpreting scores.

This single-model, single-severity experiment tests executable correctness and physical
parameter reduction. It does not establish generalization, contamination-free coding
ability, latency improvement, or broad quality retention. A zero source pass@1 makes a
retention ratio inconclusive; the runner reports that explicitly.
