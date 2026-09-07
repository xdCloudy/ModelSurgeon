# Hugging Face and safetensors guide

This guide walks through a small, reproducible Hugging Face workflow. It uses the CLI for inspection and evidence-producing experiments, then the typed Python APIs for physical edits. Physical edits are intentionally not hidden behind a convenience command: the caller must choose the targets, destination, budgets, and validation steps explicitly.

## 1. Install and choose a disposable workspace

Use Python 3.12 or newer and a locked uv environment. The hf extra installs the optional Transformers, PyTorch, Accelerate, and safetensors boundary.

~~~bash
git clone https://github.com/xdCloudy/ModelSurgeon.git
cd ModelSurgeon
uv sync --extra dev --extra hf --locked
uv run modelsurgeon --help
~~~

Keep downloaded models, generated checkpoints, logs, and reports below an ignored work/ or artifacts/ directory. Never use the source model directory as an output directory. The source checkpoint is an immutable input and every published artifact must have a different path.

## 2. Pin and inspect a model

Use an immutable Hub commit rather than a moving branch or tag. The revision below is a placeholder; replace it with the commit you actually selected and record it with the result.

~~~bash
MODEL_ID=HuggingFaceTB/SmolLM2-135M
MODEL_REVISION=<40-character-Hub-commit>
mkdir -p work/hf-smoke

uv run modelsurgeon inspect "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --device-map cpu \
  --dtype auto \
  --json > work/hf-smoke/inspection.jsonl
~~~

inspect loads the model without changing it and records the resolved revision, loader controls, architecture family, parameter count, and canonical component IDs. Review the JSON before choosing a mutation. Remote model code is disabled by default; only add --trust-remote-code after reviewing and recording the source and license.

For a cached local snapshot, pass the snapshot path and add --local-files-only to the proof command below. Do not treat a local path as a substitute for recording the original Hub revision.

## 3. Run a bounded first experiment

Create a small UTF-8 calibration text file from data you are allowed to use. Keep the run bounded while developing; increase the budget only after the contract and provenance are correct.

~~~bash
uv run modelsurgeon first-surgeon-hf-proof "$MODEL_ID" ./calibration.txt \
  --revision "$MODEL_REVISION" \
  --device-map cpu \
  --dtype auto \
  --output work/hf-smoke/proof-data \
  --sequence-length 32 \
  --max-tokens 64 \
  --max-candidates 3000 \
  --safe-perplexity-delta 0 \
  --seed 42 \
  --split-seed 43 \
  --tool-revision "$(git rev-parse HEAD)" \
  --json > work/hf-smoke/proof-result.json
~~~

The proof workflow applies reversible MLP-channel masks, measures the selected quality signal, and writes a leakage-audited dataset. A failed split or leakage check refuses publication. Use --device-map auto and an explicitly selected low-precision dtype only after validating the same model on the target GPU.

For a resumable calibration manifest, use the strict plan described in the [calibration CLI contract](../design/calibration-cli.md):

~~~bash
uv run modelsurgeon calibrate calibration-plan.json \
  --cache work/hf-smoke/calibration-cache.json \
  --dry-run
uv run modelsurgeon calibrate calibration-plan.json \
  --cache work/hf-smoke/calibration-cache.json
~~~

--dry-run performs no dataset or cache write. A normal run publishes the cache atomically; --refresh rebuilds it without replacing a valid cache if the new run is interrupted.

## 4. Publish a physical safetensors edit

Physical editing is a library operation. Load a local, revision-pinned snapshot, apply one supported edit, and publish through the atomic safetensors writer. The following example removes selected MLP channels from every adapter-defined transformer layer. It is illustrative: the indices must be justified by a measured experiment and must leave a valid width.

~~~python
from pathlib import Path

from modelsurgeon.adapters.huggingface import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
    remove_huggingface_mlp_channels,
)
from modelsurgeon.adapters.safetensors import write_safetensors_checkpoint_atomic

source = Path("/data/models/smollm2-135m")
destination = Path("work/hf-smoke/candidate-mlp")
if destination.exists():
    raise FileExistsError(destination)

loaded = load_causal_lm(
    HuggingFaceLoadRequest(
        model=str(source),
        revision="<resolved-local-snapshot-identity>",
        local_files_only=True,
        device_map="cpu",
        dtype=HuggingFaceDType.FLOAT32,
    )
)

result = remove_huggingface_mlp_channels(loaded.model, (0, 4))
cpu_state = {
    name: tensor.detach().to(device="cpu").contiguous()
    for name, tensor in loaded.model.state_dict().items()
}
report = write_safetensors_checkpoint_atomic(
    source=source,
    destination=destination,
    tensors=cpu_state,
    configuration=loaded.model.config.to_dict(),
)
print(result.to_record())
print(report.destination, report.source_shard_sha256)
~~~

The writer stages shards beside the destination, verifies tensor shapes, dtypes, sizes, and hashes, checks that the source did not change, and publishes with atomic no-replace semantics. An existing destination, source alias, invalid tensor, or verification disagreement fails closed. Save the tokenizer and its revision as separate provenance only after choosing a new destination; never overwrite the source tokenizer or checkpoint.

Attention-head and transformer-layer edits use the corresponding remove_huggingface_attention_heads and remove_huggingface_transformer_layers APIs. Complete GQA groups are required for attention edits, and layer removal returns an explicit old-to-new identity mapping. Run a dry plan or reversible masked experiment before publishing any physical edit.

## 5. CPU, 12 GB GPU, and failure recovery

- CPU-only: use --device-map cpu, --dtype auto, short sequence lengths, and small token/candidate budgets. This is the most portable validation path.
- 12 GB GPU: use --device-map auto, record the device and driver, and bound sequence length, calibration tokens, batch size, and candidate count. Keep enough host RAM and disk for the staged destination and cache.
- Out of memory: reduce the batch or token budget and retry from the same immutable source. Do not resume a run after changing its model revision, preprocessing contract, or output identity.
- Invalid or unsupported model: keep the error and resolved revision as evidence. Do not enable remote code or add an unreviewed adapter just to make the run pass.
- Interrupted publication: remove only the failed destination staging area after inspecting the retained error. A valid published destination is never partially replaced.

## 6. Inspect, report, and reproduce

Persisted experiments can be inspected without execution and replayed only through an explicitly trusted local adapter:

~~~bash
uv run modelsurgeon reproduce run_<sha256> \
  --metadata work/experiments.sqlite3 \
  --artifacts work/artifacts \
  --repository . \
  --lock uv.lock \
  --dry-run
~~~

The dry run reports the exact model, dataset, mutation, tool, seed, hardware, lock, command arguments, original metrics, and mismatches. It never executes stored command text. See [reproducing persisted runs](../experiments/reproduce-run.md) for trusted replay and tolerance rules.

## Supported-claim boundary

The [architecture compatibility matrix](../architecture-compatibility.md) and persisted run evidence are authoritative. A successful toy fixture or one model family does not establish support for every Transformers architecture, dtype, device map, or checkpoint layout.

