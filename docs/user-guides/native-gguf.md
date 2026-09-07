# Native GGUF guide

This guide covers bounded, out-of-core edits to a quantized GGUF. Native GGUF surgery maps the file, copies untouched encoded tensors, selectively decodes affected blocks, and streams a new file. It does not materialize a full floating-point model. Read the [compatibility matrix](../architecture-compatibility.md) before selecting a family or quantization.

## 1. Prepare a licensed, pinned fixture

Keep the source GGUF and every generated file in disposable, ignored storage. Record the model repository, license, Hub revision, filename, SHA-256, GGUF architecture, quantization, and the exact ModelSurgeon and llama.cpp revisions.

Install the project development environment:

~~~bash
git clone --filter=blob:none https://github.com/xdCloudy/ModelSurgeon.git
cd ModelSurgeon
uv sync --extra dev --locked
mkdir -p work/gguf-smoke
~~~

For host-specific validation, build the pinned llama.cpp revision required by the compatibility contract. The build is an external tool, not a replacement for ModelSurgeon's parser and writer checks:

~~~bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git .llama.cpp
git -C .llama.cpp checkout 95b8e33e16bb9a60de780a70930ebf729db6a90a
cmake -S .llama.cpp -B .llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .llama.cpp/build --config Release --target llama-cli llama-quantize -j 2
~~~

CPU-only validation may build without CUDA. A CPU pass does not establish CUDA offload support, and a CUDA fallback must be recorded as incomplete rather than reported as a GPU result.

## 2. Inspect before editing

The stable CLI does not yet expose a generic native-GGUF surgery command. Use the typed adapter boundary from a small script so the family, layout, and quantization decisions are explicit:

~~~python
from pathlib import Path

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import discover_gguf_components, open_gguf

source = Path("/data/models/SmolLM2-135M-Q4_K_M.gguf")
with open_gguf(source) as mapped:
    discovery = discover_gguf_components(mapped.container, family=ModelFamily.LLAMA)
    print(discovery.family, discovery.shape)
    for tensor in discovery.tensors:
        print(tensor.descriptor.name, tensor.descriptor.dimensions)
~~~

Unknown architecture, missing metadata, malformed offsets, unsupported codecs, or inconsistent tensor shapes are hard errors. Do not guess the family from the filename.

## 3. Plan and execute a bounded Q4_K_M MLP edit

The example below removes a codec-aligned channel group across every layer. Choose the indices from an evidence-backed mutation plan; the planner rejects non-canonical indices, invalid geometry, and edits that remove the complete feed-forward width. Register every codec required by the changed tensors. A Q4_K-only example registers Q4_K_CODEC; mixed-codec files need the matching registry entries too.

~~~python
from pathlib import Path

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    CodecRegistry,
    GGUFDiskEstimate,
    Q4_K_CODEC,
    discover_gguf_components,
    open_gguf,
    preflight_gguf_disk,
)
from modelsurgeon.surgery import (
    execute_native_gguf_mlp_channel_removal,
    plan_native_gguf_model_mlp_channel_removal,
)

source = Path("/data/models/SmolLM2-135M-Q4_K_M.gguf")
destination = Path("work/gguf-smoke/SmolLM2-135M-Q4_K_M-mlp-candidate.gguf")
if destination.exists():
    raise FileExistsError(destination)

with open_gguf(source) as mapped:
    discovery = discover_gguf_components(mapped.container, family=ModelFamily.LLAMA)
    plan = plan_native_gguf_model_mlp_channel_removal(
        discovery,
        removed_channels=tuple(range(0, 256)),
    )
    disk = preflight_gguf_disk(
        destination,
        destination.parent,
        GGUFDiskEstimate(
            output_bytes=source.stat().st_size * 2,
            scratch_bytes=0,
            alignment_bytes=mapped.container.alignment,
        ),
    )
    codecs = CodecRegistry()
    codecs.register(Q4_K_CODEC)
    result = execute_native_gguf_mlp_channel_removal(
        mapped,
        plan,
        destination,
        disk,
        codecs,
    )
    print(result.output_discovery.shape)
    print(result.write_result.sha256)
~~~

The disk estimate is intentionally conservative. Keep output and scratch on a filesystem with enough free space for the estimate and safety margin. The writer verifies the input digest, output layout, changed tensor shapes, and byte identity of untouched tensors before publication. It never writes the source and never replaces an existing destination.

The same execution boundary supports attention-head removal, transformer-layer removal, and low-rank replacement. Each operation has its own planner and compatibility contract; do not reuse an MLP plan for another mutation kind.

## 4. Resume and discard safely

GGUF writes checkpoint only fsynced tensor boundaries in private staging and manifest files beside the destination. If the process is interrupted, rerun the exact same script with the same source, plan, destination, codecs, and limits. The writer verifies the input and plan identities before resuming.

If the operation is no longer wanted, discard only the private artifacts for that exact destination:

~~~python
from modelsurgeon.adapters.gguf import discard_resumable_gguf

discard_resumable_gguf("work/gguf-smoke/SmolLM2-135M-Q4_K_M-mlp-candidate.gguf")
~~~

Changed source bytes, changed layout, stale manifests, or corrupted committed prefixes fail closed. Do not copy a staging file into place manually and do not resume with a different source or mutation plan.

## 5. Validate the published artifact

First reopen the output through ModelSurgeon's parser and compare the reported shape and tensor layout with the plan. Then validate with the exact external runtime revision selected for the evidence run:

~~~bash
.llama.cpp/build/bin/llama-cli \
  --model work/gguf-smoke/SmolLM2-135M-Q4_K_M-mlp-candidate.gguf \
  --gpu-layers 99 \
  --prompt "ModelSurgeon GGUF smoke" \
  --n-predict 8 \
  --no-warmup
~~~

Retain the complete command, runtime version and commit, model SHA-256, GPU and driver (if used), stdout/stderr, exit status, and any compatibility report. Run the CPU path separately when CPU fallback is part of the claim. WSL2 and Windows are separate runtime identities; record their paths, Python/runtime versions, and filesystem locations independently.

For a Q4_K_M physical edit, compare baseline, surgery, and (where applicable) matched requantization controls. A smaller file alone is not evidence of a useful model. Retain quality, reload, throughput, memory, disk, and failure results, including negative or inconclusive outcomes.

## 6. CPU, 12 GB GPU, and failure recovery

- CPU-only: keep the model and output on the same filesystem, use small fixtures first, and validate with a CPU llama.cpp build. Native GGUF edits remain bounded by row/chunk limits rather than full-model RAM.
- 12 GB GPU: use --gpu-layers only after confirming the target runtime and model fit. Start with a tiny licensed Q4_K_M fixture and a short generation; record VRAM and offload logs.
- Disk exhaustion: preflight output and scratch paths before execution; never delete an unrelated file to make space. An incomplete publication is not a usable model.
- Unsupported layout or codec: stop at the explicit error. Do not infer a codec, reinterpret tensor axes, or convert the whole model to float as an undocumented workaround.
- Runtime failure: retain the artifact digest, command, and logs. A parser pass without an external runtime load is only structural evidence.

## 7. Reproduce the surrounding experiment

If the edit is part of a persisted experiment, inspect the immutable run before replaying it:

~~~bash
uv run modelsurgeon reproduce run_<sha256> \
  --metadata work/experiments.sqlite3 \
  --artifacts work/artifacts \
  --repository . \
  --lock uv.lock \
  --dry-run
~~~

Replay requires a trusted local adapter and rejects missing evidence, changed inputs, corrupt artifacts, unsafe environment drift, and tolerance failures. The command text stored in evidence is never executed automatically.

## Supported-claim boundary

The [architecture compatibility matrix](../architecture-compatibility.md), codec conformance vectors, and retained run manifest define the claim. The repository's Q4_K_M proof is specifically bounded to its measured family and runtime revision; it is not a blanket claim for every GGUF architecture or quantization.

