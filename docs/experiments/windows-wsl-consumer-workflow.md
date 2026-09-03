# Windows 11 and WSL2 consumer workflow

This runbook separates portable contract coverage from a machine-specific CUDA/GGUF
smoke. A passing portable test suite does not establish that a particular GGUF model
loads or offloads on a particular Windows or WSL2 host; retain the JSON report and
the exact model, tool, driver, and command identities for that claim.

## Verified reference run

On 2026-09-03, the following focused suite passed on both the native Windows checkout
and Ubuntu 24.04 under WSL2:

```text
pytest tests/test_experiment_identity.py tests/test_artifact_store.py \
  tests/test_checkpoint_destination.py tests/test_gguf_disk.py \
  tests/test_gguf_resume.py tests/test_runtime_telemetry.py \
  tests/test_runtime_io_counters.py -q
```

Both environments reported `35 passed`. This covers path and artifact identity,
transactional checkpoint interruption, bounded GGUF disk preflight and resumable
writes, and process I/O/runtime telemetry. The WSL2 host exposed an NVIDIA GeForce
RTX 3060 with 12 GiB of VRAM through `nvidia-smi`. The local Windows
`llama-cli.exe` reported version `0.3.0-dev`, build `10729`, commit `458681e1d`.

This is intentionally recorded as contract evidence, not as native-GGUF CUDA-offload
evidence: no licensed local GGUF fixture was available and the project Windows virtual
environment did not have PyTorch installed. Consequently, no model was loaded with
CUDA layers and issue #200 remains open until the smoke below is retained.

## Run the portable checks

From an elevated PowerShell only when your environment requires it, use the project
virtual environment rather than a system Python:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_experiment_identity.py tests/test_artifact_store.py tests/test_checkpoint_destination.py tests/test_gguf_disk.py tests/test_gguf_resume.py tests/test_runtime_telemetry.py tests/test_runtime_io_counters.py -q
```

For WSL2, keep the checkout on the Windows filesystem only if the performance cost is
acceptable; use the same command with the WSL virtual environment and `PYTHONPATH=src`
when running an editable-less setup:

```bash
cd /mnt/c/Users/Cloudy/ModelSurgeon
PYTHONPATH=src /path/to/venv/bin/python -m pytest \
  tests/test_experiment_identity.py tests/test_artifact_store.py \
  tests/test_checkpoint_destination.py tests/test_gguf_disk.py \
  tests/test_gguf_resume.py tests/test_runtime_telemetry.py \
  tests/test_runtime_io_counters.py -q
```

## Native GGUF and CUDA-offload evidence

Use a licensed, revision-pinned GGUF fixture. Do not overwrite a source model or
publish a generated artifact over an existing result. Build `llama.cpp` from the
revision required by the compatibility contract, prepare the bounded fixtures, then
run the matrix and retain its JSON output:

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git .llama.cpp
git -C .llama.cpp checkout 95b8e33e16bb9a60de780a70930ebf729db6a90a
cmake -S .llama.cpp -B .llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .llama.cpp/build --config Release --target llama-cli llama-quantize -j 2
python tools/prepare_gguf_compatibility.py --llama-cpp .llama.cpp --output .compatibility --quantize .llama.cpp/build/bin/llama-quantize
```

Run one explicit load with GPU layers enabled and retain stdout/stderr, the model
SHA-256, `nvidia-smi`, the `llama-cli --version` output, and the full argument vector.
For example, replace the placeholder path and layer count with the fixture's known
values:

```bash
.llama.cpp/build/bin/llama-cli --model /absolute/path/to/fixture.gguf --gpu-layers 99 --prompt "ModelSurgeon CUDA smoke" --n-predict 8 --no-warmup
```

Then run `tools/run_gguf_compatibility.py` with the exact fixture declarations used by
the scheduled [GGUF compatibility workflow](../../.github/workflows/gguf-compatibility.yml).
The matrix proves structural compatibility; the explicit `--gpu-layers` invocation
proves the host-specific offload path. Treat an unavailable CUDA runtime, a CPU fallback,
or a missing fixture as a failed or incomplete evidence run rather than a pass.

## Platform differences

- Windows uses a writable descriptor for the checkpoint `fsync` path; this is covered
  by the platform-specific implementation in `checkpoint_destination.py`.
- Windows and WSL2 share a CUDA driver, but their Python, PyTorch, `llama.cpp`, and
  filesystem paths are separate identities. Record them independently in persisted
  provenance.
- WSL paths use `/mnt/<drive>/...`; Windows executable paths require `.exe`. Never
  reuse one platform's command line verbatim on the other.
- mmap semantics and filesystem throughput can differ when a WSL process accesses the
  mounted Windows drive. Keep both the source fixture and generated artifacts on the
  same filesystem during a measurement, and record that location with the result.
- Interrupts must leave no published partial checkpoint. Re-run the resume test after
  changing storage location or filesystem implementation.
