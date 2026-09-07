# Empirical hardware and runtime profile

`modelsurgeon.experiments.hardware_profile` adds the v1.3 evidence boundary above the
CPU-first `HardwareInventory`. A `HardwareProfile` binds the immutable inventory,
normalized hardware context, runtime revisions, benchmark settings, probe protocol, and
every probe outcome into one content-addressed record.

## Protocol

The preregistered protocol contains bounded memory-bandwidth, host/device-transfer,
representative-GEMM, model-load, and llama.cpp-offload probes. Each measured probe has at
least two warmups and seven repetitions and reports median, deterministic rank-based p95,
and population dispersion. Probe failures, unsupported runtimes, and unknown/unrun probes
are retained with reasons instead of being treated as zero measurements.

## Drift and unavailable capabilities

Every observation carries the normalized hardware context ID. A measured probe is rejected
if its context changes or if available environment fingerprints disagree across repetitions.
Optional thermal, clock, power, RAM, and VRAM fields remain null when the host cannot expose
them. CPU-only, low-VRAM, laptop, and workstation target classes therefore remain valid
without inventing GPU or sensor values.

The profile identity includes OS, CPU, memory, disk, CUDA devices and driver revisions,
software versions, runtime revisions, settings, protocol, and retained outcomes. It is
safe to use as a comparison/grouping key only when profile IDs match.
