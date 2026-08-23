# Hardware inventory contract

`collect_hardware_inventory` emits a versioned, JSON-safe reproducibility record for
the operating system, CPU architecture and logical cores, total and available RAM,
the selected scratch/output filesystem's total and free bytes, Python and
ModelSurgeon versions, and optional PyTorch/CUDA state.

The base path uses only the Python standard library and is valid when PyTorch, CUDA,
an NVIDIA driver, or a GPU is absent. If PyTorch is installed, the collector records
its package version, compiled CUDA version, runtime availability, and bounded per-GPU
name, memory, and compute-capability records. NVIDIA driver versions are queried with
a three-second, non-shell `nvidia-smi` call when that executable exists. Optional
runtime failures become deterministic warnings rather than invalidating CPU, RAM,
disk, OS, or software facts.

Callers must pass the actual destination or scratch path whose capacity constrains an
operation. Values are a point-in-time scheduling snapshot; experiment provenance
persists the complete record rather than assuming later host state is unchanged.
