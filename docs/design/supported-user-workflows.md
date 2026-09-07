# Supported consumer workflows

The supported consumer surface is intentionally evidence-bounded. A clean
installation can build a read-only optimize plan from an immutable source
revision using the documented Python API or CLI. The plan is the first
approval boundary: it resolves profile, quality, budget, lineage, rollback,
and uncertainty before any model or artifact write is considered.

Hugging Face/safetensors and native GGUF have separate guides because their
storage and execution contracts differ. The architecture compatibility matrix
is authoritative for family, dtype, quantization, and operation support. A
walkthrough uses a tiny licensed fixture or a user-supplied pinned source; it
does not claim broad support from a single model.

Every guide retains exact revisions, configuration, seed, hardware/runtime
details, command arguments, and failures. `supported` means the selected
contract is covered; `unknown` means evidence is missing; `unsupported` means
the capability is outside the verified boundary; and `failed` means a hard
validation or budget gate rejected the request. None of these outcomes is
converted into a silent fallback.
