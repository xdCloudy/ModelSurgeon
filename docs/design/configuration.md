# Configuration Schema

Status: Accepted for v0.1

ModelSurgeon configuration is a versioned, immutable hierarchy implemented by `modelsurgeon.config.Settings`. Every section rejects unknown keys so misspellings cannot silently change an experiment.

## Sections

- `model`: source path/ID, immutable revision, container format, and requested compute dtype.
- `calibration`: dataset identity, split, bounded samples/batch/sequence length, and seed.
- `features`: independently enabled weight, spectral, activation, gradient, correlation, topology, and runtime groups.
- `objective`: quality and resource constraints plus an ordered, duplicate-free set of optimization dimensions.
- `hardware`: full/tensor/streaming/automatic memory mode, RAM/VRAM ceilings, CPU offload, and mixed precision.
- `safety`: overwrite, remote-code, and atomic-write policy.

Safe defaults prohibit checkpoint overwrite and remote model code, require atomic writes, enable CPU offload, and target 98% quality retention. Resource limits must be positive, probabilities remain within `[0, 1]`, sample limits are positive, and seeds are non-negative.

## Environment overrides

Environment variables use the `MODELSURGEON_` prefix and `__` between nested fields:

```text
MODELSURGEON_HARDWARE__MAX_VRAM_GB=11
MODELSURGEON_HARDWARE__MEMORY_MODE=streaming
MODELSURGEON_SAFETY__TRUST_REMOTE_CODE=false
```

`load_settings()` reads UTF-8 YAML or TOML and applies sources in this order:

```text
schema defaults < configuration file < environment < CLI overrides
```

CLI integrations pass dotted overrides such as `hardware.max_vram_gb`; nested siblings are merged rather than erased. Unsupported extensions, non-mapping roots, invalid UTF-8, and parse failures raise `ConfigurationFileError` before schema validation. `dump_resolved_settings()` emits canonical JSON with no secret-bearing schema fields.

## Canonical form

`Settings.canonical_dict()` converts paths and enums into JSON-compatible values. `canonical_json()` uses sorted keys, compact separators, UTF-8 text, and the explicit `schema_version`. This form is suitable as an input to deterministic run IDs and reproducibility manifests. It does not include secrets because the v1 schema has no secret-bearing fields.

