# Configuration Schema

Status: implemented for the v2.2 direct API/provider boundary; local-first
first-run setup is documented in [the setup contract](first-run-setup.md).

ModelSurgeon configuration is a versioned, immutable hierarchy implemented by `modelsurgeon.config.Settings`. Every section rejects unknown keys so misspellings cannot silently change an experiment.

## Sections

- `model`: source path/ID, immutable revision, container format, and requested compute dtype.
- `calibration`: dataset identity, split, bounded samples/batch/sequence length, and seed.
- `features`: independently enabled weight, spectral, activation, gradient, correlation, topology, and runtime groups.
- `repair`: explicit no-repair control or bounded first-party LoRA/teacher-logit distillation settings and target modules.
- `quantization`: explicit no-quantization control or the CPU-scoped first-party `dynamic_int8` HF path.
- `objective`: quality and resource constraints plus an ordered, duplicate-free set of optimization dimensions.
- `hardware`: full/tensor/streaming/automatic memory mode, RAM/VRAM ceilings, CPU offload, and mixed precision.
- `safety`: overwrite, remote-code, and atomic-write policy.
- `provider`: optional conversational provider identity, endpoint metadata, and
  hard request budgets. Its default is `kind=none`; the section contains no
  secret values.

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

`discover_settings()` exposes the same resolved `Settings` object together with
a value-free, ordered source record (`defaults`, `file`, `environment`, `cli`)
and a stable configuration digest. Source keys are sorted dotted paths; secret
values and local paths are not copied into the discovery record. The CLI and
Python diagnostics use this exact discovery result, so a diagnostic bundle can
be reproduced without relying on process-global ordering.

## Canonical form

`Settings.canonical_dict()` converts paths and enums into JSON-compatible values. `canonical_json()` uses sorted keys, compact separators, UTF-8 text, and the explicit `schema_version`. This form is suitable as an input to deterministic run IDs and reproducibility manifests. It does not include secrets because the v1 schema has no secret-bearing fields.

## Optional provider and no-LLM mode

The provider section is independent of optimization policy. The default is an
explicit no-LLM configuration:

```yaml
provider:
  kind: none
  provider_id: none
  model_id: none
  model_revision: none
```

`modelsurgeon optimize --no-llm` applies the same settings at the highest
configuration-precedence level. Provider selection cannot change hard model,
quality, resource, or safety constraints. Local selection may additionally set
`model_path` and `runtime_revision`; remote selection uses `endpoint` and an
optional `api_key_env` reference. Provider endpoints must be absolute HTTP(S)
URLs without credentials, query strings, or fragments; API keys are referenced
only by an uppercase environment-variable name and are never included in
canonical settings.

Use `modelsurgeon provider diagnostics --json` to inspect the selected mode
without starting a model or contacting a remote endpoint. The record includes
deterministic configuration discovery, a capability summary, and explicit
`detected`, `measured`, `configured`, `unsupported`, or `unknown` cells. A
missing local model/runtime, missing remote key, offline remote selection, or
unprobed remote capability is retained as a stable negative/unknown result;
the diagnostic does not substitute a provider or silently downgrade it. Direct
CLI/Python optimization does not resolve a provider and therefore remains
usable in a clean core-only environment.

## First-run data configuration

`modelsurgeon setup diagnostics` and `modelsurgeon setup init` manage the
user-owned mutable data root separately from optimization settings. The setup
manifest records the data location, offline flag, local fixture manifest, and
declared free-space budget; it does not contain credentials and is not accepted
as a substitute for the existing `--config` settings file. The setup command
never downloads models or installs optional runtimes. See the [first-run setup
contract](first-run-setup.md) for the Windows/local/offline matrix and the
explicit unsupported cells.

