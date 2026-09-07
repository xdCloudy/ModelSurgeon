# Unified optimize plan v1

The `modelsurgeon optimize` command is the production-facing planning boundary
for structural optimization. It resolves the existing `Settings` contract,
selects a bounded preset plus hardware and quality profiles, and emits a
stable `optimize_plan` record. Planning is read-only: it does not load a model,
download data, overwrite a source checkpoint, or fabricate measured results.

## Contract

The Python API is `build_optimize_plan(settings, ...)` in
`modelsurgeon.optimization`. The CLI accepts YAML/TOML configuration plus
`--model`, `--revision`, `--preset`, `--hardware-profile`, and
`--quality-profile` overrides. Equivalent resolved settings use the same
canonical JSON and therefore the same `plan_id`.

The record retains:

- resolved settings and their digest;
- source model path and immutable revision status;
- selected profiles, exact bounded budgets, and conservative cost estimates;
- exact follow-on command arguments and explicit approval points;
- resume token, no-source-overwrite rollback policy, and artifact lineage; and
- supported, unsupported, failed, or unknown outcome plus uncertainties.

The built-in profiles are `cpu-small`, `cpu-large`, `gpu-12gb`, and `gpu-24gb`.
Presets are `fast`, `balanced`, and `quality`. A missing source path or
revision is retained as an `unknown` plan rather than silently treated as
executable. GGUF generic mutation is explicitly `unsupported`; callers should
use the native GGUF workflow.

## Safety and approvals

Every plan lists review, source identity, and resource-budget approval points.
An execution-shaped plan additionally requires artifact-write approval. The
current release has no executor, so `--execute` only emits the approval-gated
plan and still performs no mutation. Persisted plan artifacts use exclusive
create semantics unless the validated safety setting explicitly permits
overwrite.

## Examples

```text
modelsurgeon optimize --model models/tiny --revision abc123 --dry-run --json
modelsurgeon optimize config.toml --preset quality --hardware-profile gpu-12gb
```
