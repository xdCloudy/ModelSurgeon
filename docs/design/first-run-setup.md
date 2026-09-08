# Local-first setup and first-run contract

Status: implemented in issue #486; installer packaging remains unclaimed.

ModelSurgeon setup is a local, non-downloading operation. The supported path
creates a user-owned data root, checks the core Python runtime, records an
existing local fixture manifest when one is supplied, and reports provider and
resource limitations before a model is loaded. It does not install Python
packages, download model weights, contact a hosted provider, or claim a
platform installer.

## The three model roles

The first-run layout keeps these roles separate:

| Location | Role | Ownership |
| --- | --- | --- |
| `Surgeon Tensors/` | Learned Meta-Surgeon assets used by ModelSurgeon | ModelSurgeon |
| `Text LLM/` | Optional conversational control-plane model | User/provider selection |
| `Models/` | Target models inspected or edited | User |

Text-model output is not surgery authority. The deterministic CLI/Python
planning, measurement, validation, and artifact paths remain usable with the
explicit `provider.kind=none` default.

## Commands

Diagnostics are read-only and never create directories:

```bash
uv run modelsurgeon setup diagnostics --data-dir ./work/modelsurgeon-data --json
```

Initialize a local/offline layout against the checked-in supported fixture
manifest:

```bash
uv run modelsurgeon setup init \
  --data-dir ./work/modelsurgeon-data \
  --fixture tests/fixtures/tiny_hf_models_v1.json \
  --offline \
  --min-free-gb 1 \
  --json
```

The manifest is metadata used for an offline setup check. It contains no
weights and does not establish that every model family, runtime, or device is
supported. A real target model still needs its own immutable local path or
revision and the normal inspection/preflight gates.

The default data root is `%LOCALAPPDATA%\ModelSurgeon` on Windows and
`$XDG_DATA_HOME/ModelSurgeon` (or `~/.local/share/ModelSurgeon`) on Linux and
macOS. Use `--data-dir` when a secondary SSD or another user-owned location is
preferred. Protected installation locations and a `Setup.exe` are future
packaging options, not current support claims.

## Diagnostic outcomes

Every check has one of these outcomes: `supported`, `unavailable`,
`unsupported`, `failed`, or `unknown`. Examples include:

- `missing_data_directory`: diagnostics found a path that initialization can
  create; `setup init` is the next action.
- `permission_denied`: the selected root cannot accept a temporary write.
- `disk_budget_exceeded`: free space is below the explicitly declared
  `--min-free-gb` budget. The budget is for setup metadata and fixture
  preparation; setup never downloads a large model to consume it.
- `offline_fixture_required` or `fixture_missing`: offline mode has no usable
  local supported-fixture manifest.
- `no_llm`: no conversational provider is selected; direct CLI/Python remains
  available.
- `adapter_unavailable` or `missing_api_key`: a selected optional provider is
  not ready. Setup does not guess a substitute.
- `unsupported_platform` or `unsupported_python`: the core support matrix does
  not cover the current platform/runtime.

An overall `ready` report requires every check to be `supported`. A
`needs_attention` report preserves the exact check outcomes so missing,
unsupported, failed, and unknown cells are not collapsed into a false pass.

## Generated layout

Initialization creates these directories and a short root README:

```text
<data-root>/
├── Surgeon Tensors/       # learned Meta-Surgeon assets
├── Text LLM/              # optional conversational model only
├── Models/                # user/target models only
├── Campaigns/active/
├── Campaigns/completed/
├── Evidence/
├── Benchmarks/
├── Artifacts/
├── Cache/
├── Logs/
└── Config/modelsurgeon.toml
```

`Config/modelsurgeon.toml` is a secret-free first-run manifest. It is not a
replacement for the existing optimization settings file accepted by
`--config`; the two contracts are intentionally separate. Provider credentials
are referenced only by environment-variable name in the existing provider
configuration and are never written by setup.

