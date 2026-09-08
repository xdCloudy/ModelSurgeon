# ModelSurgeon product layout

Status: current local-first layout contract; installer packaging is not claimed.

ModelSurgeon has two distinct surfaces: a protected Python/CLI installation
environment and a user-owned mutable data root. The latter is intentionally
selectable because target models, caches, learned assets, and evidence can be
large. The current implementation creates only the user-owned data root; it
does not ship `Setup.exe`, install into `Program Files`, or manage a system
package manager.

## Mutable data root

```text
<data-root>/
├── Surgeon Tensors/       # ModelSurgeon's learned Meta-Surgeon assets
├── Text LLM/              # optional conversational control-plane model
├── Models/                # user/target models being inspected or edited
├── Campaigns/
│   ├── active/
│   └── completed/
├── Evidence/
├── Benchmarks/
├── Artifacts/
├── Cache/
├── Logs/
├── Config/
│   └── modelsurgeon.toml
└── README.md
```

The three model directories are not interchangeable:

- `Surgeon Tensors/` is for ModelSurgeon's learned assets. It is not a target
  model directory and it is not a text-model cache.
- `Text LLM/` is optional conversational control-plane input. It may be empty;
  no-LLM mode is a supported direct CLI/Python path.
- `Models/` is for user-selected target models. Target model immutability,
  revision identity, and output staging remain governed by the existing
  inspection and surgery contracts.

## Location and permissions

On Windows the default is `%LOCALAPPDATA%\ModelSurgeon`. On Linux and macOS
the default is `$XDG_DATA_HOME/ModelSurgeon`, falling back to
`~/.local/share/ModelSurgeon`. `--data-dir` selects another user-owned path,
such as a secondary SSD. Setup creates missing directories only in `setup init`
and probes write access with a temporary file. It reports a stable
`permission_denied` outcome instead of continuing with a guessed path.

The `--min-free-gb` value is a declared preflight budget for setup metadata and
fixture preparation. It is checked against the filesystem containing the data
root and is included in the JSON report. It is not a promise that a model fits,
and setup never downloads one to test the budget.

## First-run sequence

1. Run `modelsurgeon setup diagnostics` with the intended data root.
2. Resolve `missing_data_directory`, `permission_denied`, or
   `disk_budget_exceeded` before continuing.
3. In offline mode, provide an existing local supported-fixture manifest and
   accept that the manifest contains metadata rather than model weights.
4. Run `modelsurgeon setup init` to create the layout and secret-free manifest.
5. Select a text model only if conversational control-plane use is wanted;
   otherwise use the explicit no-LLM direct CLI/Python path.
6. Supply a separate immutable target model path/revision for inspection or
   optimization.

Every check is retained as `supported`, `unavailable`, `unsupported`,
`failed`, or `unknown`. A provider that is absent, uncredentialed, or outside
the current adapter boundary is reported as such; it is never silently
replaced by another provider.

