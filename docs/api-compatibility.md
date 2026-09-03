# Public API and compatibility policy

ModelSurgeon is experimental research software. The supported Python API is deliberately
small: import from the package-level namespaces below and use only names listed in each
namespace's `__all__`. Implementation submodules, private names, and objects not exported
by these namespaces are experimental and may change without a compatibility promise.

| Namespace | Stable contract |
| --- | --- |
| `modelsurgeon.adapters` | Framework-neutral sources, sessions, capability discovery, and family detection. |
| `modelsurgeon.graph` | Canonical component IDs, component graphs, validation, serialization, and remapping. |
| `modelsurgeon.datasets` | Calibration identities, validated mutation examples, leakage-safe splits, and partition manifests. |
| `modelsurgeon.features` | Versioned feature records and bounded feature extraction interfaces. |
| `modelsurgeon.surgery` | Transactional mutation requests, plans, outcomes, and physical-surgery entry points. |
| `modelsurgeon.evaluation` | Typed evaluation reports, compatibility evidence, and bounded llama.cpp validation. |
| `modelsurgeon.experiments` | Experiment identity, persistence, artifacts, resource budgets, and reproducibility records. |
| `modelsurgeon.surgeon` | Typed predictor bundles, training, calibration, and ranking contracts. |
| `modelsurgeon.search` | Constraints, objectives, Pareto archives, policies, and resumable search state. |
| `modelsurgeon.explain` | Decision summaries, attribution records, and deterministic reports. |

The CLI is the stable end-user orchestration boundary; use `modelsurgeon --help` for its
versioned command contracts. Direct Hugging Face, PyTorch, `llama.cpp`, GGUF codec, and
storage implementation imports are intentionally not stable public API.

## Compatibility rules

- Persisted records and artifacts carry an explicit schema version. Readers reject unknown
  or incompatible versions rather than guessing a meaning.
- Additive Python fields or record properties may be introduced in a compatible release.
  Removing, renaming, or changing the meaning of an exported object or serialized field
  requires a new schema version and a migration/compatibility test.
- New public objects must be exported from one tabled namespace, have type annotations and
  user-facing documentation, and receive focused contract coverage.
- Experimental capabilities must remain explicitly marked as experimental or unsupported in
  the architecture compatibility matrix; structural tests alone do not imply a runtime claim.
- Commands and record readers fail explicitly for unavailable optional dependencies,
  unsupported model formats, and incompatible schemas.

The package is currently pre-1.0. These rules prevent silent contract drift while the public
surface is stabilized for the v1.0 release; they do not promise broad semver compatibility
for implementation-only imports.
