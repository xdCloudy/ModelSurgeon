# Public API and compatibility policy

ModelSurgeon is experimental research software. The supported Python API is deliberately
small: import from the package-level namespaces below and use only names listed in each
namespace's `__all__`. The evaluation namespace includes the versioned benchmark
evidence and preregistered protocol contracts. Implementation submodules, private names, and objects not exported
by these namespaces are experimental and may change without a compatibility promise.

| Namespace | Stable contract |
| --- | --- |
| `modelsurgeon.adapters` | Framework-neutral sources, sessions, capability discovery, family detection, and fail-closed competitor execution records. |
| `modelsurgeon.graph` | Canonical component IDs, component graphs, validation, serialization, and remapping. |
| `modelsurgeon.datasets` | Calibration identities, validated mutation examples, leakage-safe splits, hardware cost examples, profile-partitioned manifests, and versioned mutation interaction evidence. |
| `modelsurgeon.features` | Versioned feature records, bounded primitive extractors, pre-mutation interaction feature contracts, and source-fitted architecture-normalized meta-feature schemas. |
| `modelsurgeon.surgery` | Transactional mutation requests, plans, outcomes, and physical-surgery entry points. |
| `modelsurgeon.evaluation` | Typed benchmark/evaluation reports, frozen protocol and baseline evidence, quantization-order attribution, physical compression/quality-loss Pareto evidence, profile-bound kernel/offload microbenchmarks, structured artifact reconciliation, compatibility, transfer-suite, and meta-surgeon matrix evidence, provenance-complete benchmark cells, and bounded llama.cpp validation. |
| `modelsurgeon.surgery` | Transactional mutation plans plus versioned physical artifact outcomes, cumulative lineage reconciliation, reload-bound HF sequences, bounded native GGUF sequences, unified repair outcome budgets/artifact lineage, and fail-closed hardware-specific alignment rules. |
| `modelsurgeon.surgery` | Shared format-neutral artifact integrity and rollback gates for HF/GGUF publication. |
| `modelsurgeon benchmark` | Deterministic plans, resumable runs, immutable imports, audits, reports, and physical HF/GGUF deployment evidence. |
| `modelsurgeon.experiments` | Experiment identity, persistence, artifacts, resource budgets, reproducibility records, and immutable empirical hardware/runtime profiles. |
| `modelsurgeon.surgeon` | Typed predictor bundles, training, calibration, ranking, versioned current-state embeddings, state-dependent predictor contracts, fixed-budget ranking-objective studies, bounded structural-model comparisons, explicit lineage/compatibility decisions, bounded target adaptation records, fail-closed transfer-confidence decisions, and signed pretrained registry cards. |
| `modelsurgeon.active_learning` | Deterministic acquisition, diversity, uncertainty, schedules, budgets, and state-bound interaction-aware replanning with rollback lineage. |
| `modelsurgeon.search` | Constraints, objectives, Pareto archives, policies, resumable search state, and versioned deployable architecture state/distance contracts. |
| `modelsurgeon.explain` | Decision summaries, attribution records, and deterministic reports. |

The CLI is the stable end-user orchestration boundary; use `modelsurgeon --help` for its
versioned command contracts. Direct Hugging Face, PyTorch, `llama.cpp`, GGUF codec, and
storage implementation imports are intentionally not stable public API.

## Compatibility rules

- Persisted records, protocol manifests, and artifacts carry an explicit schema version. Readers reject unknown
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
