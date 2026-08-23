# Changelog

## Unreleased

- Add a bounded, read-only memory-mapped GGUF v2/v3 container parser with typed
  metadata, exact tensor ranges, endian detection, and corruption checks.
- Add stable lazy GGUF tensor handles with tensor-scoped byte reads and complete-block
  chunk iteration under explicit allocation limits.
- Discover reconciled GGUF physical component graphs with explicit tensor-axis and
  architecture coupling constraints.
- Add deterministic dependency-free tiny transformer doubles and a revision-pinned,
  weights-free Hugging Face integration fixture manifest.
- Add a golden offline loader-to-discovery-to-graph-to-CLI integration test.
- Add a CPU-first hardware/software inventory with optional bounded CUDA, NVIDIA
  driver, GPU memory, RAM, and disk-capacity probing.
- Add versioned scalar/vector feature records with sample context and explicit direct
  quantized, locally dequantized, or high-precision provenance.
- Add versioned calibration dataset, preprocessing, tokenizer, sample identity,
  licensing/trust, and deterministic hash-ranked selection contracts.

- Add a safe Hugging Face causal LM loader with CPU defaults, explicit dtype/device controls, and resolved-revision provenance.
- Add deterministic Hugging Face module, parameter, attention-head, KV-head, and MLP-channel discovery.
- Add a versioned, framework-neutral component dependency, coupling, and mutation-constraint graph schema.
- Build deterministic transformer hierarchy, dataflow, residual, projection-coupling, and mutation-constraint graphs from discovery records.
- Validate graph endpoints, reciprocal edges, forbidden cycles, constraint membership, and complete mutation coupling closures with exact diagnostics.
- Persist and strictly reload canonical versioned component graphs with required adapter and immutable-model provenance.
- Pin the native GGUF container/codec specification, low-memory surgery decisions, and independent-reader conformance vector.
- Define exact GGUF block layouts, axis/divisibility planning, bounded codec operations, validation, error metrics, and no-substitution registry semantics.
- Define versioned GGUF family aliases, tensor/component axes, coupling groups, metadata updates, and block renaming contracts.
- Expand model inspection into stable human/JSON identity and canonical component output with categorized failures.

All notable changes will be documented here. This project follows Keep a Changelog and semantic versioning once public APIs stabilize.

## [Unreleased]

### Added

- Initial package, configuration, logging, component identity, architecture walking, Hugging Face loading, CLI, tests, and CI.
- v0.1–v1.0 architecture and finite GitHub roadmap.
- Immutable hierarchical configuration with model, calibration, feature, objective, hardware, and safety sections.
