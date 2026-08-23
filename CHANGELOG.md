# Changelog

## Unreleased

- Add a safe Hugging Face causal LM loader with CPU defaults, explicit dtype/device controls, and resolved-revision provenance.
- Add deterministic Hugging Face module, parameter, attention-head, KV-head, and MLP-channel discovery.
- Add a versioned, framework-neutral component dependency, coupling, and mutation-constraint graph schema.
- Build deterministic transformer hierarchy, dataflow, residual, projection-coupling, and mutation-constraint graphs from discovery records.
- Validate graph endpoints, reciprocal edges, forbidden cycles, constraint membership, and complete mutation coupling closures with exact diagnostics.
- Persist and strictly reload canonical versioned component graphs with required adapter and immutable-model provenance.
- Pin the native GGUF container/codec specification, low-memory surgery decisions, and independent-reader conformance vector.

All notable changes will be documented here. This project follows Keep a Changelog and semantic versioning once public APIs stabilize.

## [Unreleased]

### Added

- Initial package, configuration, logging, component identity, architecture walking, Hugging Face loading, CLI, tests, and CI.
- v0.1–v1.0 architecture and finite GitHub roadmap.
- Immutable hierarchical configuration with model, calibration, feature, objective, hardware, and safety sections.
