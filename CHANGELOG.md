# Changelog

## Unreleased

- Add a safe Hugging Face causal LM loader with CPU defaults, explicit dtype/device controls, and resolved-revision provenance.
- Add deterministic Hugging Face module, parameter, attention-head, KV-head, and MLP-channel discovery.

All notable changes will be documented here. This project follows Keep a Changelog and semantic versioning once public APIs stabilize.

## [Unreleased]

### Added

- Initial package, configuration, logging, component identity, architecture walking, Hugging Face loading, CLI, tests, and CI.
- v0.1–v1.0 architecture and finite GitHub roadmap.
- Immutable hierarchical configuration with model, calibration, feature, objective, hardware, and safety sections.
