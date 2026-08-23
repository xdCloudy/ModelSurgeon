# Changelog

## Unreleased

- Resolve model-wide native Llama/Qwen MHA, GQA, and MQA Q/K/V/O head-removal
  rules with fixed explicit head dimensions, safe KV grouping, exact codec
  alignment strategies, and fail-closed rejection of semantic remapping.
- Add matched no-surgery requantization controls that stream the same exact-codec
  block ranges as a structural plan and separately attribute requantization,
  surgery, and combined metric deltas.
- Execute coupled native quantized GGUF MLP channel removal with bounded encoded
  copies, one-row selective repacking, exact-codec requantization, resumable
  transactional output, untouched-tensor hashes, and output-graph validation.
- Plan coupled native Llama/Qwen GGUF MLP channel removal across gate/up/down
  axes with reconciled shapes, metadata, identities, parameters, and file size.
- Stream changed GGUF blocks through exact original or selected codecs with
  payload validation, bounded round-trip checks, and quantization error summaries.
- Selectively read and dequantize only GGUF repack block spans under simultaneous
  encoded, decoded, and peak-working-memory ceilings with touched-range reports.
- Validate GGUF mutation axes against exact codec blocks, distinguish direct copy,
  repack, and whole-slice strategies, and expose non-automatic aligned proposals.
- Compile closed mutation plans into allocation-free physical tensor shape/index
  transforms, metadata updates, identity mappings, and reconciled storage deltas.
- Add explicit retained, removed, renumbered, split, and merged component identity
  remaps that compose across surgeries without silent identity fallback.
- Serialize canonical mutation plans, revisions, outcomes, deltas, and explicit
  identity mappings with verified IDs and default local-path redaction.
- Resolve requested mutation targets into deterministic transitive component
  closures with constraint and coupled-edge reasons before any model change.
- Add deterministic, immutable transactional mutation request, compatibility,
  precondition, delta, plan, apply/rollback, and safe ownership contracts.
- Add a finite Gemma GGUF surgery compatibility contract for dense Gemma v1,
  with Gemma 2/3 failing closed pending their extra normalization and attention rules.
- Add native and legacy-prefix Mistral GGUF surgery mappings with required
  sliding-window metadata, strict GQA geometry, and complete coupled edit axes.
- Add explicit dense Qwen2/Qwen3 GGUF surgery mappings and GQA constraints, with
  recognized MoE variants failing closed pending expert/router support.
- Add a versioned Llama GGUF surgery adapter with strict physical tensor maps,
  coupled edit axes, MHA/GQA geometry, and fail-closed compatibility validation.
- Add prioritized native IQ4_NL and IQ4_XS codecs with nonlinear codebook
  packing, endian-aware scales, exact-type dispatch, and fail-closed unsupported
  IQ writes.
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
- Add bounded streaming Hugging Face calibration ingestion, tokenization, and
  canonical atomic manifest caching.
- Add canonical, duplicate-safe activation hook ownership with rollback and guaranteed
  capture cleanup across success, exception, and interrupt paths.
- Add mergeable fixed-memory streaming moments, extrema, RMS, sparsity, activation,
  histogram, and percentile accumulators.
- Add masked per-component activation summaries with sample and aggregation-axis
  provenance.
- Add graph-aligned per-channel and finite token position/class activation features
  with memory bounded by channels and configured buckets.
- Add hard-ceiling-aware full, tensor, and streaming memory-mode planning with
  auditable peak RAM, VRAM, and scratch estimates.
- Add conservative GGUF output/scratch disk preflight and remaining-space monitoring
  with explicit alignment and safety-margin accounting.
- Add transactional GGUF v2/v3 output planning and bounded chunk streaming with
  staged validation, fsync, atomic publication, and SHA-256 integrity provenance.
- Add byte-for-byte unchanged GGUF tensor copying with complete-block chunk bounds and
  per-tensor output checksums.
- Add checksummed, resumable disk-backed tensor intermediates with fixed chunk memory,
  atomic manifests, stale-artifact recovery, and scoped cleanup.
- Add tensor-boundary resumable GGUF output with input/plan identity checks, committed
  prefix verification, partial-range truncation, and no incomplete publication.
- Add pinned license-compatible encoded/decoded conformance vectors for every native
  GGUF codec type with byte-order, field-packing, shape, and checksum validation.
- Add bounded F32, F16, and BF16 GGUF codecs with odd-count streaming, exact endian
  behavior, and round-to-nearest-even BF16 encoding.
- Add a bit-exact block-aware Q8_0 codec with bounded range access, endian preservation,
  validation, and quantization error reports.
- Add a dedicated Q6_K super-block codec with pinned 6-bit packing, subgroup scales,
  endian-aware encoding, range access, and validation.
- Add the distinct Q5_K super-block codec and explicit Q5_K_S/Q5_K_M whole-file recipe
  metadata handling without tensor-layout substitution.
- Add the distinct Q4_K super-block codec and explicit Q4_K_S/Q4_K_M recipe metadata
  without treating whole-file recipes as tensor types.
- Add separate Q2_K and Q3_K super-block codecs with non-interchangeable validation,
  packed scale/high-bit handling, endian-aware deltas, and bounded range access.
- Bound v1 IQ-family native writes to prioritized IQ4 targets with explicit read-only
  IQ2 and deferred IQ1/IQ3 decisions plus pinned codebook provenance.

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
