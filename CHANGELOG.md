# Changelog

## Unreleased

- Add native quantized GGUF transformer-layer removal by omitting complete block tensors,
  canonically renaming following blocks, updating block-count metadata, and checksum-verifying
  byte-identical bounded direct copies of every retained encoded payload.
- Add non-overwriting atomic safetensors checkpoint publication with deterministic single or
  sharded layouts, index/config staging, source-shard integrity checks, bounded tensor/shard
  sizes, and streamed payload checksum verification before visibility.
- Add bounded selected-Linear SVD replacement for Hugging Face models using real two-factor
  modules, with requested/effective rank, reconstruction error, and measured parameter and
  per-token FLOP reconciliation.
- Add physical Hugging Face transformer-layer removal with retained-weight identity,
  canonical and KV-cache execution-index renumbering, exact residual-bypass equivalence,
  parameter reconciliation, and successful real-model save/reload.
- Add model-wide physical Hugging Face MHA/GQA head removal with complete-KV-group safety,
  synchronized Q/K/V/O block resizing, metadata/parameter reconciliation, bit-exact
  grouped-mask evidence, and successful real-model save/reload.
- Add model-wide physical Hugging Face gated-MLP channel removal with synchronized gate/up
  rows and down columns, global configuration/linear metadata updates, exact parameter
  reconciliation, mask-equivalence evidence, and successful real-model save/reload.
- Add equal-budget active/random/utility-only learning-curve studies with normalized AULC,
  seeded bootstrap confidence intervals, dependency-free SVG plots, and explicit negative
  results when active selection does not beat the strongest baseline.
- Add auditable surgeon retraining triggers for new-example count, elapsed budget, and drift,
  plus all-criteria challenger promotion that always retains the incumbent after training
  failure or missing/insufficient validation evidence.
- Add durable active-learning evaluation schedules that bind acquisition metadata to
  resulting experiment and dataset-example IDs and resume partial batches without
  rerunning or changing selection.
- Add candidate-boundary active-learning budgets for attempt count, wall time, tier cost,
  GPU time, and disk use, with explicit observed-versus-reserved failed-attempt charging.
- Add deterministic explore/exploit acquisition with exact high-value, uncertainty, and
  diversity fractions, per-selection reasons and conditional propensities, overlap fill,
  and explicit zero/oversubscribed budget behavior.
- Add seeded farthest-first diversity selection across normalized numeric, categorical,
  and topology spaces with observable O(candidate-count) working memory and explicit
  100,000-candidate/4,096-selection ceilings.
- Add versioned mutation-equivalence keys with order-insensitive component closures,
  namespaced adapter-declared equivalence, deterministic pool deduplication, and exclusion
  of already completed or in-flight work.
- Add bounded candidate-pool scoring for utility, named outcomes, calibrated safe
  probability, and uncertainty, with batch-size-stable ordering and explicit quarantine
  records for incompatible feature schemas.
- Add canonical graph-valid active-learning pools capped at 100,000 candidates, with cheap
  mutation-free features, complete revision provenance, bounded append invocations, and
  digest-verified exact-record resume.
- Add optional fixed-budget MLP uncertainty comparison for deep ensembles and Monte Carlo
  dropout, reporting interval calibration and active-selection lift with deterministic,
  schema-versioned stochastic predictions and backward-compatible dropout configuration.
- Add a fixed-budget tree-surgeon uncertainty comparison across ensemble, bootstrap, and
  quantile intervals, reporting coverage, error-ranking utility, CPU/model-memory cost,
  and schema-versioned uncertainty values for downstream acquisition.
- Add validation-only Platt and isotonic safe-mutation probability calibration with
  deterministic selection, versioned serialization, and stored Brier, ECE, and complete
  reliability-curve evidence for every candidate method.
- Make large Hugging Face proof campaigns scale by reusing one validated mutation-target
  graph index and caching vectorized per-layer weight statistics instead of rescanning the
  component graph and synchronizing individual GPU scalars for every candidate.
- Make durable artifact and checkpoint publication work on Windows, publish the
  package's PEP 561 typing marker, and keep platform-specific memory and llama.cpp
  validation paths strict-type-checkable.
- Add turnkey `first-surgeon-evidence` LightGBM proof reporting with grouped held-out
  bootstrap metrics, identical-candidate random/magnitude comparisons, immutable-bundle
  inference smoke tests, source revisions, artifact hashes, and bounded training telemetry.
- Add production Hugging Face MLP-channel proof execution with exact intermediate-channel
  masking, activation/static feature capture, causal-LM perplexity measurement, and a real
  local-Transformers CI smoke path.
- Preserve grouped split identities and resolved model/schema versions across `train-surgeon`
  and `predict-surgeon`, and make LightGBM 4.x consume validated NumPy matrices with stable
  backend feature names while retaining the semantic preprocessing schema.
- Add explicit device-capability mixed-precision decisions for FP32/FP16/BF16/autocast,
  with recorded compute/accumulation dtypes, fail-closed or explicit FP32 fallback,
  and metric precision binding that rejects silent dtype/autocast drift.
- Add manifest-bound adaptive calibration batching with exact sample-boundary resume,
  whole-sample token/memory ceilings, measured RAM/VRAM model updates, and explicit
  memory-exhaustion backoff without sample reordering or skipping.
- Add append-only per-stage runtime telemetry with partial interrupt retention, wall/CPU
  timing, token/candidate throughput, peak RAM/VRAM, process I/O bytes, stable hardware
  normalization contexts, and experiment database schema v5 persistence.
- Add bounded MLP duplicate-channel ranking with weight-cosine candidate screening,
  activation-correlation confirmation, explicit candidate/confirmation budgets, and
  deterministic ranking.
- Add candidate-restricted tensor/output cosine similarities with configurable adjacent,
  explicit, or bounded all-pairs generation, blockwise evaluation, and an explicit
  zero-vector policy that returns zero while recording the degeneracy.
- Add per-component gradient norm, weight×gradient, and first-order removal
  sensitivity features with explicit mean-over-batches semantics and typed missing-gradient outcomes.
- Add opt-in bounded selected-parameter gradient collection with detached CPU snapshots,
  explicit missing-gradient reporting, per-gradient size ceilings, and guaranteed
  gradient cleanup before and after every calibration backward pass.
- Add fixed-memory activation covariance collection with exact diagonal Welford
  statistics, deterministic rank-bounded Nyström sketches, workspace preflight,
  and small-case approximation-accuracy reporting.
- Add seeded randomized spectral extraction with configurable rank/oversampling,
  low-rank reconstruction-error estimates, power iteration, and hard CPU workspace
  preflight before tensor snapshot allocation.
- Add exact bounded singular-value extraction with spectral/effective/stable rank,
  normalized decay, energy-rank thresholds, explicit size declines, and recorded
  float64 convergence tolerance/provenance.
- Add configurable linear percentiles, fixed-width histograms, skewness, and excess
  kurtosis for weight tensors with recorded bin/interpolation definitions and finite
  constant-tensor behavior.
- Add deterministic per-tensor weight statistics with detached CPU snapshots,
  float64 accumulation, shape/dtype/device provenance, and explicit empty/non-finite rejection.
- Validate generated GGUFs with the pinned llama.cpp revision using a bounded
  one-token forward/generation check, captured tool/command provenance, capped logs,
  and fail-closed timeout or non-zero-exit reporting.
- Execute model-wide native quantized GGUF attention-head removal with direct
  Q/K/V/O encoded copies, bounded one-row O repacking, explicit fixed head-length
  metadata, resumable output, error ceilings, and output-graph validation.
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
