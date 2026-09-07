# Changelog

## Unreleased

- Add the v2.7 canonical conversational campaign state store. Campaign/session
  linkage, exact spec and hard constraints, policy and approval state, provider
  context, resource budgets, lifecycle, deterministic versioned transitions,
  WAL-backed restart/reconnect recovery, and append-only negative evidence are
  authoritative and fail closed. Transcript text, summaries, provider memory,
  secrets, and untrusted tool payloads are not persisted as campaign state.

- Add the experimental interpreted `OptimizationSpec` preview boundary for
  conversational requests. It renders the exact canonical objective contract,
  hard constraints, preferences, budgets, allowed operations, unresolved
  fields, diagnostics, provenance, and approval requirement; unsupported or
  unresolved fields remain blocked. Material edits produce a new spec identity
  and deterministic diff, while explicit confirmation releases only the exact
  digest-bound spec submission. No mutation, optimizer execution, or live
 provider/model evidence is claimed.

- Freeze the v2.6 bounded conversational tool-calling release boundary with a
  machine-readable dependency/evidence record, a fail-closed release audit,
  explicit four-tool allowlist and outcome vocabulary, direct-API compatibility
  checks, and documented live-evidence/process-isolation limitations. Tighten
 dispatcher lifecycle ordering so final result-envelope validation completes
 before an automatically managed read-only transaction is committed.

- Add a versioned chat inspection context built from direct GGUF discovery,
  CPU-first hardware profile, memory inventory, and architecture capability
  APIs. Pass bounded engine-owned context to the intent compiler, distinguish
  detected inventory from provider declarations, and retain explicit
  unsupported, failed, and unknown results without claiming live evidence.

- Add the v2.6 adversarial conversational tool-boundary corpus for shell/code,
  traversal, secret-exfiltration, prompt-injection, malformed-schema, forged
  measurement, contradictory-provenance, and replay attacks. Tighten schema
  range/regex validation, redact secret-shaped retained diagnostics, bind result
  provenance to the request tool identity, and retain explicit unverified,
  unsupported, failed, unknown, timeout, and cancelled outcomes. This is
  bounded contract evidence only; no live provider, campaign, model, or shell
  execution is claimed.

- Add bounded provider/tool/secret isolation for the v2.9 conversational
  security boundary: indirect credentials, recursive diagnostic and payload
  redaction, copied untrusted requests, provider capability-drift detection,
  untrusted result trust-zone markers, and observable fail-closed
  `isolation_failure` outcomes. The implementation is capability-isolated but
  does not claim hostile-process containment or live hosted-provider evidence.

- Add the experimental `modelsurgeon chat <provider-model-path>` entry slice.
  It bootstraps a bounded local GGUF provider, validates model identity and
  runtime provenance, emits deterministic session/turn records, and routes
  interpretation through the existing intent compiler and policy evaluator.
  The command is interpretation-only: no optimizer, tensor selection,
  approval, mutation, artifact write, hosted-provider claim, or live evidence
  is added. Unsupported, failed, unknown, malformed, timeout, and cancelled
  outcomes remain explicit.

- Add the bounded v2.0 autonomous optimizer release boundary: a versioned
  compatibility/evidence manifest, capability classifications, scientific
  limitations report, deterministic reproduction contract, and fail-closed
  release audit. This records protocol and control-plane evidence only; no
  live benchmark, signed package, or reference model artifact is claimed.

- Add a fail-closed conversational transaction boundary that separates
  read-only tool handles from consequential execution, requires explicit
  commit before publishing consequential success, rolls back active work on
  cancellation/timeout/failure, rejects undeclared consequential retries, and
  retains deterministic transaction IDs alongside replay receipts. No live
  campaign or model evidence is claimed by this control-plane contract.

- Add the v2.0 autonomous benchmark release-candidate contract with exact
  protocol identities, bounded matrices, retained negative/unsupported cells,
  confidence-bounded competitiveness claims, complete reference-artifact
  mutation/repair/quantization/runtime/license/rollback lineage, and a
  fail-closed independent-audit/replay publication decision. No live benchmark
  evidence is claimed or bundled.

- Freeze the v2.1 conversational intent contract around the existing
  `OptimizationSpec`/`ObjectiveContract` schema, with a machine-readable
  compatibility record, explicit current-versus-planned availability, and a
  replay test that validates schema versions, deterministic corpus evidence,
  provenance, and fail-closed non-executable outcomes.

- Add an explicit, secret-free provider configuration boundary with deterministic
  precedence, no-LLM defaults and CLI overrides. Direct optimization remains
  usable without a text-model installation; provider diagnostics report stable
  no-LLM, missing-key, and unavailable-adapter outcomes without silent fallback.

- Add a bounded offline local GGUF text-model provider with explicit architecture
  and runtime validation, hard context/memory/time/response budgets, structured
  output validation, cancellation, deterministic provider provenance, and
  retained unsupported/failed/timeout/cancelled outcomes.

- Add version-1 capability-scoped conversational tool schemas with strict
  allowlisting, deterministic request/tool IDs, bounded budgets, typed
  negotiation/refusal results, approval-gated consequential calls, and no
  direct tensor-removal or arbitrary command authority. Add the trusted
  dispatcher boundary with request validation, replay protection, bounded
  retries, cancellation, budget enforcement, and engine-owned provenance.

- Add version-2 grounded tool-result envelopes with deterministic result IDs,
  engine-supplied canonical/unverified/unavailable provenance, source,
  evidence, artifact and campaign lineage, stale-request checks, UTC
  observation timestamps, and retained bounded raw negative payloads.

- Add a bounded autonomous `optimize --execute` orchestrator with a deterministic
  stage DAG/cursor, atomic resume state, recorded approvals and overrides,
  trusted runtime adapters, retained alternatives, and fail-closed measured-only
  promotion. Missing or infeasible final evidence now produces an explicit
  negative outcome without an accepted artifact.

- Bind optimize approvals to canonical plan digests and material plan diffs,
  retain expiry and secret-free operator context, replay identical decision
  evidence deterministically, and emit signed offline-verifiable final
  reproducibility packages with explicit external-artifact and non-determinism
  claim limits.

- Add fail-closed hosted and compatible endpoint adapters with secret-free
  endpoint configuration, external credential references, capability probing,
  structured-output negotiation, bounded retries/rate-limit handling, model
  identity checks, redacted failures, and strict typed-response validation.

- Add a bounded, versioned intent compiler equivalence/refusal corpus with
  exact canonical-spec, ambiguity, diagnostic, provenance, resource-budget,
  and retained-negative-result checks.

- Add a fail-closed conversational intent policy evaluator with explicit
  confidence categories, ambiguity/refusal diagnostics, hard-constraint and
  preference conflict detection, deterministic safety precedence, and
  provenance-linked decisions that never emit a spec unless policy is
  executable.

- Add versioned conversational intent records that preserve the original
  request, source spans, normalized units, ambiguity and interpretation links,
  provider/tool provenance, and an exact emitted-spec digest or fail-closed
  non-executable outcome with canonical JSON round trips.

- Add a measured physical-strategy decision contract that requires no-repair and
  quantization-only controls, immutable artifact identity, complete provenance,
  and deployment/optimization/artifact budgets before selecting a repair result;
  incompatible, unknown, and failed outcomes remain explicit.

- Add an approved campaign coordinator and fail-closed promotion gate around
  lease-aware execution, requiring complete measured evidence, passed hard
  constraints, committed transactions, immutable child artifacts, and stable
  source identity before promotion; rejected/failed candidates replan
  deterministically while unknown candidates remain visible.

- Add deterministic strategy selection for required strong baselines, surgeon,
  acquisition, search policy, budgets, and approval-visible alternatives, with
  a rule-based fallback until held-out meta-evidence is validated.

- Add automatic capability and candidate-space planning with compatibility
  matrix checks, plugin/runtime refusals, deterministic model/hardware/objective
  identity, explicit exclusions, and dry-run candidate/evaluation/artifact
  bounds before mutation.

- Add a stable Python API quickstart, executable notebook, profile reference,
  consumer troubleshooting decision tree, and explicit HF/GGUF support
  boundaries for fresh-environment workflows.

- Add a typed resource-aware worker scheduler for local CPU/GPU and explicitly
  registered remote workers, with authenticated capability profiles,
  content-addressed task inputs/results, deterministic placement, campaign
  quotas, heartbeat recovery, and idempotent publication.

- Add a versioned objective contract that separates fail-closed hard
  constraints from weighted, lexicographic, or Pareto soft objectives,
  retains uncertainty and evidence, and gates custom objective plugins on
  capability cards, trust modes, approvals, and provenance.

- Add bounded target-runtime export capability contracts for Transformers,
  llama.cpp, MLX, ONNX, and vLLM with family/state matrices, converter/runtime
  provenance, explicit verified/experimental/unsupported/failed/unknown
  outcomes, and fail-closed asymmetric/low-rank/mixed-codec handling.

- Add deterministic offline benchmark and lineage explorers with canonical
  evidence projections, Pareto-front summaries, visible negative/unsupported/
  failed cells, immutable source links, safe escaping, static filtering, and
  bounded generation.

- Add fail-closed capability-negotiated plugin contracts for evaluators,
  runtimes, transformations, search policies, objectives, and registry
  providers, including metadata-only discovery, explicit allowlists, trusted
  in-process gates, bounded subprocess execution, and malformed/timeout/crash
  outcomes.

- Add crash-consistent optimize campaign progress with ordered versioned
  events, idempotent stage completion, pause/cancel/resume recovery, bounded
  ETA uncertainty, source/accepted-artifact preservation, and redacted
  diagnostic bundles.

- Add an offline local artifact registry for immutable model, surgeon, and
  evidence objects with stable catalog JSON, aliases, tags, leases, reference-
  aware garbage collection, signed bundle import/export, verification, and
  CLI/API parity.

- Add the unified read-only optimize planning contract with deterministic
  presets, hardware/quality profiles, dry-run JSON, bounded cost estimates,
  explicit approvals, source/artifact lineage, and unsupported/failed/unknown
  outcomes.

- Add read-only public benchmark auditing and falsification records for
  protocol drift, claim coverage, confidence evidence, contamination,
  reproduction, metric cherry-picking, aggregation sensitivity, and negative
  result retention.

- Add deterministic replay environment contracts for native, Docker, and
  exploratory Nix locks, mismatch preflight, resumable cursors, ordered schema
  migration copies, and tolerance-aware metric comparison.

- Add offline-verifiable signed evidence bundles with streaming Merkle checks,
  detached attestations, key rotation/revocation metadata, explicit redaction
  and external-reference claim limits, and tamper/missing-member detection.

- Add the complete consumer repair effectiveness/economics study matrix with
  physical-artifact evidence, hierarchical intervals, efficiency metrics, and
  conservative beneficial/harmful/unnecessary/infeasible recommendations.

- Add bounded joint surgery, repair, and quantization search contracts with
  hard resource budgets, compatibility gates, immutable lineage, rollback and
  terminal outcomes, conservative acquisition, and measured-only promotion.

- Add the bounded long interaction-aware sequence study contract across two
  families, stateless/additive/state-aware policies, horizons 10/20/50, and
  three seeds, retaining checkpoint/artifact lineage, regret/violation/cost
  evidence, adversarial interaction classes, rollback outcomes, and explicit
  unsupported or negative claims.
- Add the closed multi-axis architecture Pareto study contract with complete
  family/hardware/method/budget/seed matrices, conservative deployable
  frontiers, quality-cost hypervolume, paired deterministic bootstrap
  comparisons, and retained negative, unsupported, failed, and unknown cells.
- Add capability-gated heterogeneous architecture states with per-layer widths,
  low-rank factors, sparsity, and mixed codecs, analytic cost reconciliation,
  explicit unsupported/failed/unknown outcomes, and staging-based
  materialize/reload/benchmark evidence that preserves source artifacts.
- Add bounded calibrated RBF-compatible surrogate acquisition for complete
  architecture states, with mixed-space features, held-out objective and
  feasibility calibration, expected/hypervolume improvement batches, hard
  constraint gates, fit/evaluation budgets, explicit fallback outcomes, and
  persisted model/fit/resume records.
- Add bounded Pareto-beam and constrained evolutionary architecture policies
  over complete legal candidate states, with uncertainty-conservative dominance,
  deterministic seeded mutation/crossover, elitism and diversity, explicit
  retained evidence outcomes, and serializable resume state.
- Add a resumable architecture-search lifecycle with explicit predicted,
  reserved, materialized, evaluated, accepted, rejected, and failed stages,
  atomic digest-checked snapshots, measured-only frontier promotion, and
  architecture/equivalent-path duplicate charging.
- Add bounded multi-axis architecture candidate spaces with canonical state and
  placement identities, lazy seed-ranked retention, static hardware/legality
  pruning, deterministic page resumes, and explicit rejection counts.
- Add a bounded multi-machine hardware-selection study contract with three-profile
  coverage, repeated host/run evidence, deterministic hardware-aware versus blind
  comparisons, provenance capture, and explicit negative or inconclusive claims.
- Add hardware-conditioned search candidates with placement-aware identity,
  uncertainty-conservative hard constraints, measured-evidence promotion gates,
  deployment-objective ranking, and hardware-preserving resume/report/lineage records.
- Add versioned hardware-conditioned quality, safety, and utility predictors with
  explicit CPU/GPU/offload/quantization context, pre-mutation-only features,
  unsupported profile gates, and held-out-profile null-result ablations.
- Add a versioned crash-consistency fault matrix for publication and subprocess
  boundaries with owned-staging recovery invariants, bounded child-tree termination,
  clipped logs, and explicit unknown/unsupported evidence.
- Add a deterministic 2,048-case mutation property-suite contract covering writable
  native GGUF codecs and operations, seeded failure injection, deterministic shrinking,
  source/rollback/reconciliation evidence, and retained unsupported or unknown outcomes.
- Add a bounded corpus contamination audit for exact and near-duplicate text,
  generated outputs, declared model ancestry, licenses, unavailable sources,
  remediation protocol identities, and retained unknown limitations.
- Add a versioned cross-platform stability contract with metric tolerance provenance,
  exact environment and artifact identities, within-tolerance/expected-variation/drift
  classifications, unsupported and unknown cell retention, and additive report migration.
- Add a deterministic, license-safe hostile-input corpus for HF, safetensors,
  GGUF, config, manifest, path, and subprocess boundaries with content-addressed
  manifests, bounded runners, timeout/resource/path-escape outcomes, and retained
  zero-finding campaign evidence.
- Add a matched repair/recoverability dataset contract with exact no-repair controls,
  budget/method metadata, held-out evidence, immutable artifact lineage, complete cost and
  provenance records, group-disjoint splits, and retained failed, unsupported, and unknown cells.
- Add bounded immutable teacher-target chunks with model/tokenizer/example compatibility keys,
  checksum validation, resumable publication, cache hit/miss byte telemetry, and deterministic
  random/domain-balanced/diversity/recoverability selection that excludes held-out examples.
- Add conditional repair-recoverability predictors for held-out gain and success probability,
  calibrated intervals, action/budget compatibility, model/state/candidate leakage checks,
  conservative no-repair selection, and retained negative, failed, unsupported, and unknown cells.
- Add action- and hardware-conditioned repair cost predictors for tokens, time, memory, artifacts,
  cache bytes, and optional energy, with calibrated upper bounds, grouped leakage checks, and
  conservative hard-budget preflight that does not treat missing energy sensors as zero.
- Add a unified bounded repair outcome schema for no-repair, LoRA,
  selected/full fine-tuning, and logit/feature distillation with explicit
  budgets, held-out improvement evidence, teacher/data/tokenizer lineage,
  artifact-byte reconciliation, immutable children, and exact rollback/search
  parent rules for rejected or failed repairs.
- Add an offline signed pretrained meta-surgeon registry with complete model
  cards, source evidence and compatibility metadata, HMAC content identities,
  pre-deserialization schema guards, local resolve/list/verify APIs, and
  immutable adapted-child lineage.
- Add fail-closed transfer-confidence decisions with compatibility and
  capability preflight, deterministic abstention reasons, support/distance/
  uncertainty/calibration thresholds, and selective-risk coverage/cost curves
  that retain unsupported, unknown, and false-confidence outcomes.
- Add bounded zero-shot and few-shot target adaptation records for budgets
  0/8/16/32/64/128 with five seeded support selectors, explicit compatibility
  gates, target-test leakage protection, immutable parent/child lineage,
  learning curves, cost evidence, and retained rollback or negative outcomes.
- Add a versioned dense Llama/Qwen/Mistral/Gemma meta-surgeon transfer matrix
  with raw and architecture-normalized views, cold/within-target/meta controls,
  explicit checkpoint/size/family/corpus axes, five separate evidence metrics,
  three-seed grouped-interval requirements, and retained negative or
  unsupported cells.
- Add a deterministic cross-model transfer suite with leave-one-checkpoint,
  leave-one-size, leave-one-family, zero-target, and few-shot folds, strict
  lineage-safe source/target separation, complete per-fold metric evidence,
  resumable results, and versioned research protocol documentation.
- Add a fixed-budget ranking-objective study for pointwise, pairwise, listwise,
  and sequence methods with identical held-out lists, censored evidence,
  ranking/regret/calibration/frontier metrics, resource accounting, and a
  confidence-bounded recommendation policy.
- Add a bounded structural-model study for MLP, set, and graph surgeon paths
  with model/state-held-out splits, five-seed deterministic inference,
  resource accounting, topology/order/parameter ablations, and fail-closed
  complexity selection.
- Add state-bound interaction-aware acquisition with utility, safety,
  uncertainty, interaction uncertainty, diversity, cumulative resource caps,
  stale-parent rejection, retained negative outcomes, and bounded
  replan/rollback decisions.
- Add source-fitted architecture-normalized meta-features for structural role,
  model/state scale, family, quantization, hardware, and bounded mutation
  history, with explicit unknown-family and unsupported-MoE masks plus
  compatibility and coverage reports.
- Add explicit model-lineage graphs and machine-readable surgeon compatibility
  decisions across ancestry, architecture, features, targets, mutations,
  codecs, hardware, state versions, and operations, with held-out ancestry
  leakage rejection and compatibility evidence retained in model cards.
- Add calibrated state-dependent predictors for safety, quality, latency, memory,
  and non-additive error with grouped state/model/candidate leakage checks,
  validation intervals, separate failure probability, additive/stateless baselines,
  explicit negative outcomes, and fail-closed unseen-state inference.
- Add the versioned fixed-width current-state embedding contract with normalized
  architecture scalars/histograms, canonical mutation sets, order-sensitive bounded
  history, metric/constraint masks, explicit evidence outcomes, and deterministic IDs.
- Add bounded pre-mutation interaction features for activation/gradient overlap,
  redundancy, topology distance, mutation order, and cumulative error with explicit
  missingness, target outcomes, provenance context, deterministic IDs, and RAM/VRAM/time
  and candidate-cell budgets.
- Add the versioned mutation interaction dataset contract for single, ordered-pair,
  and cumulative evidence, additive/non-additive reconciliation, explicit terminal
  outcomes, topology distance, provenance, and lineage-safe leakage rejection.
- Add a fail-closed multi-axis architecture sequence compiler that revalidates
  state IDs, axis status, identity remaps, hardware alignment, ordered mutation
  history, and cumulative parameter/storage reconciliation before mutation.
- Add versioned deployable architecture state and symmetric decomposed distance
  records for depth, widths, heads, hidden/embedding sizes, low rank, sparsity,
  quantization, placement, mutation order, and physical artifact lineage, with
  explicit unknown/predicted/unsupported axes and fail-closed import.
- Add bounded hardware-cost predictors with train-only feature schemas,
  validation-calibrated intervals, held-out model/artifact/hardware checks,
  analytic and mean baselines, explicit OOD rejection, and retained negative
  or unsupported evidence cells.
- Add profile-partitioned hardware cost dataset records that retain repeated deployment
  distributions, artifact checksums, runtime configuration, benchmark provenance, and
  leakage-safe model/artifact/hardware lineage splits.
- Add fail-closed hardware-specific alignment rules that separate graph/codec legality
  from measured preference and require matched training/held-out microbenchmark evidence.
- Add profile-bound kernel/offload microbenchmark partitions with boundary-crossing shapes,
  ten-repetition confidence envelopes, explicit unstable outcomes, and retained unsupported
  or failed cells.
- Add immutable empirical hardware/runtime profiles with bounded bandwidth, transfer,
  GEMM, load, and llama.cpp probe protocols, repeated summaries, explicit unavailable
  capabilities, and environment-drift rejection.
- Add the v1.2 physical compression and quality-loss Pareto study contract with a
  preregistered five-target matrix, physical HF/GGUF lineage gates, deployment metric
  reconciliation, retained negative cells, and conservative interval frontiers.
- Add `modelsurgeon benchmark deploy` with content-addressed HF/GGUF plans,
  resumable JSON-lines runners, shared deployment metric units, hardware context,
  and distinct unsupported/failure/timeout/OOM/drift/invalid/interrupted outcomes.
- Add cumulative Hugging Face physical surgery sequencing with atomic child,
  reload, generation, parameter/storage reconciliation, ordered identities, and
  failed-stage rollback evidence.
- Add cumulative native GGUF sequencing with re-discovery boundaries, untouched
  payload identity checks, RAM/scratch limits, reload/generation evidence, and
  fail-closed child cleanup.
- Add the matched four-arm quantization-order study contract with reconciled
  quantization, surgery, and interaction effects plus explicit unsupported and
  inconclusive cells.
- Add the resumable `modelsurgeon benchmark` matrix CLI with deterministic plans, imports, audits, and reports.
- Add matched BF16/F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M quantization baseline records with explicit unsupported-tool evidence and loss-decomposition fields.
- Add the v1.1 artifact-bound competitor Pareto study record with uncertainty gates, complete deployment metrics, and explicit no-claim evidence.
- Add framework-neutral cumulative physical artifact outcomes with strict HF/GGUF publication gates and lineage reconciliation.
- Add shared bounded artifact integrity gates with fault injection, scoped staging recovery, reload/generation checks, and non-overwriting promotion.
- Add a versioned deployable benchmark evidence schema with explicit terminal
  outcomes, equal-budget identities, metric units, uncertainty, artifact lineage,
  provenance, deterministic reports, and a v0-to-v1 migration.
- Freeze the v1.1 equal-budget competitive benchmark protocol with content-addressed
  model/task/method decisions, exact metric and budget contracts, retained unknown cells,
  preregistered statistical limits, and an explicit contamination/license audit.
- Add a fail-closed subprocess competitor adapter contract with executable-version and
  artifact-bound support claims, transactional output publication, read-only source checks,
  bounded logs/artifacts, resource telemetry, and distinct failure outcomes.
- Add revision-pinned Wanda and SparseGPT equal-budget baseline records for two model families,
  matched 20%/50% sparsity and three-seed cells, explicit magnitude/random controls, and a
  retained unsupported evidence matrix when external executables are unavailable.
- Add independent LLM-Pruner, SliceGPT, ShortGPT, and Minitron-style structured baseline cards
  with 10%/20% physical targets, operation matrices, license exclusions, and reload/generation-
  bound artifact reconciliation.

## 1.0.0 - 2026-09-07

- Define the supported package-level Python API, experimental implementation boundary, and
  schema-versioning compatibility policy for the v1.0 stabilization work.
- Smoke-test plain, non-interactive help for every public CLI command and document
  shell completion discovery without modifying user shell configuration.
- Normalize malformed or empty GGUF mmap inputs to the public parser-error boundary and
  add deterministic hostile-byte and checkpoint-path alias coverage.
- Document the Windows 11 and WSL2 consumer workflow, including validated portable
  path, interrupt, disk, mmap, and telemetry coverage plus the retained native
  GGUF/CUDA-offload evidence required for a host-specific compatibility claim.
- Retain measured Windows 11 and WSL2 SmolLM2-135M Q4_K_M CUDA-offload evidence,
  fixture provenance, platform identities, and explicit runtime-version limitations.
- Add the `metric-schema-v1` golden compatibility contract for persisted metric records,
  evaluator metric definitions, migration identities, and supported schema versions.
- Add `modelsurgeon report` for deterministic, evidence-backed JSON and offline HTML reports
  from persisted runs or candidates, with redacted provenance and actionable incomplete-ID errors.
- Add `modelsurgeon features` for manifest-selected, CPU-safe, record-budgeted feature extraction,
  component filtering, atomic partition reuse, and explicit per-extractor skip reasons.
- Publish the v1.0 scientific results ledger with content-addressed evidence run IDs, Q1–Q8 and
  quantization findings, consumer-hardware costs, and explicit prediction/masking/physical-surgery
  limitations.
- Add end-to-end Hugging Face/safetensors and native GGUF user guides covering pinned inputs,
  bounded experiments, safe publication, Q4_K_M edits, resume, reproduction, and failure recovery.
- Add approval-gated release automation for verified sdist/wheel builds, clean Python 3.12
  installation, changelog generation, GitHub artifact attestations, and PyPI trusted publishing.
- Declare the existing NumPy runtime requirement so clean wheel installations can import the
  complete public CLI surface.
- Add `modelsurgeon calibrate` for strict revision-pinned calibration plans, bounded tokenization,
  content-addressed atomic cache publication, dry-run isolation, and interruption-safe refresh.
- Add `modelsurgeon generate-dataset` to start or resume a trusted mutation campaign and emit
  validated, leakage-safe JSONL splits with a standalone progress and failure manifest.
- Add `modelsurgeon reproduce RUN_ID` with schema-v2 resolved recipes, exact-command dry
  runs, content-addressed integrity and environment checks, trusted replay adapters, and
  original-linked tolerance comparisons for reproduced metrics.
- Add hardware-bound CPU RAM, CUDA VRAM, runtime, process-I/O, and GGUF-streaming
  regression budgets with a small pull-request lane, separate consumer CPU/GPU workflows,
  canonical reports, strict alert decisions, and measured RTX 3060 workstation evidence.
- Complete the revision-pinned 100M–7.25B consumer-hardware scale study with measured
  analysis, feature, reversible mutation, forward-evaluation, RAM, VRAM, cache-disk,
  placement, and failure evidence plus conservative defaults for a 12 GB GPU host.
- Extend the dependency-ordered roadmap from v1.1 through v2.0 with implementation-ready
  competitive benchmarking, physical deployment, hardware-aware, multi-axis, interaction,
  transfer, repair, rigor, production and autonomous-optimizer milestones plus idempotent
  GitHub materialization and read-back audit tooling.
- Publish a generated 108-cell HF/GGUF architecture compatibility matrix with separate
  verified, experimental, unsupported, and unknown states so untested combinations cannot
  be presented as supported.
- Add a schema-versioned, immutable, Apache-2.0 multi-family evaluation ladder from 135M
  through 7.25B parameters with explicit purpose, consumer-hardware modes, dataset
  compatibility, access constraints, and a content-addressed ladder identity.
- Demonstrate deterministic low-memory Q4_K_M MLP surgery across every transformer layer,
  including copy-only legacy Q5_0 handling, model-wide metadata safety, pinned llama.cpp
  load/perplexity/generation/throughput evidence, and working Windows RSS telemetry.
- Add an exhaustive GGUF family/storage-profile/operation compatibility matrix whose support
  claims require artifact-bound pinned llama.cpp load/forward evidence, with explicit
  structural-only and unsupported cells plus a scheduled real-fixture regression gate.
- Add a bounded Llama/Qwen coordinated hidden-dimension feasibility study that inventories
  global and normalization consumers, computes codec-axis alignment, and emits an explicit
  no-mutation rejection while rotary/config and weight-tying proofs remain incomplete.
- Add bounded one-tensor native GGUF low-rank replacement with selective encoded-block decode,
  explicit NumPy SVD workspace preflight, validated same-codec requantization, separate
  reconstruction/quantization errors, and unchanged-payload checksum reconciliation.
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
  fail-closed timeout or non-zero-exit reporting, current official version parsing,
  and non-interactive CLI termination.
- Benchmark baseline/candidate GGUF perplexity with pinned llama.cpp under one
  content-addressed sample and runtime configuration, require matching tokenizer
  metadata, parse finite structured estimates, and preserve bounded raw failure logs.
- Benchmark pinned llama.cpp GGUF prompt and generation throughput with phase latency
  samples, child-process peak RAM/VRAM, explicit warmup/thread/offload/context settings,
  raw failure logs, and drift-gated comparison ratios.
- Discover and configure external llama.cpp quantization binaries without bundling,
  reject missing or revision-drifted tool sets, capture exact invocations and bounded
  logs, transactionally publish validated outputs, and index legacy mixed-recipe blocks
  for byte-preserving copy without claiming native codecs.
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
- Add a versioned provider conformance matrix and deterministic offline tests
  for local, compatible endpoint, hosted, and no-LLM modes. Shared invocation
  now revalidates provider identity, request provenance, response provenance,
  and structured output before accepting a supported result.
- Freeze the v2.2 provider-layer release boundary with dependency evidence,
  explicit no-LLM unsupported cells, core provider-import isolation checks, and
  a machine-readable release audit. This is control-plane protocol evidence,
  not live provider or model-quality benchmark evidence.
