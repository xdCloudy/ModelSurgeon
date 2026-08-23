# ModelSurgeon Architecture

## Purpose and boundaries

ModelSurgeon learns to predict the consequences of neural-network structural changes, executes controlled mutations, evaluates them, and searches for smaller or faster models under explicit quality constraints. It is not a generic training framework, benchmark suite, model hub, or wrapper around a single pruning method.

Two paths are equally fundamental:

```text
HIGH-PRECISION PATH                     LOW-HARDWARE PATH
HF/safetensors                          existing quantized GGUF
      |                                        |
inspect + instrument                    mmap + inspect in place
      |                                        |
structural surgery                      selective dequantize
      |                                        |
new checkpoint                          mutate + requantize
      |                                        |
optional quantization                   streaming new GGUF
```

The low-hardware path never requires a complete floating-point model in RAM or on disk for ordinary surgery. Unknown GGUF architectures and unsupported quantization layouts fail closed.

## End-to-end pipeline

```mermaid
flowchart TD
  M[Target model] --> A[Format and architecture adapter]
  A --> G[Stable component and coupling graph]
  G --> I[Static and runtime instrumentation]
  I --> F[Versioned feature records]
  F --> P[Surgeon prediction]
  P --> C[Candidate selection]
  C --> U[Transactional mutation]
  U --> E0[Tier 0 structural validation]
  E0 -->|pass| E1[Tier 1 calibration]
  E1 -->|promising| E2[Tier 2 behavioural]
  E2 -->|selected| E3[Tier 3 full validation]
  E0 -->|fail| R[Rollback]
  E1 -->|poor| R
  E3 --> D[Experiment dataset]
  D --> P
```

## Package architecture

| Package | Responsibility |
|---|---|
| `adapters` | HF, safetensors, GGUF, architecture-family and quantization boundaries |
| `graph` | stable component IDs, discovery, topology, coupling and mutation constraints |
| `features` | versioned static, spectral, activation, gradient, redundancy and runtime records |
| `instrumentation` | calibration data, hook lifecycle and bounded aggregation |
| `surgery` | mutation contract, masks, physical edits, provenance and rollback |
| `evaluation` | tiered structural, perplexity, behavioural, latency and resource metrics |
| `experiments` | definitions, state machine, persistence, cache, budgets and resumability |
| `datasets` | mutation examples, schemas, validation and leakage-safe splits |
| `surgeon` | heuristics, linear/tree/MLP models, calibration, uncertainty and inference |
| `search` | candidates, active learning, constraints, objectives and sequential policies |
| `explain` | attribution, decision summaries and reports |
| `cli` | user workflows; business logic remains in packages above |

Adapters expose canonical capabilities rather than leaking library-specific objects. Feature extractors and mutations consume component graph nodes. Storage schemas contain canonical IDs and versioned records, never live framework objects.

## Component identity and graph

A component ID is a validated, immutable sequence of typed path segments. Examples include `model.layers.17.self_attn.head.4` and `model.layers.12.mlp.up_proj.channel.1830`. IDs remain stable across instrumentation, mutation, evaluation, datasets, predictions and reports. Physical surgery emits an explicit old-to-new identity mapping; it never silently reinterprets an ID after indices shift.

```mermaid
classDiagram
  class ComponentNode {
    ComponentId id
    ComponentKind kind
    shape
    dtype
    adapter_metadata
  }
  class ComponentEdge {
    EdgeKind kind
    constraints
  }
  ComponentNode "1" --> "*" ComponentEdge
  ComponentEdge "*" --> "1" ComponentNode
```

Edges express `parent`, `child`, `consumes`, `produces`, `coupled-with`, and `constrained-by`. A mutation plan is valid only when the adapter resolves every affected node, its coupled closure, axis semantics, and shape/quantization constraints.

## Native GGUF surgery

GGUF is a first-class analysis and physical-surgery format. The subsystem separates container I/O, architecture mapping, tensor layout, quantization codecs, mutation planning and validation.

```mermaid
flowchart LR
  S[Source GGUF mmap] --> X[Metadata + tensor index]
  X --> Q[Architecture and quantization adapters]
  Q --> P[Coupling-aware mutation plan]
  P --> B[RAM/VRAM/disk budget planner]
  B --> T[Transactional output writer]
  S -->|unchanged byte ranges| T
  S -->|affected blocks only| D[Decode/dequantize]
  D --> M[Float mutation]
  M --> R[Requantize + validate blocks]
  R --> T
  T --> V[llama.cpp load/forward validation]
```

The writer uses copy-on-surgery: untouched tensors are copied byte-for-byte when container alignment permits; changed tensors alone are decoded, mutated, encoded and written. It preflights free disk space, writes to a staging path, checkpoints completed ranges, verifies checksums/metadata, and atomically publishes the destination. Interrupted output remains resumable or safely removable; the source is immutable.

Memory modes are `full`, `tensor`, and `streaming`. Each operation declares a peak-memory estimate and a scratch-space estimate. The planner respects configurable RAM and VRAM ceilings, supports CPU-only execution and disk-backed intermediates, and may shrink chunk size after an OOM. A 20–40 GB GGUF remains memory-mapped rather than resident.

Quantization is represented by a codec interface with exact block shape, packing, alignment, decode, encode and validation behavior. F32/F16/BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K and IQ families are separate capabilities; no K-quant is inferred from another. Initial production priority is Q8_0, Q6_K, Q5_K_M and Q4_K_M. Implementations track the llama.cpp specification revision they match and use conformance vectors.

Physical GGUF mutations resolve model-family rules for Llama, Qwen, Mistral and Gemma. MLP channel removal closes over gate/up/down tensors; attention head removal closes over Q/K/V/O and accounts for GQA/MQA/KV counts; layer removal updates tensors, names and metadata; low-rank replacement touches only selected tensors. Hidden-dimension surgery remains gated behind a bounded feasibility study because it couples most of a model.

## Feature and provenance model

Each `FeatureRecord` identifies the model revision, component, extractor/schema versions, value, aggregation, sample set and precision provenance. Quantized features record codec, bits per weight, direct-versus-locally-dequantized source, sampled blocks and estimated error. This allows the surgeon to learn different uncertainty for BF16 and Q4-derived measurements.

Large activations are reduced online into mergeable statistics. Raw tensors are opt-in, bounded, disk-backed and content-addressed. GPU tensors are released at extractor boundaries.

## Experiment lifecycle and storage

SQLite stores normalized metadata and state; Parquet stores tabular feature/example partitions; content-addressed filesystem artifacts store tensors, checkpoints and reports.

```mermaid
stateDiagram-v2
  [*] --> Planned
  Planned --> Running
  Running --> Evaluating
  Running --> Interrupted
  Running --> RecoverableOOM
  Interrupted --> Running
  RecoverableOOM --> Planned
  Evaluating --> Succeeded
  Evaluating --> Rejected
  Running --> Failed
  Succeeded --> [*]
  Rejected --> [*]
  Failed --> [*]
```

Deterministic experiment IDs derive from canonical configuration and immutable input revisions. Atomic state transitions, leases, completion detection and idempotent artifact publication make campaigns reboot-safe. Records capture model/dataset revisions, git commit, lock digest, seeds, hardware/software inventory, baseline/post metrics, deltas and timings.

For native quantized surgery, every experiment can include a matched requantization control: decode and re-encode the affected region without surgery. Evaluation reports separate baseline quantization loss, surgery loss and their interaction.

## Surgeon training and active learning

The model ladder is heuristic, linear/logistic, LightGBM/XGBoost, small MLP, then set/sequence/GNN/transformer only if validation shows simpler models saturate. Targets include delta loss/perplexity/behaviour/latency/parameters and probability of satisfying a constraint. Quantization, feature precision, estimated quantization error and hardware are input features.

Leakage-safe splitters group by component, layer, mutation, model and architecture family. Cross-model claims require completely held-out models. Active learning combines calibrated uncertainty, predicted utility and diversity under explicit experiment and hardware budgets; it deduplicates canonical candidate descriptions and records selection propensity.

## Search and repair

Objectives are user-defined combinations of quality, parameter, latency, memory and disk costs with hard constraints. Search operates on graph-valid mutation plans, evaluates short sequences transactionally, and may apply LoRA repair, short fine-tuning or distillation as separately budgeted operations. A Pareto archive prevents one hard-coded reward from defining all workflows.

## Hardware, reproducibility and safety

- Primary baseline: RTX 3060 12 GB, i5-14600K, 64 GB RAM, Windows 11/WSL2.
- All substantial loops are batched, bounded, cached and resumable.
- OOM is classified and retried with smaller batches/chunks or CPU offload.
- Original checkpoints are read-only by default; destinations must differ and publish atomically.
- Remote model code is opt-in and recorded.
- `modelsurgeon reproduce RUN_ID` reconstructs config, revisions, seed, environment and commands.
- GPU CI is optional and scheduled; PR CI uses tiny CPU fixtures.

## Architectural decisions

ADRs under `docs/design/` record stable IDs, local-first storage, progressive surgeon complexity, tiered evaluation, transactional mutation and native copy-on-surgery GGUF design.

