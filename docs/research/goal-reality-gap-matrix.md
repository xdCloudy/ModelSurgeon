# Goal-to-reality gap matrix

This matrix is the current implementation audit against `docs/goal.md`. It is
deliberately conservative: an item is not marked complete because a schema,
fixture, planner, or external adapter exists. It requires a first-party path
and retained evidence from the real model/runtime boundary.

| Goal requirement | Current evidence | Status | Remaining work |
| --- | --- | --- | --- |
| First-party autonomous execution | `HuggingFaceOptimizeRuntime` is the default executor for the narrow HF gated-MLP cell. | Partial | Extend the same evidence contract to attention, layers, low-rank, repair, quantization, and GGUF. |
| Real baseline measurement | `HuggingFaceMLPProofRuntime.baseline_measurement()` runs the pinned model over the local calibration corpus. | Partial | Add task-quality, memory, and deployment-runtime baselines to the general optimize contract. |
| Real candidate generation/search | Component-graph MLP candidates are enumerated deterministically, optional verified Meta-Surgeon predictions can order the real candidate measurements, and cumulative masks are measured. | Partial | Connect active-learning state and held-out transfer evaluation; measured outcomes remain authoritative over predictions. |
| Physical surgery and cumulative state | The first-party HF runtime executes a bounded cumulative MLP-channel sequence, publishes a distinct safetensors child at every accepted step, reloads and smoke-tests each child, and rehydrates the selected child on process restart. | Partial | Bind the full parent/child sequence into the public optimize artifact record and connect mixed attention/layer/low-rank edit sequences to the first-party runtime. |
| Quality and deployment evaluation | Perplexity, forward timing, reload, inference smoke, repeated load time, prompt/decode throughput, latency, RAM/VRAM sampling, artifact size, and sample arrays are retained in stage evidence for the HF cell. | Partial | Add exact task metrics and format-specific deployment runtimes beyond the current PyTorch path. |
| Artifact lineage and immutability | Source tree and complete reloadable child-directory manifest digests, no-overwrite publication, digest-verified resume, and orchestrator promotion checks exist. | Partial | Bind the full parent/child manifest sequence and container provenance into the public optimize artifact record. |
| Repair and quantization | Existing libraries contain separate contracts and experiments. | Not connected | Connect only after real reload/evaluation evidence exists for each format and method. |
| Pareto selection and stopping | Orchestrator requires measured final Pareto evidence; narrow runtime selects a feasible measured child. | Partial | Add multi-objective archives, deterministic stopping reasons, budgets, and alternatives over cumulative states. |
| Hardware-grounded behavior | Hardware profiles and inventory exist; the narrow runtime measures CPU forward timing. | Partial | Replace planning-only envelopes with runtime-enforced RAM/VRAM/disk measurements and failure evidence. |
| Stateful evidence flywheel | The first-party HF search retains real pre-mutation feature records for every measured candidate, publishes revision-keyed partitions through the atomic feature cache, and records cache paths/checksums in active-search evidence. | Partial | Enforce rediscovery/state-dependent candidates and add reusable post-mutation outcome ingestion without allowing stale or conflicting partitions. |
| Meta-Surgeon comparison | An optional signed, schema-checked Meta-Surgeon bundle can rank real candidates from retained pre-mutation features; the active-search record retains predictions, ordering, physical measurements, and regret for Meta-Surgeon, magnitude, random, and no-guidance baselines. | Partial | Run equal-budget held-out model/family/hardware comparisons and retain transfer, search-regret, and frontier results, including negative results when guidance does not improve the measured baseline. |
| Conversational control | Confirmed chat plans route through the same first-party runtime as direct optimize execution, with canonical campaign state and approval evidence. | Partial | Extend chat-executable coverage as additional model formats, operations, and task metrics gain real runtime evidence. |
| Independent acceptance campaign | Unit/fixture coverage is broad; no fresh multi-family end-to-end acceptance campaign has been claimed. | Not complete | Run and retain fresh evidence across multiple model families/configurations, including honest no-feasible cells. |

The default optimize path therefore remains an experimental, narrow verified
cell. This document must be updated after each real acceptance tranche; it is
not a substitute for that evidence.
