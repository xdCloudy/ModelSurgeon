# Hardware-conditioned search objectives

The search layer treats deployment placement as part of candidate identity. A
candidate therefore includes its structure identity, accepted parent checkpoint,
runtime profile, CPU/GPU placement, GPU offload boundary, predicted evidence,
and measured evidence. Two candidates with the same structure but different
offload boundaries cannot silently collapse into one search state.

## Feasibility and promotion

`HardwareConstraint` expresses a hard maximum or minimum for latency, memory,
disk, throughput, or parameter count. Every evidence value carries a source,
status, and uncertainty. Constraint checks use the upper bound for maxima and
the lower bound for minima. A predicted value can identify a promising
structure/placement pair, but it cannot promote that pair when a hard
constraint requires measurement. Unknown and unsupported evidence remains
explicit and is never treated as zero.

## Objective-aware selection

`DeploymentObjective` supplies an explicit direction, reference scale, and
weight for a requested deployment metric. Ranking uses the measured objective
score after feasibility is established, so parameter reduction is not an
automatic win when it worsens the requested deployment objective. Predicted
states remain available for evaluation and future evidence collection.

## Durable evidence

Candidate identity, resume state, reports, and lineage records all retain the
hardware profile and offload boundary. An accepted lineage checkpoint can only
be attached to a promotable assessment. This keeps rollback and resumed search
reproducible when the same structure is evaluated on multiple runtimes or
placement boundaries.

The checked-in tests cover identity separation, conservative bounds,
prediction-only gating, objective-aware ranking, and persistence across these
records. They are contract fixtures; deployment studies must add measured
runtime evidence for the target hardware.
