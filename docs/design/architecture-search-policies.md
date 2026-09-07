# Architecture search policies

Architecture policy v1 consumes complete states emitted by the bounded candidate
space. A state carries its canonical architecture identity, objective estimates
with optional confidence bounds, hard-constraint observations, evidence status,
operation, parent IDs, and provenance. The policy never constructs an
architecture that is not already present in the legal candidate pool.

The Pareto beam policy filters predicted hard-constraint violations, retains
unsupported, failed, unknown, and already-selected outcomes in its decision
record, and computes a conservative Pareto frontier. A candidate can dominate
another only when its worst-case objective bounds are no worse and strictly
better for at least one objective. Beam selection then uses deterministic
seeded ties and assignment diversity.

The evolutionary policy seeds a bounded population from that same frontier. It
performs one-axis mutations and assignment crossovers only when the resulting
complete state exists in the legal candidate pool. Elites are retained, new
survivors are selected from the conservative frontier, and population,
generation, and evaluation budgets are hard limits. Parent IDs and operation
names remain in every decision for auditability.

`ArchitecturePolicyState.to_record()` and `from_record()` provide a stable
resume cursor. The policy ID commits the configuration and objective/constraint
definitions, so a resume cannot silently use a different policy. Identical
candidate and evidence order produces identical decisions without process RNG
state.

This contract selects predicted states; only the architecture lifecycle may
promote passing measured evidence into an accepted deployable frontier. The
checked-in fixtures are synthetic protocol evidence. They do not claim that
either policy improves real deployment quality, cost, or hypervolume; that
requires the downstream multi-axis study with held-out models and measured
artifacts.
