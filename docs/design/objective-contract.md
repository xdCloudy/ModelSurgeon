# Versioned objective contract

The objective contract is the boundary between autonomous search and the
metrics used to select a candidate. It makes the distinction between a
requirement that must never be traded away and a preference that may be
optimized explicit and serializable.

## Contract shape

Every contract has schema version `1`, at least one hard constraint, and at
least one soft objective. Metrics declare their units. The contract ID is a
stable hash of the canonical contract record and therefore includes the
approval policy and plugin bindings.

Hard constraints are evaluated first. A missing observation, a threshold
violation, or an uncertain bound that cannot prove the constraint is safe
produces `infeasible`; no objective score is produced. This prevents reward
optimization from compensating for an unmet quality, resource, or safety
requirement.

Soft objectives may use identity, baseline-ratio, or min-max normalization.
Missing or unmeasured soft evidence produces `unknown` and leaves the
candidate incomparable. A Pareto comparison only succeeds when both
candidates are feasible and fully comparable.

## Evidence and provenance

Observations can carry lower and upper bounds and sorted evidence references.
Maximization uses the lower bound and minimization uses the upper bound. This
keeps uncertain measurements conservative. The caller is responsible for
persisting the observation evidence alongside the run, artifact, and source
identity that produced it.

Supported built-in metrics include quality, perplexity, latency, peak RAM,
peak VRAM, file size, optimization time, repair cost, energy, latency gain,
and parameter count. Each has one canonical unit; mismatched units are
rejected at construction time.

## Objective plugins

An objective outside the built-in metric set must include an immutable plugin
binding with the plugin name, capability, plugin ID, configuration digest,
and trust mode. Evaluation requires a matching `objective` capability card,
the declared capability and trust mode, and (when configured) an explicit
approval allowlist entry. Missing or mismatched cards produce `unsupported`.
Plugin execution remains subject to the shared plugin resource budgets and
subprocess/trusted-in-process boundaries; the objective contract does not
weaken those controls.

The legacy settings translator in `contract_from_settings` maps existing
quality, perplexity, parameter-count, latency, memory, and disk-size options
to this contract without changing their direction or unit semantics.
