# Measured physical-strategy selection

The final physical decision is separate from search prediction. A candidate can
be considered for publication only when it has measured quality, measured
deployment and optimization costs, a positive artifact size, a distinct
immutable child artifact, and complete provenance for repair, teacher, dataset,
quantization, artifact, and deployment.

## Required controls

Every repair decision requires measured no-repair and quantization-only controls.
Those controls make it possible to attribute any quality or deployment change to
repair rather than to quantization or an unmeasured baseline. A missing or
predicted control produces an `unknown` outcome. An incompatible required
control produces `unsupported`.

## Fail-closed budgets

Quality, deployment cost, optimization cost, and artifact size are hard gates.
Candidates that fail one of them are excluded with a reason and cannot be
promoted by a predicted economic benefit or by a smaller artifact alone. If the
required controls were measured but every final strategy fails those gates, the
decision is `failed`.

Among eligible candidates, selection is deterministic: maximize measured
quality, then minimize deployment cost, optimization cost, artifact size, and
finally candidate ID. The decision record includes a schema version, stable
decision ID, selected candidate, measured controls, and all exclusions so the
result can be audited or replayed.
