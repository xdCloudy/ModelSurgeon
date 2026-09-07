# Surgeon lineage and compatibility

`modelsurgeon.surgeon.compatibility` is the v1.6 compatibility boundary for
cross-family meta-surgeon training and inference. It consumes explicit model
lineage and contract evidence; it never infers ancestry, family, or capability
from repository names or hidden provider metadata.

## Lineage and leakage

`ModelLineageGraph` records immutable model revisions, explicit aliases,
parent models, derivation kind, family, and architecture revision. Aliases are
globally unique and parent references must resolve. Cycles fail closed. The
transitive closure is used by `validate_heldout_lineage()` so a derived model
cannot cross a training/held-out boundary through an ancestry alias.

## Compatibility decisions

`decide_compatibility()` compares ancestry, architecture, feature schema,
target schema, mutation schema, codec, hardware, state schema, and declared
operation capability independently. Every dimension emits a machine-readable
reason with one of `exact`, `compatible`, `adaptable`, `unsupported`, or
`unknown`. Missing required evidence is unknown; incompatible versions and
missing operations are unsupported. Adaptable decisions are allowed only when
the caller explicitly enables adaptation. Each decision is content-addressed
by its complete source, target, policy, and reason evidence.

`SurgeonEvaluationCard` can retain the serialized compatibility decision so a
bundle refuses inference when required features, operations, or state versions
are absent or incompatible.
