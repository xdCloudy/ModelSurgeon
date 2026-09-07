# Architecture-normalized meta-surgeon features

`modelsurgeon.features.meta_schema` is the v1.6 cross-model feature boundary.
It turns architecture, state, quantization, hardware, and bounded mutation
history into a fixed schema without pretending that unlike structures are
equivalent.

## Source-only fitting

`fit_meta_feature_normalizer()` accepts only samples explicitly marked
`source`. Numeric means/scales, categorical vocabularies, mutation-history
tokens, source sample IDs, and source model revisions are retained in the
normalizer record and contribute to its deterministic schema ID. Target
samples can only be transformed; they cannot change fitted statistics or
vocabulary.

Numeric fields use normalized layer depth, width/hidden and head/KV ratios,
log parameter count, and mean state values. Missing values are zero-filled
only alongside an explicit unknown mask. Family, structural role,
quantization, and hardware are categorical fields with a reserved unknown
bucket. History is bounded and tokenized from the source vocabulary.

## Compatibility and unsupported structures

The transformed vector retains field-level measured, unknown, and unsupported
statuses. Unknown target families remain unknown rather than being assigned a
dense family embedding. MoE samples are marked unsupported by default for
numeric transfer fields unless the caller explicitly enables MoE handling.
`MetaFeatureCompatibility` and `MetaFeatureCoverageReport` expose comparable
fields, missingness, unsupported values, unknown categories, and vocabulary
collisions for the pinned multi-family ladder.
