# Mutation record serialization

The mutation record schema persists the canonical request and plan, input and
tool revisions, preconditions, expected and actual resource deltas, outcome
state, and old-to-new component identity mappings. Records use strict JSON
objects with no unknown fields and canonical key ordering. The request-derived
SHA-256 mutation identity is verified during every read, so record changes cannot
silently change the operation being described.

An identity mapping has one old component and zero or more canonical new
components. Zero targets explicitly means removed; the redundant `removed` flag
is checked on read to detect ambiguous or corrupted data. Later remapping logic
can build composition semantics on this stable wire representation.

Local input paths are replaced with `<redacted-local-path>` by default while
content and tool revisions remain available for reproduction. Callers may opt
into retaining the local path only for trusted local records. Redaction never
changes the deterministic mutation identity.
