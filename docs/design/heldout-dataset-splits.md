# Model and architecture-family held-out splits

Issue #87 creates generalization splits where complete models or complete architecture families are held out from training.

## Validation and family identity

Examples must pass the #85 mutation dataset validator before split construction. Architecture-family grouping consumes the canonical `ModelFamily` contract from #11 (`llama`, `qwen`, `mistral`, `gemma`) and fails closed on unsupported family strings rather than inferring from model names.

If one model identifier is observed with conflicting canonical families across records, split creation fails.

## Complete-model boundary

Model mode groups by `model.identifier` and intentionally ignores `model.revision` for assignment. Every revision alias, branch name, tag, or immutable commit recorded under the same model identifier therefore remains on the same side of the boundary.

The manifest still records every observed model revision so the alias/revision coverage is auditable.

## Complete-family boundary

Architecture-family mode groups all models with the same canonical family key. No example from a held-out family appears in training.

## Assignment strategies

### Explicit holdout

`validation_holdouts` and `test_holdouts` name complete model identifiers or canonical family names. All unlisted groups remain in training. This supports the direct research design:

- train: models/families A, B, C
- test: unseen model/family D

Requested keys must exist, validation and test holdouts must be disjoint, and at least one training group must remain.

### Seeded ratios

When no explicit holdouts are configured, complete groups are ordered by SHA-256 over `(seed, heldout_key)` and assigned whole by the same deterministic ratio-deficit principle used for grouped splits.

Seeded train/validation/test requires at least three distinct model or family groups. Architecture-family mode reports the observed count and recommends adding families or using explicit holdouts when this condition is not met.

## Manifest provenance

The manifest records the mode, assignment strategy, seed, ratios, requested holdouts, group/example counts, complete group keys, model identifiers, model revisions, assigned partitions, and example IDs. The algorithm identifier is `heldout-groups-v1`.
