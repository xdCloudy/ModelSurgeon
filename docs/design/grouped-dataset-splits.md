# Grouped mutation dataset splits

Issue #86 creates deterministic train/validation/test manifests while preventing configured component, layer, or mutation-family groups from crossing split boundaries.

## Validation boundary

Only mutation examples that pass the #85 dataset validator are eligible. Duplicate example IDs or corrupt provenance fail before any assignment is produced.

## Connected grouping

Grouping is transitive. Each example contributes one or more leakage keys, then examples sharing any key are unioned into a connected component. This means a multi-component mutation can bridge otherwise separate groups: if example A touches component X, example B touches X and Y, and example C touches Y, A/B/C are one indivisible split group.

This is necessary because hashing each example or affected-component tuple independently would allow X or Y to appear in more than one partition.

### Component mode

Each affected component ID is a group key. Any examples sharing an affected component are connected.

### Layer mode

Canonical layer paths are derived from affected component IDs using the common `layer`, `layers`, `block`, `blocks`, or `h` path segment followed by a numeric index. The complete layer path is retained, so encoder and decoder layers with the same numeric index remain distinct.

For non-standard component paths, an integer `layer_index` mutation parameter is an explicit fallback. If no layer identity can be established, splitting fails rather than inventing a group.

### Mutation-family mode

An explicit `mutation_family` mutation parameter is preferred. Otherwise family identity is mutation kind plus `candidate_scope` when present, falling back to mutation kind alone.

## Deterministic assignment

Connected groups receive stable content-derived group IDs. Groups are ordered by SHA-256 over `(seed, group_id)` and assigned whole to the partition with the largest normalized deficit against configured example-count ratios. Input order therefore does not affect the manifest.

The algorithm identifier is `connected-groups-greedy-v1`.

## Manifest provenance

A manifest records:

- schema/algorithm version;
- grouping mode;
- unsigned 64-bit seed;
- train/validation/test ratios;
- group counts and example counts per partition;
- every group ID, leakage key, assigned partition, and example ID.

This gives downstream leakage auditing enough information to verify that no configured key crossed a split boundary.
