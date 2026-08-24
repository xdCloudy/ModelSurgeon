# Dataset leakage audit

`modelsurgeon.datasets.audit_dataset_leakage` is the fail-closed boundary between persisted mutation examples/split manifests and downstream training.

## Required inputs

The audit consumes a validated tuple of `MutationExampleRecord` values plus either a `GroupedSplitManifest` or `HeldOutSplitManifest`. It audits the persisted manifest as supplied instead of regenerating a split, so corrupted, incomplete, or externally modified manifests remain detectable.

`LeakageAuditReport.require_clean()` is the training gate. Downstream training or matrix construction must not proceed from a report containing findings.

## Leakage classes

The version-1 audit reports:

- `manifest_coverage`: dataset examples missing from the manifest, or manifest IDs missing from the dataset.
- `exact_candidate`: the same canonical mutation candidate for the same model identifier occurs in more than one partition. Model revision aliases intentionally share the identifier key.
- `near_duplicate_candidate`: candidates for the same model have the same mutation kind, targets, parameter names, and non-numeric parameter values, but differ in one or more numeric parameter values. This structural definition avoids an arbitrary numeric-distance threshold.
- `shared_component`: the same canonical component of the same model identifier occurs across partitions.
- `model_ancestry`: models with explicit common ancestry occur across partitions.
- `target_derived_feature`: a feature is explicitly classified as target-derived by policy or feature metadata.

## Explicit evidence only

The audit does not guess model lineage from repository names, revision strings, or architecture labels. Lineage is supplied as canonical `ModelAncestry` entries. Transitive ancestry is resolved deterministically; cycles fail closed.

Target-derived feature detection uses explicit evidence:

- `LeakageAuditConfig.target_feature_names`;
- `LeakageAuditConfig.target_feature_extractors`;
- feature metadata `target_derived=true`; or
- feature metadata `source_phase` equal to `post_mutation`, `delta`, or `target`.

This keeps the audit conservative without falsely treating ordinary pre-mutation features as target leakage based only on their numeric values.

## Candidate identity

Exact candidate identity includes the model identifier and canonical mutation ID. The model revision is deliberately excluded so aliases such as a branch, tag, and commit of one logical model cannot be split into duplicate candidates.

Near-duplicate identity retains the model identifier, mutation kind, target component IDs, parameter keys, and all non-numeric parameter values. Numeric parameter values are replaced by a typed placeholder before hashing. The audit therefore catches parameter sweeps of the same structural candidate across partitions while allowing the same operation shape on unrelated models.

## Machine-readable result

Every finding records:

- a stable leakage kind;
- a deterministic key;
- affected partitions;
- affected example IDs; and
- a human-readable detail.

Findings are canonically ordered, and the report records the audit version, manifest kind, example count, and explicit policy configuration.
