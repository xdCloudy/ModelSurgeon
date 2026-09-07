# Interpreted OptimizationSpec preview

The experimental v2.3 chat entry exposes a deterministic preview of the exact
`ObjectiveContract` record that a future trusted execution adapter would
receive. The preview is a read-only projection of the existing compiler and
policy result; it does not inspect target weights, choose tensors, create a
mutation plan, or execute a model operation.

`modelsurgeon.search.build_spec_preview(intent)` renders canonical hard
constraints, soft objectives/preferences, any typed budgets or allowed
operations, unresolved field IDs, policy diagnostics, and request/provider/tool
provenance. Unsupported fields remain visible and block execution. A
non-executable outcome never carries a spec.

Conflict diagnostics are part of this canonical preview rather than a hidden
provider decision. A hard contradiction is refused with its minimal field/span
witness. An ambiguous soft preference remains unresolved until an explicit
clarification selection is recompiled; no preview or confirmation can infer an
ordering or relax a hard constraint.

The `spec_digest` is the digest of the exact serialized `spec` object, while
`spec_identity` is the existing `ObjectiveContract.contract_id`. A changed
constraint, preference, approval policy, or plugin binding therefore changes
the spec identity. `SpecPreview.recompile()`/`.edit()` returns a new preview
with a deterministic `SpecDiff`; a prior confirmation is never reusable for a
material change.

Calling `SpecPreview.confirm(approval_id=...)` is the explicit confirmation
boundary. It returns a `SpecSubmission` containing the exact spec and digest;
the default objective policy requires a trusted approval ID. The current chat
command intentionally stops at the preview and makes no execution or live
provider/model evidence claim. Future execution adapters must accept only the
confirmed submission and remain responsible for their own trusted approval and
mutation boundaries.

The versioned contract is recorded in
`docs/research/v2.3-spec-preview-v1.json`. This feature is planned/experimental
until the v2.3 vertical-slice release is complete.
