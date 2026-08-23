# Mutation dataset validation

Issue #85 validates serialized supervised mutation examples before split, training, or export. Validation returns all deterministic issues it can classify instead of failing on the first corrupt row.

## Machine-readable issues

Each issue contains a stable rule, record index, optional example ID, field path, error code, and human-readable detail. The current rule set is:

- `schema`: required fields and typed missingness/state contracts.
- `finite_range`: finite feature/metric/timing values, positive schema/count fields, and unsigned 64-bit seeds.
- `component_reference`: canonical component IDs, mutation identity, and affected-component agreement.
- `revision_provenance`: mutation model/tool revisions and feature sample dataset/tokenizer provenance.
- `target_calculation`: arithmetic and unit consistency for measurable delta targets.
- `duplicate_id`: duplicate dataset example IDs.

Raw mappings and typed `MutationExampleRecord` instances use the same validation path, so corrupt persisted rows can be audited without first constructing a typed record.

## Delta calculation convention

When a measured delta metric can be associated with measured baseline and post metrics, the expected target is:

`delta = post - baseline`

The validator recognizes a direct metric name and the common `delta_<metric>` / `<metric>_delta` forms. Comparison uses explicit absolute and relative tolerances from `DatasetValidationConfig`. A measured delta with no matching baseline/post metric is left untouched because some behavior targets are produced directly rather than by scalar subtraction.

If units are supplied, the baseline and post units must agree and the delta unit must match the source unit.

## Provenance checks

The nested mutation record is parsed by the canonical mutation serializer. Its mutation ID and affected components must match the example, its input revision must match the model revision, and its tool revision must match the example version context.

Feature component IDs must parse canonically. Data-derived features with a sample context must match the example dataset identifier, revision, split, tokenizer, and tokenizer revision.

## Dataset-wide identity

Example IDs are unique within one validation call. Duplicate IDs are always reported, even when the rows are byte-identical; retry collapsing belongs to the dataset builder before persisted dataset publication.
