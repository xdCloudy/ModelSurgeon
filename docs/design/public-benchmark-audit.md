# Public benchmark audit and falsification

`modelsurgeon.evaluation.public_audit` is a read-only audit boundary for
published study records.  It recomputes coverage and claim support from typed
protocols, cells, metrics, confidence bounds, lineage, contamination status,
reproduction status, and retained negative outcomes.

Blocking findings include protocol drift, missing or unexpected cells,
contaminated or unreproduced measured cells, claims over unsupported cells,
missing confidence bounds, undeclared metrics, and omitted negative results.
The auditor also compares mean and median directional decisions to expose
aggregation-sensitive claims.  It never edits evidence and always retains a
limitation explaining that the audit is bounded by supplied records and does
not certify model safety.
