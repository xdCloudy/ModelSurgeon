# Component graph validation

Graph construction performs structural checks; `validate_component_graph` performs complete cross-record validation before a graph can drive instrumentation or mutation. It returns every deterministic violation rather than stopping at the first, and each violation names the exact canonical component IDs and stable rule identifier.

Version 1 validation requires reciprocal parent/child and produces/consumes edges, acyclic child hierarchy and producer dataflow, resolvable endpoints and constraint members, and complete pairwise coupled plus constrained edges for every mutation constraint. Constrained edges must name an existing constraint whose member set contains both endpoints.

`validate_graph_records` accepts raw nodes, edges, and constraints so import and migration code can diagnose dangling records that the strict `ComponentGraph` constructor refuses. `GraphValidationReport.raise_for_errors` converts the complete report to `GraphValidationError` only when a caller needs fail-fast control flow.
