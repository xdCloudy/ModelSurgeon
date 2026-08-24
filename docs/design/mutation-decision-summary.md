# Deterministic mutation decision summaries

`generate_mutation_decision_summary` joins one canonical mutation candidate with its
calibrated score, expected analytical delta, local attribution, and quantization context.
It rejects mismatched candidate identities and never invokes an LLM or performs a new
prediction.

Both JSON and human text are projections of the same immutable summary. They include the
candidate/component/mutation identities, calibrated and raw safe probability, utility,
uncertainty, named outcome predictions, analytical parameter/FLOP/memory/storage deltas,
top local evidence, and quantization provenance. Evidence is ranked by absolute
contribution with the feature name as a stable tie break.

Missing analytical deltas are emitted as one entirely unknown delta group; individual
zeroes are never substituted. Missing quantization context is printed as `unknown`, and
an unavailable attribution carries its actual reason with no fabricated evidence.
Numeric human rendering uses a stable 12-significant-digit representation, while canonical
JSON sorts keys and uses no incidental whitespace.
