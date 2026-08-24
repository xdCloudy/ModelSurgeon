# Active-learning diversity selection

Issue #111 uses seeded farthest-first traversal over normalized numeric features,
categorical Hamming distance, and topology-set Jaccard distance. Explicit non-negative
weights combine those spaces, and seeded candidate hashes provide deterministic initial
choice and tie-breaking.

The selector never constructs a candidate-by-candidate distance matrix. It retains one
minimum-distance scalar per input candidate plus the bounded selected-index set, yielding
O(pool size + selection size) working memory. Inputs are capped at 100,000 candidates and
selections at 4,096. The report exposes `working_distance_values` so the memory shape is
observable and testable.

Feature widths, canonical IDs, finite numeric values, categorical/topology values, counts,
weights, and bounds fail closed. Zero requested selections return a deterministic empty
report for downstream acquisition edge cases.
