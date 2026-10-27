# Active-learning diversity selection

Issue #111 uses seeded farthest-first traversal over numeric features normalized
independently to the min/max range across the candidate pool and any initial centers,
categorical Hamming distance, and topology-set Jaccard distance. Normalization is fit
without outcomes and is invariant to numeric units; constant columns contribute zero.
Explicit non-negative weights combine these spaces, and seeded candidate hashes provide
deterministic initial choice and tie-breaking. When acquisition has already selected
high-value or uncertain candidates, it passes them as fixed centers; the first diverse
candidate is then the farthest remaining point from that selected set. Initial centers
are bounded by the configured selection limit. Reports record the normalization revision
and center count because both affect distances and selection order.

The selector never constructs a candidate-by-candidate distance matrix. It retains one
minimum-distance scalar per input candidate plus the bounded selected-index set, yielding
O(pool size + selection size) working memory. Inputs are capped at 100,000 candidates and
selections at 4,096. The report exposes `working_distance_values` so the memory shape is
observable and testable.

Feature widths, canonical IDs, finite numeric values, categorical/topology values, counts,
weights, and bounds fail closed. Zero requested selections return a deterministic empty
report for downstream acquisition edge cases.

## Upstream algorithm review

The greedy k-center acquisition principle was compared with Google's archived
[`kcenter_greedy.py`](https://github.com/google/active-learning/blob/efedd8f1c45421ee13af2b9ff593ad31f3835942/sampling_methods/kcenter_greedy.py)
at revision `efedd8f1c45421ee13af2b9ff593ad31f3835942` (Apache-2.0; the repository was
archived in April 2026). It accepts previously selected centers and updates nearest-center
distances greedily, which motivated the fixed-center support used by acquisition. Its
implementation computes pairwise Euclidean distances and catches every exception around
feature transformation. ModelSurgeon does not copy its code: the local selector keeps
linear working memory, uses mixed numeric/categorical/topology distances, and fails closed
on malformed feature schemas.
