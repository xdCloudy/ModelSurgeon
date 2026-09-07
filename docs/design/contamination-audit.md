# Corpus contamination and ancestry audit

`modelsurgeon.datasets.contamination` is the machine-readable audit boundary
for benchmark, calibration, repair, adaptation, validation, and generated
output corpora. It is separate from the existing mutation-example audit so
the same exact and near-duplicate decision can be applied to every published
bundle.

Sources are pinned by corpus ID, revision, split, license, kind, availability,
and declared ancestry. Items retain only safe derived identity in reports:
item ID, corpus ID, exact text digest, byte count, generated flag, and ancestry
IDs. Raw text is used only during the bounded audit and is not serialized.

The audit uses exact SHA-256 matches and deterministic token-shingle Jaccard
matching under explicit item and pair budgets. Exact and ancestry overlaps
block affected claims. Near overlaps produce a move-and-new-protocol action.
Unavailable or unauditable licenses produce an unknown limitation, never a
clean result. Resource exhaustion is retained as an unknown finding.

Every report has a deterministic ID. A remediation protocol ID hashes the
original protocol identity and the exact finding IDs moved, so corrected
manifests cannot silently reuse the old claim identity.

