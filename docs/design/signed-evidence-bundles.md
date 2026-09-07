# Signed immutable evidence bundles

`modelsurgeon.experiments.signed_evidence` defines an offline-verifiable
bundle index for optimization evidence.  A bundle hashes every regular file
as a canonical POSIX-path member, derives a deterministic SHA-256 Merkle root,
retains source/config/data/hardware lineage, and supports bounded streaming
verification without loading the tree into memory.

Detached HMAC-SHA256 attestations are stored with public key metadata only;
key material is supplied externally to signing and verification.  Key records
retain active, rotated, or revoked status and predecessor/successor links.
Changing any member byte, path, size, Merkle root, lineage field, key metadata,
or signature causes verification failure.  Missing or unavailable external
blobs produce an incomplete verification result rather than a stronger claim.

Redacted paths and external references are explicit index entries and lower the
bundle claim to `bounded_reproducibility`.  They cannot silently appear as
missing members or be treated as reproducible local bytes.  Index rendering
contains no signing secret, and the verifier enforces member and byte limits.
