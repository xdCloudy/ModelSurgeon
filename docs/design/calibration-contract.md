# Calibration dataset and sample identity contract

Calibration schema v1 pins dataset name, immutable revision, split, license, trust
classification and rationale, preprocessing identity/configuration digest, and
tokenizer identity/configuration digest. Selected samples retain both native IDs and
content SHA-256 digests, preventing silent content changes behind stable row IDs.

`sha256-rank-v1` ranks each candidate by a SHA-256 digest over the complete contract,
seed, and sample ID. Selection is therefore independent of source enumeration order
and identical for the same contract, candidates, count, and seed. Duplicate IDs,
invalid digests, unsupported algorithms, and requests larger than the candidate pool
fail before a selection record is emitted.
