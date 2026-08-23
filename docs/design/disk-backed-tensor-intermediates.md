# Disk-backed tensor intermediates

Large decoded or transformed tensors can be staged as an append-only directory of
fixed-size binary chunks. The immutable specification records canonical tensor
identity, shape, dtype, item size, chunk ceiling, and the SHA-256 identity of the input
plan. Retained memory is bounded by one configured chunk; no reconstruction API returns
a complete tensor.

Each chunk is written and fsynced under a private name, atomically promoted, and then
recorded in a canonical manifest. The manifest contains sequential byte counts and
SHA-256 digests and is itself atomically replaced after every committed chunk. Resume
requires the exact specification and verifies every committed file before exposing it.

Interrupted manifest temporaries and uncommitted chunk files are reported as stale.
Callers may explicitly recover those two cases by discarding only uncommitted files;
changed plans, missing chunks, and checksum failures always fail closed. Explicit
cleanup removes only the exact intermediate directory, preserving neighboring run
artifacts.
