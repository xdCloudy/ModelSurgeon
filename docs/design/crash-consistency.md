# Crash consistency and bounded fault matrix

`modelsurgeon.resilience` records the recovery contract for source artifacts,
accepted artifacts, owned staging paths, leases, manifests, checksums, and
child processes. The fault matrix covers source/staging/checksum/fsync/rename,
disk-full, OOM, cancellation, process death, timeout, and deadlock boundaries.
Every point has a deterministic seed, a documented recovery action, and a
machine-readable evidence record.

`bounded_subprocess` accepts only an argument array with `shell=False`. On a
timeout it terminates the process tree, waits for a bounded grace period, and
retains clipped combined logs. A timed-out child is recorded as unknown rather
than passing silently. `verify_recovery_invariants` rejects removal of any
path outside the declared staging prefix and checks source and accepted
content identities before recovery is considered safe.

The contract does not claim to emulate destructive power loss. CI uses
controlled fault injection and temporary fixtures; scheduled Windows/Linux
runs should add real filesystem and process evidence.

