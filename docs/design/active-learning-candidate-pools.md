# Active-learning candidate pools

Issue #108 publishes canonical graph-valid candidates without mutating the model. The
existing graph enumerator validates requests and coupling closure before pool publication,
and both enumeration and publication enforce the milestone ceiling of 100,000 candidates.

Pools are canonical JSONL. Every record contains the complete mutation candidate, cheap
mutation-free features (scope, node kind, layer, coupling/constraint counts, and request
parameter count), plus run, graph, model, tool, seed, and enumerator provenance.

Publication is resumable at exact record boundaries. Each bounded invocation may append a
limited number of new records, fsyncs data, then atomically replaces a manifest containing
the committed count and prefix SHA-256. Resume streams and verifies the entire committed
prefix against both the digest and deterministic candidate IDs before appending. Missing,
changed, corrupt, or uncheckpointed state fails closed.
