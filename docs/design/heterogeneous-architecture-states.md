# Heterogeneous architecture states

Heterogeneous architecture v1 represents every layer's effective attention
width, MLP width, low-rank replacements, sparsity entries, and quantization
codec in one immutable, complete state. Requested and effective choices are
retained together; the compiler does not silently normalize an unsupported
choice.

`HeterogeneousCapabilities` is an explicit runtime and artifact-format
contract. Compilation rejects format/runtime mismatches, unsupported codecs,
layer limits, low-rank or sparsity requests, and mixed quantization when the
selected runtime cannot represent them. Accepted states receive a
content-addressed identity and analytic parameter/storage estimates that are
the sum of their per-layer estimates.

`materialize_heterogeneous_state` writes only to a staging file. It verifies the
source digest before and during the operation, reloads the staged artifact,
checks the effective per-layer choices and reconciled costs, runs the supplied
benchmark callback, and publishes with an atomic replace only after all gates
pass. A failed or unknown reload leaves no output artifact and preserves the
source. Benchmark values, repetitions, runtime identity, and every explicit
unsupported/failed/unknown reason remain in the evidence record.

The checked-in fixture uses a tiny callback-backed artifact so the contract can
exercise materialize/reload/benchmark and rollback paths without claiming that
the callback is a model exporter. Real Hugging Face and GGUF exporters must
provide their own capability evidence and reload implementation before a state
can be treated as a physical deployment result.
