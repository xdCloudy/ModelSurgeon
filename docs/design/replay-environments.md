# Deterministic replay environments

`modelsurgeon.experiments.replay_environment` is the typed preflight boundary
for replaying signed evidence.  A recipe retains the exact command, sorted
seeds and input digests, schema versions, metric tolerances, and environment
lock identity.  Native, Docker, and exploratory Nix environments are separate
identities; Docker must declare its GPU passthrough boundary.

Replay compares the locked environment with an observed host before execution.
Definition, toolchain, architecture, hardware class, passthrough, and missing
capabilities are blocking mismatches, so an expensive run cannot start on an
incompatible machine.  Cursors advance only through the next declared step,
making interruption and resume deterministic.

Schema migration copies form an ordered chain.  Each copy retains the original
bundle digest and the previous output digest, so migrations never overwrite or
silently replace the source.  Metric replay compares exact keys with declared
absolute/relative tolerances and reports missing or undeclared metrics as
non-passing.
