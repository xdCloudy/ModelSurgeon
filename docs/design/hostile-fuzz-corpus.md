# Hostile-input corpus and bounded parser harness

ModelSurgeon keeps a small, deterministic corpus for untrusted model and
execution boundaries. The corpus is intentionally made from tiny malformed
bytes, not model weights, so it is license-safe and cheap to run in ordinary
CI.

## Contract

`modelsurgeon.validation` publishes schema `1.0` and protocol revision
`hostile-fuzz-v1`. Cases cover HF checkpoint metadata, safetensors, GGUF,
JSON configuration and manifests, fixture paths, and subprocess logs. A case
ID hashes its kind, seed, expected outcome, and exact payload. Its manifest
retains the payload digest, limits, source boundary, minimized flag, and
regression flag without publishing the payload itself.

The default corpus has seven cases, a 512 KiB total byte ceiling, and a
declared 0.25 CPU-hour campaign budget. Callers may supply tighter limits.
Parser runners retain rejected, unsupported, failed, resource-limited, and
unknown behavior. Subprocess runners disable the shell, enforce a wall timeout,
and trigger child termination when combined stdout and stderr exceed the capture
budget. They record non-zero exits as rejected evidence. The runner polls disk
usage under its temporary directory while the child runs and terminates the
child after an observed overage; it checks the directory again after the run.
Polling can overshoot the threshold and miss brief spikes between scans. A
resource-limited result records the largest sampled byte count. A caller-supplied
working directory is outside that scan, so this is not a disk quota.
`max_memory_bytes` is retained in corpus metadata but is not enforced by the
runner. The runner does not isolate a child from filesystem access available to
the current user. A timeout or excessive output is a retained result, never a
silent pass.

`resolve_scoped_path` rejects absolute paths, traversal, symlink components,
and resolved paths outside the fixture root. It is the path target to use for
manifest, checkpoint, and subprocess fixture inputs.

## Reproduction

```text
python -m pytest tests/test_validation.py
python -m ruff check src tests
python -m mypy src/modelsurgeon
```

The test suite verifies deterministic IDs and manifests, all boundary kinds,
negative parser results, traversal and symlink rejection, and subprocess
timeout behavior. No crash, timeout, or path escape was observed by this
minimal corpus; these zero-finding results remain represented as a report,
rather than being inferred from an empty log.

