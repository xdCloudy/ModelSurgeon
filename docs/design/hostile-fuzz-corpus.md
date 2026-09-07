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
unknown behavior. Subprocess runners disable the shell, enforce a timeout,
bound captured output and sandbox bytes, and record non-zero exits as rejected
evidence. A timeout or excessive output is a retained result, never a silent
pass.

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

