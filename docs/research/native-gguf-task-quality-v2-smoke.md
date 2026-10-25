# Native GGUF task-quality v2 CPU smoke

This paired smoke records the first physical run of the native GGUF exact-match
evaluator through `NativeGGUFOptimizeRuntime._task_quality`. It found and
verified a CLI-output parsing defect fixed in the same change. It is a plumbing
and reproducibility check, not evidence that pruning preserves general task
quality or improves the quality-compression frontier.

## Method and provenance

- Repository source and four-example test fixture: `main` revision
  `72cf8017d48669157dd6b51b7c2ad0a1917320a1`.
- Dataset SHA-256:
  `768001546f0a18cd548cad36e173db0ba2443581b4829632b4effc30a4ab3280`.
- llama.cpp: `0.7.0-dev`, commit `c7fc50941`, CPU-only execution, 12 threads,
  context 512, 128 generated tokens, temperature 0, top-p 1, reasoning off.
- Source GGUF SHA-256:
  `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`.
- Candidate GGUF SHA-256:
  `06969df1ce0505ad2f7d506497c9f5f9731c3a3887d07351ccafbd3edd7d3fe3`.
- The source and candidate are a Qwen3.5-derived MoE model pair. The candidate
  was already physically produced by a separate experiment; this measurement
  calls the task-quality evaluator directly and does not establish that the
  native optimizer supports this MoE architecture.
- llama.cpp's `--output` transcript option was introduced in
  [upstream commit `3de7dd4`](https://github.com/ggml-org/llama.cpp/commit/3de7dd4).
  No upstream implementation code was copied.

## Results

| Artifact | Bytes | Exact matches | Accuracy | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Source | 21,713,462,496 | 1 / 4 | 25% | 65.98 s |
| Candidate | 21,399,217,408 | 1 / 4 | 25% | 68.82 s |

The candidate is 314,245,088 bytes (1.4472%) smaller. All four per-example
completion digests match between source and candidate, but only one completion
matches the fixture's exact reference. The observed runtimes are single-run
measurements and do not establish a throughput change.

## Defect and correction

The pinned CLI writes startup and loading messages to stdout, so treating
captured stdout as the model completion made exact-match evaluation invalid.
The runtime now asks llama.cpp to write its single-turn transcript to a bounded
temporary file, disables reasoning, extracts only the assistant section, and
fails closed for missing, oversized, or unrecognized transcripts. The evaluator
version advances to `code_exact_match_v2` and removes one complete enclosing
Markdown code fence after whitespace and line-ending normalization.

The 25% score on both artifacts shows that this four-example fixture is too
small and formatting-sensitive to support a broad quality-retention claim. No
candidate gate qualification, quality improvement, or speedup is claimed. A
larger held-out task suite and repeated paired runs are required before using
this fixture for structural-pruning decisions.
