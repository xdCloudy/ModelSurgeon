# v1.1 competitor Pareto study

The v1.1 study binds two model families and two sizes to the same held-out
corpus, three seeds, evaluator identity, hardware profile, and resource
budgets. It retains the integrated methods, unstructured and structured
baselines, quantization arms, and magnitude/random controls.

Every point must identify a reloadable artifact by SHA-256 and carry quality,
artifact size, RAM, VRAM, prompt speed, decode speed, and optimization cost.
Measured metrics require paired confidence intervals from at least 100
bootstrap repetitions. A cell without a complete artifact bundle is retained
as unsupported, failed, or unknown.

The conservative frontier is empty until deployable measured points exist.
Consequently the published v1 evidence makes no superiority claim. This is a
valid negative/inconclusive result, not evidence that any method is superior or
inferior. Independent reruns should replace only the retained cell evidence
through the benchmark CLI and must preserve the study identity and artifact
digests.
