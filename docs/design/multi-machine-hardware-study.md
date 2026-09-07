# Multi-machine hardware-aware selection study

The evaluation contract compares one frozen candidate set on at least three
materially different profiles, including a CPU-only or low-VRAM host. Every
profile carries runtime, hardware, environment, source-artifact, corpus, and
optimization-budget identities. Each candidate/objective/arm cell requires at
least three repeated runs.

The hardware-aware arm selects independently per profile. The hardware-blind
baseline selects from pooled blind evidence and is then evaluated on each
profile. Selection comparisons retain host-level means and deterministic 95%
normal-approximation intervals. A hardware-specific choice is reported only
when measured evidence supports it; unsupported, failed, and unknown cells are
retained and prevent a positive claim.

The final claim is positive only when every measured profile/objective
comparison beats or ties the blind baseline and at least one choice differs.
An all-measured tie with no changed choice is a negative result, while missing
or unsupported cells are inconclusive or unsupported. This prevents a missing
host or failed runtime from being silently interpreted as a successful
hardware-aware optimization.

The checked-in fixtures use deterministic synthetic observations to exercise
CPU-only, low-VRAM, and full-GPU profiles, profile-specific choices,
repetition validation, and an unsupported profile cell. They are protocol
fixtures, not claims about a particular production model or machine.
