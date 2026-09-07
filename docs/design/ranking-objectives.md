# Ranking objective study

`modelsurgeon.surgeon.ranking_objectives` defines a deterministic comparison
boundary for pointwise, pairwise, listwise, and sequence ranking objectives.
The study is deliberately an evaluator rather than an automatic objective
switch: every candidate objective receives the same held-out lists, training
step budget, and reported inference-cost accounting.

## Evidence contract

Each `RankingCandidateList` identifies a parent architecture state, model
revision, split, and lineage group. A candidate retains its predicted score,
measured utility, violation probability, violation outcome, and cumulative
utility when the outcome is complete. Unsupported, failed, and unknown
candidates remain explicit evidence, but cannot contribute a partial measured
outcome to a ranking or metric.

Validation and test lists are required; training lists are rejected. This
keeps the study boundary explicit and prevents a caller from presenting
training evidence as held-out objective evidence. Every objective is evaluated
on the same list identities, with deterministic tie breaking by candidate ID.

## Metrics and cost controls

The result reports NDCG, regret, precision@k, violation rate@k, Brier
calibration error, and cumulative frontier utility. Bootstrap intervals use a
seeded list-level resampling procedure. The configuration also records one
equal training-step budget and an inference-cost value for each objective, so
quality comparisons cannot silently omit resource tradeoffs.

## Recommendation policy

The pointwise result is the baseline. A more complex objective is recommended
only when its regret interval is entirely below the pointwise regret interval.
Otherwise `recommended_objective` is `None` and the result carries a negative
or neutral recommendation reason. This preserves a useful study result when
the evidence is censored, tied, or insufficient for a superiority claim.
