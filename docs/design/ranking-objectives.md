# Ranking objective study

`modelsurgeon.surgeon.ranking_objectives` defines a deterministic comparison
boundary for pointwise, pairwise, listwise, and sequence ranking objectives.
The study is deliberately an evaluator rather than an automatic objective
switch: every candidate objective receives the same held-out lists. A caller
may supply a declared equal training-step budget and measured inference costs;
the evaluator does not invent these values or independently verify their
provenance.

## Evidence contract

Each `RankingCandidateList` identifies a parent architecture state, model
revision, split, and lineage group. A candidate retains its pointwise predicted
score, optional predictions from each ranking-objective model, measured
utility, violation probability, violation outcome, and cumulative utility when
the outcome is complete. When objective-specific predictions are supplied,
they must cover all four methods. The legacy scalar prediction is used for
every method when no objective-specific predictions are supplied, preserving
the older input contract without implying a difference between methods.

Measured utility and cumulative utility are outcomes only. They are used to
score a ranking after it has been produced; they must never choose the ranking.
In particular, sequence ranking consumes the sequence model's predicted score
and cannot use measured cumulative utility as an oracle. The evaluator compares
supplied model predictions; it does not train pointwise, pairwise, listwise, or
sequence models itself.

Unsupported, failed, and unknown candidates remain explicit evidence, but
cannot contribute a partial measured outcome to a ranking or metric.

Validation and test lists are required; training lists are rejected. This
keeps the study boundary explicit and prevents a caller from presenting
training evidence as held-out objective evidence. Every objective is evaluated
on the same list identities, with deterministic tie breaking by candidate ID.
The schema and protocol revisions identify the change from outcome-based
sequence ranking to prediction-only ranking.

## Metrics and cost controls

The result reports NDCG, regret, precision@k, violation rate@k, Brier
calibration error, and cumulative frontier utility. Confidence intervals use a
seeded cluster bootstrap over `lineage_group_id`; each lineage group receives
equal weight regardless of how many lists it contains. This keeps related
lists from being counted as independent replicates. Training-step and inference-cost
fields are `null` unless supplied by the caller; resource costs are not
estimated by this evaluator. These fields document the caller's evidence, but
the evaluator cannot verify training logs or benchmark methodology. Rankings
and the `recommended_objective` policy remain quality/regret based and are not
a measured deployment-cost Pareto comparison.

## Recommendation policy

The pointwise result is the baseline. A more complex objective is recommended
only when its regret interval is entirely below the pointwise regret interval.
Otherwise `recommended_objective` is `None` and the result carries a negative
or neutral recommendation reason. This preserves a useful study result when
the evidence is censored, tied, or insufficient for a superiority claim.
