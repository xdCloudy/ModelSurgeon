# Active-learning sample-efficiency study

Issue #116 requires active, seeded-random, and utility-only strategies to use identical
experiment-count grids and seed sets. Each curve records predictive performance versus
experiments and computes normalized trapezoidal area under the learning curve (AULC).

Strategy summaries report mean AULC and seeded bootstrap confidence intervals across runs.
If active mean AULC does not exceed the strongest equal-budget baseline, the versioned study
contains an explicit negative-result statement. A dependency-free SVG renderer plots mean
performance against experiments and carries that statement into the visual artifact.

Missing strategies, unequal budgets, unequal seed sets, non-finite measurements, or invalid
confidence/bootstrap bounds fail closed.

The multi-model Q7 replay compares uncertainty-only, uncertainty plus feature-space diversity,
and paired seeded-random acquisition under the same initial labeled set, budgets, and seeds.
The hybrid policy fixes its acquisition split at 75% Bernoulli predictive uncertainty and
25% diversity for the preregistered run. Diversity uses the current source-only trained
preprocessor output plus pre-mutation categorical and mutation-topology metadata; all
currently labeled examples seed the k-center set. Validation and test examples never enter
the acquisition pool. Each budget point records the newly selected sorted, hashed candidate
IDs so the chosen sets can be reconstructed without exposing raw example identifiers. The Q7 v2 runner records the
policy split and evaluates AULC, paired target-reaching experiments, and measured candidate
evaluation time against both uncertainty-only and random controls. The v1 result is retained;
the three-model v2 replay and its negative generalization result are documented in
`docs/research/v0.8-q7-feature-diversity-evidence.md`. This remains an offline label-replay
study, not evidence that the acquisition policy improves physical model compression or
downstream task quality.
