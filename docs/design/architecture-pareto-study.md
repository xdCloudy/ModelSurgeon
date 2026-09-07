# Multi-axis architecture Pareto study

The v1.4 study matrix compares greedy, Pareto beam, evolutionary, surrogate,
one-axis, and competitor controls over the same complete state, objective,
evaluation, family, hardware, budget, and seed dimensions. The matrix is
closed: unsupported, failed, unknown, and explicitly negative cells are
retained rather than dropped from the denominator.

Measured frontier points carry a deployable state ID, output artifact digest,
family and hardware identity, held-out quality interval, deployment-cost
interval, repetitions, and provenance. Conservative frontier computation
requires worst-case interval separation, so overlapping measurements do not
become a superiority claim. A bounded quality/cost hypervolume is reported per
cell when complete evidence exists.

Method comparisons are paired by family, hardware profile, budget, and seed.
The study records the mean delta and a deterministic seed bootstrap interval;
positive and negative claims require the interval to exclude zero. Incomplete
physical pairs are explicitly unsupported or inconclusive. The default matrix
is a preregistered unavailable-artifact matrix and therefore makes no
improvement claim. Synthetic contract fixtures demonstrate the comparison and
negative-result paths but are not physical deployment evidence.
