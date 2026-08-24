# Active-learning acquisition policy

Issue #112 allocates an exact bounded budget among high expected value (utility multiplied
by calibrated safe probability), uncertainty, and diversity. Fractions must be finite,
non-negative, and sum to one. Largest-remainder apportionment converts them to deterministic
integer quotas.

Each phase takes its highest-scoring unselected candidates with seed-hash tie-breaking.
Overlap between phases is skipped; any resulting shortfall is filled by a deterministic
combined score. Every selected candidate records its phase reason, policy score, rank, and
propensity. Because this policy is deterministic conditional on inputs and seed, selected
propensity is truthfully recorded as 1.0 rather than presenting score normalization as a
sampling probability.

A zero budget returns an empty report. An oversubscribed budget is capped to the available
candidate count and selects every candidate exactly once. Both cases preserve requested and
effective budgets and are deterministic.
