# Active-learning acquisition policy

Issue #112 allocates an exact bounded budget among high expected value (utility multiplied
by calibrated safe probability), uncertainty, and diversity. Fractions must be finite,
non-negative, and sum to one. Largest-remainder apportionment converts them to deterministic
integer quotas.

High-value and uncertainty phases take their highest-scoring unselected candidates with
seed-hash tie-breaking. The diversity phase retains scalar-score ranking by default. Callers
can instead provide a feature/topology row for every candidate; those rows must align exactly
with the acquisition candidate IDs. In feature mode, seeded farthest-first selection runs on
the candidates remaining after the first two phases, and its normalization/report are embedded
in the acquisition record. This makes the diversity quota measure feature-space coverage
rather than rank a caller-supplied scalar. Overlap between phases is skipped; any remaining
shortfall is filled by the deterministic combined score. Every acquired candidate records its
phase reason, policy score, rank, and propensity. Because this policy is deterministic
conditional on inputs and seed, selected propensity is truthfully recorded as 1.0 rather than
presenting score normalization as a sampling probability.

Feature mode is opt-in to preserve the existing acquisition contract. Its report records the
candidate pool left for the diversity phase and the nested diversity selector protocol, so
replays can distinguish feature-based decisions from the legacy scalar policy.

A zero budget returns an empty report. An oversubscribed budget is capped to the available
candidate count and selects every candidate exactly once. Both cases preserve requested and
effective budgets and are deterministic.
