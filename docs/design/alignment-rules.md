# Hardware-specific alignment rules

`modelsurgeon.surgery.alignment_rules` separates hard mutation legality from empirical
preference. `AlignmentConstraint` intersects an existing graph multiple with a codec
multiple; a request is legal only when it satisfies both. No measured performance result
can weaken that constraint.

`AlignmentPreference` is optional evidence for a legal granularity. A measured preference
must link training and held-out validation results from one content-addressed
microbenchmark partition and one hardware profile, and records its false-preference rate.
Unsupported, failed, conflicting, or unseen cells produce an explicit `unknown` preference
instead of a guessed rule.

Rule lookup returns `illegal` for a hard alignment violation, `legal` with the preferred
granularity when evidence supports it, and `unknown` when only legality is known or no rule
covers the requested axis/format/codec. This boundary is advisory to mutation planning;
existing graph validation and codec safety checks remain authoritative.
