# Multi-objective reward schema v1

Soft optimization objectives are a non-empty configurable set drawn from quality,
parameter count, latency, memory, and disk. Each term declares maximize/minimize
direction, positive weight, and one normalization: identity, ratio to an explicit
non-zero baseline, or min-max with ordered finite bounds. The scalar reward is the
weighted mean of signed normalized values; its record retains every contribution.

Terms are canonically sorted by metric and duplicate metrics are rejected. Missing
observations or required baselines fail scoring rather than receiving invented
defaults. Any subset and weighting is valid, so search does not privilege a
hard-coded objective. The objective-set ID hashes the complete canonical definition.

Resolved settings serialize `objective.terms`, including weights, directions,
normalization, and bounds. Consequently a change to any reward choice changes the
canonical settings JSON used in run identity. The older `objective.optimize` list is
translated to baseline-ratio terms when explicit terms are absent.
