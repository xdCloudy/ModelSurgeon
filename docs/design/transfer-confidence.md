# Transfer confidence and abstention

Transfer inference is gated by compatibility and capability evidence before
learned confidence is considered. `decide_transfer_confidence` returns an
explicit accepted, abstained, unsupported, or unknown decision with a stable
identity and machine-readable reasons.

Unknown or unsupported architecture, mutation, and compatibility states fail
closed. They never become accepted because an ensemble is confident. Supported
but weakly covered or out-of-distribution inputs abstain when support
coverage, feature distance, uncertainty, or calibration drift exceeds the
versioned policy thresholds.

`build_selective_risk_curve` retains coverage, violation rate, selective risk,
and target evaluation cost by uncertainty threshold. Observations preserve
false-confidence cases and unsupported/unknown outcomes, enabling grouped
held-out calibration evidence and paired comparisons against uncalibrated
uncertainty without changing the decision policy after seeing test outcomes.
