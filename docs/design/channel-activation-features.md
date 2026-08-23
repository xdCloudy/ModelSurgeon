# Channel and token-dependent activation features

The collector receives channel identities from the physical component graph and emits
one statistics record for each identity in exactly that order. Each channel uses the
fixed-memory streaming statistics accumulator, so sample count and sequence length do
not increase retained memory.

Token-dependent context is intentionally finite. Callers configure a positive number
of position buckets and an optional closed set of token-class labels. Positions beyond
the configured range fold into the last position bucket. Masked tokens update neither
channel nor token buckets, and buckets without observations are reported as `None`.

The retained accumulator count is therefore:

`graph channels + position buckets + configured token classes`

This makes the memory bound explicit and testable while preserving canonical graph
identity on every per-channel output.
