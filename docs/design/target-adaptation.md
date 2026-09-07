# Zero-shot and few-shot target adaptation

Target adaptation starts from an immutable source surgeon bundle and an
explicit compatibility decision. `plan_target_adaptation` creates resumable
requests for zero-shot, shift-correction, frozen-head, linear-adapter, and
full-bounded modes across target-example budgets 0/8/16/32/64/128 and five
selection seeds.

Zero-shot requests contain no target support IDs and no target preprocessing
statistics. Few-shot requests select only labelled support-pool examples with
a stable seed-ranked order. Target test IDs are recorded separately and never
carry outcomes in the request manifest, so adaptation cannot consume test
labels or test-derived statistics.

Each result retains the parent bundle, immutable child digest when one was
produced, learning-curve points, training/evaluation costs, provenance, and an
explicit committed, rolled-back, or interrupted transition. Negative,
unsupported, failed, and unknown outcomes remain visible. A negative result
is evidence; it is not promoted into a success claim. The record API is
append-only at the request level and supports safe resume without replacing a
previous outcome.
