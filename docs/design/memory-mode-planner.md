# Memory-mode planner

Every scalable operation supplies peak RAM, VRAM, and scratch-disk estimates for
full-model, tensor-at-a-time, and streaming execution. Planning compares those
estimates with a point-in-time capacity snapshot and optional user hard ceilings.
The effective capacity for each resource is the smaller of availability and its user
ceiling.

Automatic mode considers `full`, `tensor`, then `streaming`, selecting the first mode
whose three resource estimates fit. Rejected modes and the resources they exceeded are
retained in the plan. An explicitly requested mode is either honored or rejected; it
never silently falls back. If no automatic mode fits, planning fails before model data
is loaded or output construction begins.

The selected plan exposes the exact peak estimate and effective capacity used for the
decision. These values belong in experiment provenance and allow execution code to
assert the same hard ceilings at allocation boundaries.
