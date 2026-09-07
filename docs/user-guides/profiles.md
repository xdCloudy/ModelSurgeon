# Hardware and quality profiles

The optimize planner uses named envelopes so examples remain reproducible and
resource claims remain explicit. These are planning profiles, not a probe of
the current machine.

| Profile | Device | CPU threads | Host RAM | VRAM | Disk | Intended use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cpu-small` | CPU | 2 | 8 GiB | none | 20 GiB | clean CI, laptops, tiny fixtures |
| `cpu-large` | CPU | 8 | 32 GiB | none | 100 GiB | larger CPU-only runs |
| `gpu-12gb` | `cuda:0` | 8 | 32 GiB | 12 GiB | 100 GiB | low-VRAM consumer GPU |
| `gpu-24gb` | `cuda:0` | 16 | 64 GiB | 24 GiB | 200 GiB | larger single-GPU runs |

Quality profiles are `fast` (95% retention target, one repetition), `balanced`
(98%, three repetitions), and `quality` (99.5%, five repetitions). Presets
scale evaluation and repair budgets; they never weaken configured hard
constraints.

Use `modelsurgeon optimize --help` and the public
`available_hardware_profiles()` / `available_quality_profiles()` functions to
discover the versioned names. If a profile is too small, the planner returns a
failed or unknown plan with an actionable uncertainty; it does not silently
raise the resource envelope.
