# Consumer repair effectiveness and economics

`modelsurgeon.evaluation.consumer_repair_study` defines the v1.7 matched-cell
study for bounded repair on consumer hardware.  Its default protocol spans two
model families, two size classes, three damage levels, three repair budgets,
three seeds, two hardware profiles (`cpu_offload` and `low_vram_12gb`), and
no-repair, fixed-repair, oracle-budget, and learned-joint arms.

Every cell retains source/teacher/final artifact lineage, data and hardware
identities, sorted provenance, repeated-run count, held-out quality and
deployment-cost intervals, and repair-efficiency metrics per token, second,
and joule.  Unsupported, failed, unknown, and negative-result cells are
terminal records without fabricated numeric values.

Recommendations compare matched repair arms with the no-repair baseline using
held-out quality and physical deployment cost.  They classify a region as
beneficial, harmful, unnecessary, infeasible, or unknown.  Training loss alone
cannot produce a supported recommendation, and the default matrix makes no
physical improvement claim until licensed representative artifacts are run.
