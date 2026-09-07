# Joint surgery, repair, and quantization search

`modelsurgeon.search.joint_repair` defines the bounded v1 joint-search boundary.
Each candidate retains its complete parent state, source artifact, model family,
repair method, repair budget, quantization codec, and quantization order.  The
budget is hard: no-repair, repair, and quantization arms are rejected when any
declared resource limit is exceeded.

Recoverability and repair-cost predictors may populate `JointRepairPrediction`.
Predictions are conservative acquisition evidence only.  `JointRepairStudy`
can expose a predicted frontier for measurement, but `decide()` promotes only
an accepted `JointRepairOutcome.MEASURED` artifact with held-out groups,
reconciled costs, and complete lineage.  A larger initial damage is therefore
eligible only when its measured final quality/deployment frontier is better;
predicted benefit cannot publish an unmeasured artifact.

Execution failures, unsupported compatibility, unknown outcomes, and negative
held-out results remain explicit records.  Negative results must record a
rollback and reason, while terminal outcomes cannot carry partial metrics.
Candidate and study identities are content-addressed and all tuples are
canonical for deterministic resume and audit.
