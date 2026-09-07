# Bounded teacher-target cache and repair-example selection

`modelsurgeon.surgery.teacher_cache` provides the cache and selection boundary used by bounded
repair and distillation experiments. It is deliberately format-neutral: model execution produces
the numeric targets, while this contract controls identity, persistence, selection, and evidence.

## Cache identity and publication

`TeacherCacheKey` includes the teacher model ID/revision, tokenizer digest, repair-example
revision, target kind (logits or features), target schema, and encoding (FP32 or INT8). A cache
directory is content-addressed by that complete identity; loading also validates the embedded key,
manifest digest, every chunk checksum, chunk ordering, token totals, and configured byte/value
limits. An incompatible teacher, tokenizer, example revision, target schema, or encoding therefore
cannot be reused accidentally.

Targets are written as bounded immutable chunks. `TeacherCacheStore.write_chunk()` is safe to
retry with identical data and rejects a conflicting rewrite. `publish()` only creates a manifest
after all consecutive chunks are present, so an interrupted run can resume without replacing
completed chunks. `access()` reports hit/miss, chunk, read-byte, and write-byte telemetry.

## Selection

`RepairSelectionCandidate` carries a content digest, domain, token cost, utility, diversity,
predicted recoverability, partition, and metadata. `select_repair_examples()` supports stable
random, domain-balanced, diversity, and predicted-recoverability ordering under example and token
budgets. Validation and test partitions are always excluded by default and recorded as explicit
held-out exclusions. The selection ID hashes the algorithm, configuration, selected candidates,
and exclusions, making reruns and cache identities auditable.

## Protocol boundaries

Only training candidates are eligible by default. Held-out evaluation examples must remain outside
both repair selection and teacher-target publication. Remote proprietary teachers, unbounded raw
activation storage, and silent cache eviction are outside this contract; callers must retain
explicit unsupported, failed, or incomplete evidence in the surrounding repair dataset.
