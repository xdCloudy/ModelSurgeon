# Constrained search CLI v1

`modelsurgeon search CONFIG --state search.sqlite3` reserves one deterministic
greedy, beam, or uncertainty-aware policy decision and atomically writes generation
zero of the resume log. Repeating the command with `--resume` loads the latest
checksum-verified snapshot, rejects a changed policy/lineage/frontier, skips already
reserved candidates, and appends the next generation.

The strict JSON config contains:

- immutable source checkpoint/state IDs plus sorted accepted-lineage and frontier
  checkpoint IDs;
- hard quality-retention, perplexity-delta, latency-gain, RAM, VRAM, and disk
  constraints;
- weighted quality, perplexity, latency, memory, parameter-count, and disk-size
  objectives with explicit direction and normalization;
- policy kind, evaluation budget, beam width, exploration weight, and seed; and
- an initial predicted pool whose canonical candidate/state/parent IDs carry every
  objective and constraint observation.

`--dry-run` requires no state path and performs no write. Its canonical JSON output
shows the resolved constraints, objectives, policy budget, complete initial pool,
planned selection reasons, accepted checkpoint lineage, and frontier. A normal or
resumed run additionally reports the persisted generation and all pending
evaluations. `--output` writes the same canonical result with exclusive-create
semantics.

The command reserves candidates for an evaluator; it does not fabricate measured
evidence or promote a predicted candidate. Accepted checkpoint IDs must already
come from the evaluated keep/rollback lineage policy. This separation prevents
predictions from silently becoming search roots.

```text
modelsurgeon search search.json --dry-run
modelsurgeon search search.json --state search.sqlite3
modelsurgeon search search.json --state search.sqlite3 --resume
```
