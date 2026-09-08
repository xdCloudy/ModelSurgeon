# Migration and compatibility boundary

This document defines the v3.0 migration guarantee for issue #489. Migration
is an explicit, offline data operation. It does not execute a model, call a
text LLM, change hard constraints, or promote an artifact.

The Python entry point is `modelsurgeon.migration.migrate_record`. The direct
CLI equivalent is:

```bash
uv run modelsurgeon migrate INPUT.json --output MIGRATED.json --json
```

The command refuses to overwrite an output file. Keep the source record and
validate the migrated record before resuming a campaign or publishing an
artifact.

## Guarantees

The boundary guarantees the following for supported records:

- v2.0 configuration is validated against the current `Settings` contract;
  a missing conversational provider becomes the explicit `none` provider;
- v2.0 autonomous campaign runs upgrade from orchestrator schema 2 to 3 by
  adding only the empty append-only approval-audit field, preserving plan and
  source digests, stage evidence, approvals, outcomes, reasons, and artifact
  lineage;
- v2.0 campaign evidence becomes canonical campaign evidence without dropping
  provenance, source/artifact digests, measurement flags, hard-constraint
  status, or terminal outcomes;
- `supported`, `unsupported`, `failed`, and `unknown` remain distinct; unknown
  and inconclusive evidence remains non-promotable; and
- current records are read-validated and returned unchanged.

Migration results include a separate machine-readable report. The report is
not inserted into the migrated domain record, so existing readers and direct
Python automation continue to consume their original schema contracts.

## Capability matrix

| Source | Target | Status | Guarantee or refusal reason |
| --- | --- | --- | --- |
| v2.0 settings schema 1 | current settings schema 1 | Supported | Validate all fields; default only the optional provider to `none`. |
| v2.0 `autonomous_optimize_run` schema 2 | current `autonomous_optimize_run` schema 3 | Supported | Preserve identity, evidence, approvals, outcomes, reasons, and artifact lineage. |
| `v2.0_campaign_evidence` schema 1 | current `canonical_campaign_evidence` schema 1 | Supported | Preserve identity, lineage, provenance, measurements, and outcome. |
| Current canonical campaign state/evidence | Current schema | Read-only-compatible | Validate and return unchanged; no v2.0 translation is guessed. |
| Stage-only evidence without a source digest | Canonical evidence | Unsupported | Refuse because source lineage cannot be reconstructed. |
| Non-`none` provider in a v2.0 config | Current provider configuration | Unsupported | Provider configuration is a later control-plane contract. |
| Unknown, future, or semantically mismatched schema | Any current schema | Unsupported | Fail closed; no field dropping, renaming, or meaning changes. |

The machine-readable source for this table is
[`docs/research/v3.0-migration-compatibility-v1.json`](research/v3.0-migration-compatibility-v1.json).

## Deprecation policy

Migration support and write support are separate capabilities:

| Source surface | Read/migrate policy | Write policy | Deprecation state |
| --- | --- | --- | --- |
| v2.0 settings schema 1 | Validate and canonicalize; missing provider becomes `none`. | Emit current settings shape. | Deprecated source shape; migration-supported. |
| v2.0 autonomous run schema 2 | Upgrade once to run schema 3 before resume. | Emit run schema 3 only. | Deprecated source shape; migration-supported. |
| v2.0 campaign evidence | Convert to canonical campaign evidence and retain terminal status. | Emit canonical campaign evidence only. | Deprecated source shape; migration-supported. |
| Current settings, campaign, and evidence schemas | Validate and return unchanged. | Current schemas remain the only write contract. | Supported. |
| Unknown/future/provider-specific/lineage-incomplete records | Refuse before execution. | Never emit. | Unsupported; no compatibility promise. |

No automatic in-place rewrite is performed, and no legacy record is deleted.
Future removal of a migration adapter requires a new release-boundary record
and a documented refusal message; it cannot happen as an incidental schema
cleanup.

## What is intentionally unsupported

There is no compatibility promise for opaque, unsigned, provider-specific, or
unvalidated artifacts; stage evidence that lacks immutable source lineage;
future schemas; arbitrary conversational transcripts; or records whose
meaning would require inferring a new constraint, approval, measurement, or
artifact identity. Those inputs fail before execution.

Migration does not make physical surgery, hosted providers, distributed
multi-writer recovery, or absent model-quality measurements supported. Exact
format, model-family, runtime, and hardware claims remain governed by the
[architecture compatibility matrix](architecture-compatibility.md) and the
[v2.0 release audit](release/v2.0-autonomous-optimizer-audit.md).

## Direct automation remains first-class

`modelsurgeon optimize --no-llm`, `build_optimize_plan(Settings(...))`, and
the existing structured campaign/evidence APIs remain valid without a text
model or provider installation. The migration module imports only typed core
contracts and is safe to use from offline Python automation. A migrated plan
still carries the same hard constraints, resource budgets, deterministic
identifiers, provenance, and approval boundary as the source plan.

## Verification

The focused contract is covered by
`tests/test_migration_compatibility.py` and the three fixtures under
`tests/fixtures/v2.0-*.json`. The docs consistency checks verify that the
matrix and machine-readable release record agree with the implementation.
