# Frozen v2.1 conversational intent contract

Status: the bounded v2.1 intent boundary is frozen and implemented. The
conversational product remains planned. The machine-readable freeze record is
[`docs/research/v2.1-conversational-intent-contract-v1.json`](../research/v2.1-conversational-intent-contract-v1.json).

## Boundary

The v2.1 boundary accepts a validated, typed `IntentRecord` and produces a
policy decision over the existing v2 `ObjectiveContract`. It does not define a
new optimization schema:

```text
request text
  → IntentRecord v1
  → compiler v1
  → existing ObjectiveContract v1 / OptimizationSpec
  → policy decision v1
  → deterministic ModelSurgeon APIs
```

`ObjectiveContract.to_record()` is the only optimization record emitted by the
boundary. The canonical objective contract is defined in
[`v2.0-objective-contract-v1.json`](../research/v2.0-objective-contract-v1.json)
and remains owned by `modelsurgeon.search.objective_contract`.

## Versioned records

| Record | Version | Role |
| --- | ---: | --- |
| `IntentRecord` | 1 | Original request, normalized fields, source spans, ambiguity, interpretation and provenance. |
| `IntentCompilation` | 1 | Deterministic compiler diagnostics and the existing objective contract when executable. |
| `IntentPolicyDecision` | 1 | Fail-closed outcome, confidence categories, ambiguity assessments and provenance-linked diagnostics. |
| Intent corpus | 1 | Exact expected compiler/policy results for equivalence, refusal and retained-negative cases. |
| `ObjectiveContract` | 1 | The sole stable optimization schema consumed by search/execution. |

Unknown versions are rejected. A change to field meaning, identity rules,
outcome precedence, or the optimization boundary requires a new version and a
migration/regression test; it must not be silently interpreted as v1.

## Outcomes and precedence

The compiler and policy layers expose only these outcomes:

- `executable`: a validated objective contract is safe to submit to the
  deterministic engine;
- `clarification_required`: a required hard constraint or interpretation is
  incomplete, ambiguous, low-confidence, or conflicting;
- `unsupported`: a requested field or operation is outside the declared
  contract; and
- `refused`: the request is contradictory, invalid, or attempts to bypass a
  hard boundary.

Policy precedence is fail-closed: `refused` → `unsupported` →
`clarification_required` → `executable`. No non-executable outcome carries an
objective contract. Missing hard constraints are unresolved; the compiler does
not invent thresholds from prose, provider suggestions, or confidence values.

Confidence is interpretation metadata only. It is not evidence that a surgery
is safe, that a hard constraint is met, or that an artifact may be promoted.

## Supported and unsupported scope

The v2.1 compiler maps typed hard constraints and soft objectives onto the
existing objective contract. Budgets, allowed operations, and deployment
targets are retained as explicit unsupported intent fields because the v2
objective record cannot represent them. They are not dropped or guessed.

The boundary is side-effect free. It does not call a provider, inspect weights,
select tensors, choose optimizer strategy, execute mutation, or publish an
artifact. The deterministic engine retains authority over capability checks,
budgets, mutation, rollback, measurement, acceptance, provenance, and
publication.

## Replay and evidence

The equivalence/refusal corpus is the release evidence for this contract. It
compares canonical specs, outcomes, diagnostics, ambiguity categories and
provenance for each fixture. Supported, clarification, unsupported, refused,
inconclusive, and implementation-failure results remain visible.

Run the focused contract gate with:

```text
uv run pytest tests/test_v21_intent_contract.py tests/test_intent_compiler.py tests/test_intent_policy.py tests/test_intent_corpus.py
```

The repository quality gate is:

```text
uv run ruff check .
uv run mypy src
uv run pytest --cov=modelsurgeon --cov-report=term-missing
```

The built-in compiler and its public aliases are the currently available
implementations. No provider-backed compiler is registered, so provider
availability is a known skip rather than an unverified success. Full chat UX,
provider orchestration, conversational campaign state, and tool execution are
later roadmap work.
