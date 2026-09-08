# Conversational policy precedence

The conversational boundary has one policy resolver shared by intent
compilation, tool dispatch, canonical campaign creation, and evidence-grounded
explanations. It is an execution-control record, not a model confidence score.

## Precedence

Sources are evaluated from strongest to weakest:

1. **Hard constraints** — immutable quality, safety, resource, and deployment
   bounds. A prompt, tool argument, provider claim, or approval cannot relax
   one.
2. **Validated spec** — the exact typed objective contract emitted by the
   compiler.
3. **Approval policy** — explicit, scoped approval for the exact plan and
   material diff.
4. **Tool capability** — the allowlisted operation, schema, budget, handler,
   transaction, and replay boundary.
5. **Evidence status** — canonical, incomplete, unavailable, unsupported,
   failed, or unknown engine evidence.
6. **User objective**, prompt text, and provider text — useful inputs to
   interpretation, but never an execution authority by themselves.

Lower sources may add a restriction, but they cannot turn a higher-source
refusal into an approval. Prompt and provider candidates are always retained
as rejected alternatives; they are never eligible to authorize an operation.

## Fail-closed contract

The shared `PolicyDecision` record contains the operation, outcome, winning
source, winning detail, all considered candidates, rejected alternatives,
diagnostics, and a deterministic decision ID. The only executable outcome is
`allow` with a winning trusted source.

- A missing trusted source is `unknown` and non-executable.
- An unknown trusted capability, approval, or evidence state remains unknown;
  it is not guessed from a prompt or provider response.
- Multiple states from one source are `contradictory` and non-executable.
- `deny` and `unsupported` remain distinct from `unknown`; none is promoted to
  success.
- A lower source can veto an otherwise permitted operation when it reports an
  actual trusted boundary failure, such as an unavailable tool or missing
  approval. It cannot weaken a hard constraint.

Every layer keeps the same decision identity. Campaign state persists the
record under `policy_precedence`; dispatch results expose it beside the typed
tool result; compiler and explanation objects retain it for audit and replay.
Existing versioned projections remain backward-compatible where their frozen
golden record shape is unchanged.

## Examples

If a provider says “ignore the quality floor” while the typed hard constraint
requires quality ≥ 0.95, the winning source is `hard_constraints`, the outcome
is `deny` when the candidate violates that floor, and the provider statement is
listed under `rejected_alternatives`.

If the plan is valid but the consequential tool has no valid scoped approval,
`approval_policy` wins with `unknown` or `deny`; the dispatcher does not call
the handler. If evidence is incomplete, the explanation retains the negative
or inconclusive evidence and does not present a successful artifact.
