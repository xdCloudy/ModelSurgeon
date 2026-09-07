# Deterministic clarification state machine

The v2.4 clarification boundary is implemented by
`modelsurgeon.conversation.clarification`. It is a resumable control-plane
record around the canonical `IntentRecord` and `IntentPolicyDecision`; it is
not a second objective schema or an execution authority.

## State and question policy

`ClarificationMachine.start()` evaluates the existing intent policy and derives
questions only for executable gaps:

- missing hard constraints or soft objectives;
- required unresolved ambiguities;
- low-confidence required fields; and
- ambiguous preference ordering.

Contradictory and unsupported policy outcomes do not receive a question. They
remain fail-closed. Prompts, question IDs, alternatives, source references and
diagnostic links are generated from canonical policy data and are stable for
the same intent, evidence and confidence thresholds.

The states are `incomplete`, `ambiguous`, `contradictory`, `executable`,
`repeated`, `cancelled`, and `unsupported`. Repeated and unsupported answer
attempts retain the pending questions so a later typed answer can be tried;
cancellation is terminal. An executable state contains the exact policy-owned
objective contract and no questions.

## Typed answers and authority

Answers are `ClarificationAnswer` records containing a question ID and a JSON
value. They must address an existing necessary question. A missing declaration
must provide the typed metric, direction, threshold/unit fields; the machine
adds no default safety threshold. Existing hard constraints cannot be replaced
with a non-constraint value, and an answer cannot remove any other field or
constraint.

After every accepted answer the machine creates a derived intent with an
explicit clarification provenance reference, compiles its fields through the
existing compiler helpers, and runs the existing policy evaluator again. Only a
new `executable` policy decision may expose a spec. Unsupported schema values,
contradictory declarations, repeated answers and cancelled sessions are
retained as deterministic transitions without execution.

For an ambiguous preference ordering, the necessary question contains the
canonical IDs of the competing soft fields. The answer must select one of
those fields explicitly. The machine removes only the unselected soft
preferences from the derived interpretation; hard constraints, source spans,
provenance, and rejected attempts remain retained. Contradictory hard
constraints do not receive a question: they remain refused until the user
submits a new, explicit amended intent.

## Replay and integration

`ClarificationState.canonical_json()` includes the root intent, current intent,
policy decision, questions, typed answer attempts, thresholds and transitions.
`ClarificationState.from_json()` re-evaluates policy and rejects tampering.
`replay_clarification()` starts from the root intent and applies the same typed
answers in order, including repeated and unsupported attempts.

Chat turns expose the state as `turn.clarification`. `ChatSession.answer()` /
`answer_clarification()` and `cancel_clarification()` are deterministic
control-plane operations; they never call a provider or an optimizer and they
discard no canonical evidence.
