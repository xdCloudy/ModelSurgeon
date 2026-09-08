# Bounded conversational journey

The conversational journey is a typed control-plane view over the existing
ModelSurgeon engine. Interpretation and clarification can propose a structured
objective; the trusted planner produces a read-only preview; execution accepts
only that exact preview and a scoped approval. Campaign state and retained
evidence remain canonical. Chat text is not persisted as campaign state.

The Python facade is `modelsurgeon.conversation.ConversationalJourney`:

```python
from modelsurgeon.conversation import ConversationalJourney

journey = ConversationalJourney(session, campaign_state_path="chat.campaign.sqlite3")
interpreted = journey.interpret("retain quality above 0.95 and reduce latency")
preview = journey.preview()
executed = journey.execute("approval-issued-for-the-visible-plan")
explanation = journey.explain(executed.campaign_id)
```

`preview()` is read-only. `execute()` requires the approval ID and delegates
plan, policy, resource, measurement, transaction, and artifact decisions to the
trusted adapter. An approval is never inferred from the request or the chat
provider. The execution response includes canonical campaign and evidence IDs;
an artifact reference appears only when the engine reports a validated,
immutable artifact.

Campaigns can be inspected and managed without reopening a transcript:

```text
uv run modelsurgeon campaign inspect CAMPAIGN_ID \
  --state chat-run.campaign.sqlite3 --session-id CHAT_SESSION_ID --json
uv run modelsurgeon campaign pause CAMPAIGN_ID \
  --state chat-run.campaign.sqlite3 --session-id CHAT_SESSION_ID --json
uv run modelsurgeon campaign resume CAMPAIGN_ID \
  --state chat-run.campaign.sqlite3 --session-id CHAT_SESSION_ID \
  --approval-id APPROVAL_ID --json
uv run modelsurgeon campaign explain CAMPAIGN_ID \
  --state chat-run.campaign.sqlite3 --session-id CHAT_SESSION_ID --json
```

Resume and cancel require the currently active approval bound to the campaign.
An explanation is a deterministic projection of a canonical evidence query; it
retains failed, unsupported, unknown, and inconclusive rows and reports missing
or unavailable fields. Supplying an old state digest fails with a stale-context
error. Reconnect with `inspect` or call `journey.reconnect()` to refresh the
trusted snapshot before requesting a new explanation.

Provider diagnostics use the same typed, redacted record in Python and the CLI:

```text
uv run modelsurgeon provider diagnostics --no-llm --json
```

Provider text may help interpret or explain a result, but it cannot change hard
constraints, canonical evidence, approval scope, lifecycle, or artifact status.
Prediction-only or infeasible results remain alternatives or negative evidence;
they are never promoted as deployable artifacts.

This guide describes the bounded integration verified by
`tests/test_v30_conversational_journey.py`. It does not claim universal model
family support, live hosted-provider quality, distributed recovery, optimizer
proof, or hostile-process containment.
