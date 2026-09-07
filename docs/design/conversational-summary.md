# Conversational summary boundary

Conversation summaries are bounded, non-authoritative transport views. They
reduce transcript cost without becoming a second campaign store or an input
that can alter a plan.

## Authority zones

Each summary keeps separate zones:

- `canonical_state` contains the exact versioned campaign state, including the
  spec identity and digest, hard constraints, approval, lifecycle, budgets and
  provider context.
- `canonical_evidence` contains the complete evidence records visible at the
  captured state, including unsupported, failed, unknown and inconclusive
  outcomes.
- `untrusted_transcript` contains only bounded transcript entries. It is
  context for explanation and clarification, never an execution authority.

The summary carries the campaign/session identity, state version and digest.
Rehydration compares those values and the full canonical records against the
trusted `CampaignStateStore`; it refuses stale, altered or unsupported
summaries and never replays chat to recover state.

## Loss and limits

Transcript retention uses deterministic UTF-8 budget units rather than a
provider tokenizer. Newest supported entries are retained first. Every
omission is recorded with counts, token units and a digest; unsupported roles
are explicitly listed. If the canonical state, evidence or loss markers do
not fit the summary byte budget, the operation returns `unsupported` instead
of dropping authoritative content.

The summary digest covers all zones, the loss marker and the budget. A summary
with loss is therefore still deterministic and inspectable, but callers must
not treat omitted transcript text as absent evidence. The rehydrated provider
context labels canonical and untrusted data separately.
