# Conversational tool-boundary adversarial corpus

This corpus is one of the release checks in the [v2.6 bounded tool release
record](../research/v2.6-bounded-conversational-tool-release-v1.json). The
unsupported cells are deliberate: no arbitrary shell/code tool, provider
selector, live hosted-provider call, live campaign, or hostile-process
containment is part of this evidence.

The versioned fixture
[`conversational_tool_boundary_adversarial_v1.json`](../../tests/fixtures/conversational_tool_boundary_adversarial_v1.json)
is the deterministic v2.6 corpus for the typed conversational dispatcher. It
is exercised by `tests/test_conversational_adversarial_corpus.py` through the
same request-decoding, negotiation, dispatch, transaction, result, and replay
paths used by an engine adapter.

The corpus covers:

- shell/code requests and path traversal attempts;
- secret-exfiltration fields and prompt-injection/provider-metadata smuggling;
- unknown request schema versions and malformed tool schemas;
- forged measurements that remain `unverified` rather than becoming canonical;
- contradictory canonical provenance, which fails closed;
- replayed deterministic request IDs; and
- retained failure diagnostics with secret-shaped values redacted.

The default tool surface remains a finite allowlist. Corpus inputs cannot select
a callback, command, path, provider, model session, or arbitrary metadata field.
Handlers are trusted engine registrations; provider/model text is never used to
populate trusted result outcome or provenance fields. Raw failure payloads are
retained only as bounded, separate diagnostic data and are not evidence.

Run the corpus with:

```text
uv run --extra dev pytest tests/test_conversational_adversarial_corpus.py
```

This is contract and fixture evidence only. It does not claim live provider,
campaign, model, shell, or benchmark execution. The dispatcher uses bounded
in-process cooperative cancellation and trusted handlers; a malicious handler
would require a separate process-isolation boundary.
