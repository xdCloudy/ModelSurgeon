# v2.9 adversarial resistance corpus

Issue #482 adds a versioned, deterministic resistance layer above the v2.6
tool contract and the v2.8 canonical evidence-query path. The fixture is
[`adversarial_resistance_v1.json`](../../tests/fixtures/adversarial_resistance_v1.json);
the executable audit is
[`audit_adversarial_resistance.py`](../../tools/audit_adversarial_resistance.py),
and the acceptance tests are in
[`test_adversarial_resistance.py`](../../tests/test_adversarial_resistance.py).

## Contract under test

The trusted structured policy is the only authority for allowlists, approval,
budgets, hard constraints, source identity, evidence promotion and artifact
publication. User text, provider identity/capability descriptions, provider
output, tool input and tool output are untrusted boundary data. A successful
decode does not promote untrusted content to canonical evidence.

The corpus exercises:

- prompt injection and instruction smuggling in user/provider text;
- hostile model descriptions, provider metadata drift and malformed provider
  output;
- forged evidence references and attempts to label unverified output canonical;
- secret-shaped requests and diagnostics, including retained failure data;
- path attempts, unknown request fields and budget expansion;
- conflicting approval authority and request/result copy mutation;
- malformed or hostile tool results; and
- the explicit unresolved hostile-process-containment limitation.

Every case declares a severity, remediation status, provenance source and seed,
variant budget, wall/output budget, expected typed outcome and expected
authority result. Four deterministic text variants are used where applicable:
base, whitespace, casefold and zero-width Unicode spacing. No external model,
network, credential, shell, filesystem, or subprocess is needed to run the
corpus.

## Failure and evidence policy

Negative outcomes are evidence, not omitted rows. The audit retains the case
ID, variant, severity, remediation status, typed outcome/failure code, bounded
budget, trust zone and deterministic result provenance for every observation.
Malformed data, policy refusal, isolation failure, output rejection and
unsupported behavior must remain failed closed. Provider/tool assertions may
describe or explain data, but cannot supply approvals, callbacks, execution
handles, source digests or canonical evidence lineage.

The unresolved case records the current in-process limitation: this suite does
not claim operating-system containment for a hostile third-party provider or
handler. A future claim requires a separately supervised process and audited
IPC boundary; adding more prompt cases cannot establish that property.

## Threat assumptions and residual risk

The suite assumes the deterministic engine, structured policy, canonical
campaign/evidence stores and approval/transaction boundaries are trusted. It
does not assume that text is honest, that provider metadata is stable, or that
tool output is truthful. Secret redaction is treated as a retention safety net,
not as permission to place credentials in prompts or evidence.

The residual risks are deliberately retained: in-process handlers can still
access ambient process resources, live hosted-provider behavior and credential
handling are not claimed here, and this corpus cannot prove resistance to every
future model encoding or Unicode confusable. New cases must preserve the same
versioned schema, deterministic seed, bounded budgets, provenance and explicit
remediation state.

## Reproduction

```text
uv run --locked --extra dev python tools/audit_adversarial_resistance.py
uv run --locked --extra dev pytest tests/test_adversarial_resistance.py tests/test_conversational_adversarial_corpus.py -q
```
