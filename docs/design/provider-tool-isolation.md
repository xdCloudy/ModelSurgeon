# Provider, tool, and secret isolation

Status: the bounded v2.9 isolation contract is implemented for the provider
and conversational-tool boundaries. It is a capability boundary with explicit
fail-closed observations; it is not a claim of hostile-process containment.

## Trust zones

The deterministic engine is the only trusted zone. Provider results and tool
results are marked as untrusted data until an engine-owned policy or evidence
adapter explicitly promotes a field. Provider output cannot supply tool
callbacks, execution handles, approvals, source digests, campaign state, or
accepted measurements. Tool output is checked against the allowlisted schema;
its provenance is constructed by the dispatcher from engine-owned lineage, not
from provider text or arbitrary result metadata.

Provider requests and tool handler requests are copied before crossing the
boundary. A provider that mutates its private request copy cannot mutate the
trusted request, policy, or evidence records. Provider identity and capability
metadata are snapshotted and compared after the call; drift, unavailable
metadata, or an uncopyable request returns a retained
`failed/isolation_failure` result.

## Secret handling

Credentials are referenced indirectly (for example, by an environment name or
`AuthReference`) and resolved only at transport time. Credential values are
never included in canonical settings, request provenance, endpoint metadata, or
reproducibility records. Diagnostics, provider failures, tool failures, tool
outputs, and retained raw payloads use recursive redaction for common secret
assignments, bearer values, URL userinfo, PEM blocks, and caller-supplied
secret values.

Redaction is a safety net, not a storage mechanism. Callers must not put
credentials in prompts, evidence records, issue fixtures, or chat history.

## Observable fail-closed behavior

Crashes, malformed output, timeouts, cancellation, capability drift, request
copy failures, and boundary metadata failures remain typed negative outcomes.
Unknown and unsupported outcomes remain distinct and are not promoted to
success. The result retains the stable request identity and a bounded,
redacted diagnostic so offline replay and audit can see that the boundary
failed without receiving the secret or trusting the untrusted payload.

## Process-isolation limitation

The current provider and trusted tool handler interfaces run in the host
process. Their contract exposes no executor, filesystem, network, model-session,
or generic callback capability, and the dispatcher applies wall-time, memory,
evaluation, output, cancellation, transaction, and replay limits. Those are
capability and resource controls, not an operating-system sandbox: a malicious
in-process implementation could still access ambient process resources. A
future hosted or third-party adapter that requires hostile-code resistance must
run behind a separately supervised process or service boundary with an audited
IPC protocol. This repository does not claim that evidence today.

## Offline/local-first validation

`provider.kind=none` remains the default and direct CLI/Python workflows do not
need provider credentials. The local GGUF adapter can be exercised with a
licensed tiny fixture and an installed local runtime. Hosted and compatible
endpoint runs require separately supplied credentials and live evidence; no
such live run is claimed by this contract.
