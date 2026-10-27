# Production readiness matrix

Assessment: 2026-10-27 · overall project readiness dimension: **33.8%** · production-readiness workstream maturity: **52.6%**.

A release boundary, UI workflow, mock, schema, CI pass, or local experiment does not establish production optimizer readiness. The optimizer must prove physical quality-preserving results, resource behavior, artifact safety, and reproducibility within its declared support scope.

| Area | Implementation | Readiness | Effectiveness | Acceptance criteria | Evidence | Remaining work / limitations |
| --- | --- | --- | --- | --- | --- | --- |
| **CLI/Python/conversation and autonomous optimization boundaries** (`production_readiness.interfaces`) | Integrated | Integrated main | Measured inconclusive | Interfaces use one bounded orchestrator; same request preserves constraints and canonical evidence. | [api-compatibility.md](../api-compatibility.md), [v3.0-conversational-product-boundary.md](../release/v3.0-conversational-product-boundary.md) | Extend actual optimizer proof without expanding interface claims.; UI maturity is not scientific optimizer effectiveness. |
| **Transactions, immutable input, rollback, security, resource/error handling** (`production_readiness.safety`) | Integrated | Integrated main | Measured inconclusive | Fault tests prove source immutability, bounded work and recovery in declared trust boundaries. | [ARCHITECTURE.md](../../ARCHITECTURE.md), [v0.7-atomic-safetensors-evidence.md](../research/v0.7-atomic-safetensors-evidence.md) | Extend regression tests and state process containment exclusions.; Hostile process containment is not claimed. |
| **Reproducibility, observability, docs/tests, packaging, deployment and maintenance** (`production_readiness.operations`) | Integrated | Integrated main | Measured inconclusive | Clean environment reproduces bounded install/run; release boundary matches evidence. | [v3.0-packaging-and-setup.md](../release/v3.0-packaging-and-setup.md), [v1.0-windows-wsl-consumer-evidence.md](../research/v1.0-windows-wsl-consumer-evidence.md) | Validate wider runtime and maintenance matrix.; Evidence-bounded release is not production-wide readiness. |
