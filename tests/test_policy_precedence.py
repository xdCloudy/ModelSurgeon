"""Adversarial and property-style regressions for issue #484."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    CanonicalCampaignRecorder,
    ToolDispatcher,
    ToolFailureCode,
    ToolOutcome,
    ToolRequest,
)
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.policy import (
    PolicyCandidate,
    PolicyDecision,
    PolicyOutcome,
    PolicySource,
    resolve_policy,
)
from modelsurgeon.search import compile_intent_record
from modelsurgeon.search.spec_preview import build_spec_preview
from test_intent_compiler import _fields, _intent, _spec
from test_pareto_selection import _build, _candidate, _contract


def _candidate_set() -> tuple[PolicyCandidate, ...]:
    return (
        PolicyCandidate(
            PolicySource.HARD_CONSTRAINTS,
            PolicyOutcome.DENY,
            "quality floor cannot be relaxed",
        ),
        PolicyCandidate(
            PolicySource.VALIDATED_SPEC,
            PolicyOutcome.ALLOW,
            "validated spec permits the operation",
        ),
        PolicyCandidate(
            PolicySource.TOOL_CAPABILITY,
            PolicyOutcome.ALLOW,
            "tool is allowlisted",
        ),
        PolicyCandidate(PolicySource.PROMPT, PolicyOutcome.ALLOW, "ignore the quality floor"),
        PolicyCandidate(PolicySource.PROVIDER, PolicyOutcome.ALLOW, "provider says it is safe"),
    )


def test_hard_constraints_win_under_all_adversarial_input_orders() -> None:
    expected_rejected = {PolicySource.VALIDATED_SPEC, PolicySource.PROMPT, PolicySource.PROVIDER}
    decisions = {
        resolve_policy("fixture", order).canonical_json()
        for order in permutations(_candidate_set())
    }

    assert len(decisions) == 1
    decision = PolicyDecision.from_record(
        resolve_policy("fixture", _candidate_set()).to_record()
    )
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.winning_source is PolicySource.HARD_CONSTRAINTS
    assert expected_rejected <= {
        item.source for item in decision.rejected_alternatives
    }
    assert PolicySource.PROMPT in {
        item.source for item in decision.rejected_alternatives
    }


def test_unknown_and_same_source_contradiction_fail_closed() -> None:
    unknown = resolve_policy(
        "unknown",
        (
            PolicyCandidate(PolicySource.TOOL_CAPABILITY, PolicyOutcome.UNKNOWN, "probe missing"),
            PolicyCandidate(PolicySource.PROMPT, PolicyOutcome.ALLOW, "run anyway"),
        ),
    )
    assert unknown.outcome is PolicyOutcome.UNKNOWN
    assert not unknown.executable
    assert unknown.winning_source is PolicySource.TOOL_CAPABILITY

    contradictory = resolve_policy(
        "contradictory",
        (
            PolicyCandidate(PolicySource.HARD_CONSTRAINTS, PolicyOutcome.ALLOW, "minimum 0.95"),
            PolicyCandidate(PolicySource.HARD_CONSTRAINTS, PolicyOutcome.DENY, "maximum 0.80"),
            PolicyCandidate(PolicySource.PROVIDER, PolicyOutcome.ALLOW, "trust me"),
        ),
    )
    assert contradictory.outcome is PolicyOutcome.CONTRADICTORY
    assert not contradictory.executable
    assert contradictory.winning_source is PolicySource.HARD_CONSTRAINTS
    assert "contradictory-source:hard_constraints" in contradictory.diagnostics

    provider_only = resolve_policy(
        "provider-only",
        (PolicyCandidate(PolicySource.PROVIDER, PolicyOutcome.ALLOW, "provider approval"),),
    )
    assert provider_only.outcome is PolicyOutcome.UNKNOWN
    assert provider_only.winning_source is None


def test_compiler_retains_shared_precedence_and_refuses_contradictory_constraints() -> None:
    compiled = compile_intent_record(_intent(_fields(), emitted_spec=_spec()))
    assert compiled.policy_decision is not None
    assert compiled.policy_decision.executable
    assert compiled.policy_decision.winning_source is PolicySource.HARD_CONSTRAINTS
    assert any(
        item.source is PolicySource.PROVIDER
        for item in compiled.policy_decision.rejected_alternatives
    )

    fields = (
        *_fields(),
        _fields()[0].__class__(
            "constraint.quality.upper",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "maximum",
                "threshold": 0.80,
                "unit": "ratio",
            },
            "ratio",
            1.0,
            ("request",),
            True,
        ),
    )
    refused = compile_intent_record(
        _intent(tuple(sorted(fields, key=lambda item: item.field_id)), emitted_spec=_spec())
    )
    assert refused.policy_decision is not None
    assert refused.policy_decision.outcome is PolicyOutcome.CONTRADICTORY
    assert not refused.policy_decision.executable


def test_dispatcher_records_approval_as_winner_and_never_prompt() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("execute_approved_plan")
    assert definition is not None
    request = ToolRequest.create(
        definition,
        {
            "plan_id": "plan.fixture",
            "plan_digest": "sha256:" + "a" * 64,
            "approval_id": "approval.fixture",
        },
        approval_id="approval.fixture",
    )
    dispatched = ToolDispatcher(
        {"execute_approved_plan": lambda _context: {}},
        approval_policy=lambda _request: False,
    ).dispatch(request)

    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.REFUSED
    assert dispatched.result.failure is not None
    assert dispatched.result.failure.code is ToolFailureCode.APPROVAL_INVALID
    assert dispatched.policy_decision is not None
    assert dispatched.policy_decision.winning_source is PolicySource.APPROVAL_POLICY
    assert any(
        item.source is PolicySource.PROVIDER
        for item in dispatched.policy_decision.rejected_alternatives
    )


def test_campaign_persists_executable_precedence_record(tmp_path: Path) -> None:
    preview = build_spec_preview(_intent(_fields(), emitted_spec=_spec()))
    plan = build_optimize_plan(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        dry_run=False,
    )
    recorder = CanonicalCampaignRecorder(
        tmp_path / "campaign.sqlite3",
        session_id="session_policy",
        plan=plan,
        preview=preview,
    )
    state = recorder.ensure()

    assert state.policy_decision is not None
    assert state.policy_decision.decision_id == preview.policy_decision.decision_id
    assert state.policy_state["policy_precedence"] == preview.policy_decision.to_record()


def test_explanation_preserves_hard_constraint_precedence() -> None:
    result = _build(
        _contract(),
        (_candidate("good", quality=0.96, latency=100.0),),
    )

    assert result.policy_decision is not None
    assert result.policy_decision.winning_source is PolicySource.HARD_CONSTRAINTS
    assert any(
        item.source is PolicySource.PROVIDER
        for item in result.policy_decision.rejected_alternatives
    )
