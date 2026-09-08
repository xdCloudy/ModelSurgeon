"""End-to-end coverage for the integrated bounded conversational journey."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import Settings
from modelsurgeon.conversation import (
    CampaignEvidence,
    CampaignOutcome,
    CampaignStateStore,
    ConversationalJourney,
    JourneyError,
    JourneyOutcome,
    JourneyPhase,
    ProviderKind,
    bootstrap_chat_session,
)
from modelsurgeon.search.spec_preview import build_spec_preview
from test_chat import _FixtureProvider, _gguf
from test_chat_execution import _adapter, _Runtime
from test_infeasibility import _candidate as _infeasible_candidate
from test_infeasibility import _explain as _infeasibility_explain
from test_spec_preview import _intent as _preview_intent


def _journey(tmp_path: Path, *, runtime: _Runtime) -> tuple[ConversationalJourney, object]:
    model = tmp_path / "provider.gguf"
    _gguf(model)
    adapter = _adapter(tmp_path, runtime)
    session = bootstrap_chat_session(
        model,
        runtime_revision="fixture-runtime-v1",
        provider_factory=lambda _config: _FixtureProvider(),
        execution_adapter=adapter,
    )
    return ConversationalJourney(session), session


def test_supported_journey_clarification_preview_execute_and_explain(tmp_path: Path) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime())
    try:
        interpreted = journey.interpret("retain quality and reduce latency")
        assert interpreted.phase is JourneyPhase.INTERPRETATION
        assert interpreted.outcome is JourneyOutcome.SUPPORTED
        preview = journey.preview()
        assert preview.phase is JourneyPhase.PREVIEW
        assert preview.outcome is JourneyOutcome.SUPPORTED

        executed = journey.execute("approval-chat")
        assert executed.phase is JourneyPhase.EXECUTION
        assert executed.outcome is JourneyOutcome.SUPPORTED
        assert executed.campaign_id is not None
        explanation = journey.explain(executed.campaign_id)
        assert explanation.phase is JourneyPhase.EXPLANATION
        assert explanation.evidence_refs
        assert explanation.record["record_type"] == "canonical_claim_evidence_explanation"
    finally:
        session.close()


def test_negative_journey_never_creates_plan_or_artifact(tmp_path: Path) -> None:
    _journey_instance, session = _journey(tmp_path, runtime=_Runtime())
    try:
        # A provider-disabled session is a supported direct/CLI negative path;
        # interpretation cannot silently become a plan.
        session.close()
        disabled = bootstrap_chat_session(Path("unused"), provider_kind=ProviderKind.NONE)
        try:
            negative = ConversationalJourney(disabled).interpret("do surgery")
            assert negative.outcome is JourneyOutcome.UNSUPPORTED
            assert negative.phase is JourneyPhase.INTERPRETATION
        finally:
            disabled.close()
    finally:
        if not getattr(session.provider, "closed", False):
            session.close()


def test_missing_approval_is_refused_before_execution(tmp_path: Path) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime())
    try:
        journey.interpret("retain quality and reduce latency")
        journey.preview()
        with pytest.raises(JourneyError, match="explicit approval") as error:
            journey.execute(None)
        assert error.value.code == "approval_required"
    finally:
        session.close()


def test_pause_resume_and_stale_explanation_context_are_fail_closed(tmp_path: Path) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime(interrupt_once=True))
    try:
        journey.interpret("retain quality and reduce latency")
        journey.preview()
        paused = journey.execute("approval-chat")
        assert paused.outcome is JourneyOutcome.UNKNOWN
        assert paused.campaign_id is not None

        with pytest.raises(JourneyError, match="explicit approval"):
            journey.resume(paused.campaign_id, None)
        resumed = journey.resume(paused.campaign_id, "approval-chat")
        assert resumed.phase is JourneyPhase.LIFECYCLE

        with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
            current = store.load(paused.campaign_id)
            store.append_evidence(
                paused.campaign_id,
                CampaignEvidence(
                    "evidence_stale_fixture",
                    current.source_model_digest,
                    CampaignOutcome.UNKNOWN,
                    "new retained evidence",
                    {"record_type": "fixture_stale_evidence"},
                    inconclusive=True,
                ),
                expected_version=current.state_version,
            )
        with pytest.raises(JourneyError, match="stale") as error:
            journey.explain(paused.campaign_id)
        assert error.value.code == "stale_context"
        refreshed = journey.explain(paused.campaign_id, refresh=True)
        assert "evidence_stale_fixture" in refreshed.evidence_refs
    finally:
        session.close()


def test_plan_diff_and_provider_diagnostics_share_typed_direct_contracts(tmp_path: Path) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime())
    try:
        first = build_spec_preview(_preview_intent(threshold=0.95))
        second = build_spec_preview(_preview_intent(threshold=0.97), previous=first)
        diff = journey.plan_diff(first, second)
        assert diff.to_record()["kind"] == "material"
        assert "spec.constraints[0].threshold" in diff.to_record()["changed_paths"]
        diagnostic = journey.provider_diagnostics(Settings())
        assert diagnostic.code == "no_llm"
        diagnostic_response = journey.diagnose_provider(Settings())
        assert diagnostic_response.phase is JourneyPhase.DIAGNOSTICS
        assert diagnostic_response.record == diagnostic.to_record()
    finally:
        session.close()


def test_infeasibility_alternatives_are_presented_as_typed_negative_evidence(
    tmp_path: Path,
) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime())
    try:
        explanation = _infeasibility_explain(
            (_infeasible_candidate("near", quality=0.94, latency=101.0),)
        )
        presented = journey.present_alternatives(explanation)
        assert presented.outcome is JourneyOutcome.UNKNOWN
        assert presented.record["outcome"] == "infeasible"
        assert presented.evidence_refs == ("evidence_near",)
        with pytest.raises(JourneyError, match="canonical explanation"):
            journey.present_alternatives({"outcome": "infeasible"})
    finally:
        session.close()


def test_cli_campaign_inspect_and_explain_match_canonical_store(tmp_path: Path) -> None:
    journey, session = _journey(tmp_path, runtime=_Runtime())
    try:
        journey.interpret("retain quality and reduce latency")
        journey.preview()
        result = journey.execute("approval-chat")
        assert result.campaign_id is not None
        assert result.state_digest is not None
        assert result.record["execution"] != "not_requested"
        runner = CliRunner()
        inspected = runner.invoke(
            app,
            [
                "campaign",
                "inspect",
                result.campaign_id,
                "--state",
                str(tmp_path / "chat-run.campaign.sqlite3"),
                "--session-id",
                session.bootstrap.session_id,
                "--json",
            ],
            color=False,
        )
        assert inspected.exit_code == 0, inspected.output
        inspected_record = json.loads(inspected.stdout)
        assert inspected_record["campaign_id"] == result.campaign_id
        explained = runner.invoke(
            app,
            [
                "campaign",
                "explain",
                result.campaign_id,
                "--state",
                str(tmp_path / "chat-run.campaign.sqlite3"),
                "--session-id",
                session.bootstrap.session_id,
                "--json",
            ],
            color=False,
        )
        assert explained.exit_code == 0, explained.output
        assert json.loads(explained.stdout)["campaign_id"] == result.campaign_id
    finally:
        session.close()
