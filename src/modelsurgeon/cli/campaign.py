"""Direct CLI access to canonical conversational campaign state and evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.conversation import (
    ApprovalStatus,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryOutcome,
    EvidenceQueryStatus,
    EvidenceSnapshot,
)
from modelsurgeon.explain import render_claim_evidence

campaign_app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _emit(record: dict[str, object], *, output_json: bool) -> None:
    if output_json:
        typer.echo(json.dumps(record, sort_keys=True))
    else:
        typer.echo(json.dumps(record, sort_keys=True, indent=2))


def _run[T](action: Callable[[], T], *, output_json: bool) -> T:
    try:
        return action()
    except (CampaignStateError, ValueError) as error:
        code = getattr(error, "code", None)
        if code is None:
            code = {
                "EvidenceQueryStaleError": "stale_context",
                "EvidenceQueryIntegrityError": "evidence_integrity",
                "EvidenceQueryResourceError": "evidence_budget",
            }.get(type(error).__name__, "campaign_error")
        payload = {
            "record_type": "error",
            "category": "campaign",
            "outcome": "failed",
            "code": code,
            "message": str(error),
        }
        _emit(payload, output_json=output_json)
        raise typer.Exit(2) from error


def _state(
    store: CampaignStateStore, campaign_id: str, session_id: str
) -> CampaignState:
    return store.reconnect(campaign_id, session_id)


def _require_approval(state: CampaignState, approval_id: str | None) -> None:
    if approval_id is None:
        raise CampaignStateError("explicit approval is required for lifecycle work")
    if (
        state.approval.status is not ApprovalStatus.APPROVED
        or not state.approval.active
        or state.approval.approval_id != approval_id
    ):
        raise CampaignStateError("approval is missing, expired, or bound to another campaign")


@campaign_app.command("inspect")
def inspect_command(
    campaign_id: Annotated[str, typer.Argument(help="Canonical campaign ID")],
    state: Annotated[Path, typer.Option("--state", help="Campaign SQLite state path")],
    session_id: Annotated[str, typer.Option("--session-id", help="Trusted session identity")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read canonical state without consulting chat history."""

    result = _run(
        lambda: _inspect(state, campaign_id, session_id), output_json=output_json
    )
    assert isinstance(result, CampaignState)
    _emit(result.to_record(), output_json=output_json)


def _inspect(path: Path, campaign_id: str, session_id: str) -> CampaignState:
    with CampaignStateStore(path) as store:
        return _state(store, campaign_id, session_id)


@campaign_app.command("pause")
def pause_command(
    campaign_id: Annotated[str, typer.Argument(help="Canonical campaign ID")],
    state: Annotated[Path, typer.Option("--state", help="Campaign SQLite state path")],
    session_id: Annotated[str, typer.Option("--session-id", help="Trusted session identity")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Pause a campaign without discarding retained work."""

    result = _run(
        lambda: _pause(state, campaign_id, session_id), output_json=output_json
    )
    assert isinstance(result, CampaignState)
    _emit(result.to_record(), output_json=output_json)


def _pause(path: Path, campaign_id: str, session_id: str) -> CampaignState:
    with CampaignStateStore(path) as store:
        return store.pause(campaign_id, session_id)


@campaign_app.command("resume")
def resume_command(
    campaign_id: Annotated[str, typer.Argument(help="Canonical campaign ID")],
    state: Annotated[Path, typer.Option("--state", help="Campaign SQLite state path")],
    session_id: Annotated[str, typer.Option("--session-id", help="Trusted session identity")],
    approval_id: Annotated[str | None, typer.Option("--approval-id")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume only with the approval bound to the current campaign."""

    result = _run(
        lambda: _resume(state, campaign_id, session_id, approval_id), output_json=output_json
    )
    assert isinstance(result, CampaignState)
    _emit(result.to_record(), output_json=output_json)


def _resume(
    path: Path, campaign_id: str, session_id: str, approval_id: str | None
) -> CampaignState:
    with CampaignStateStore(path) as store:
        current = _state(store, campaign_id, session_id)
        _require_approval(current, approval_id)
        return store.resume(campaign_id, session_id)


@campaign_app.command("cancel")
def cancel_command(
    campaign_id: Annotated[str, typer.Argument(help="Canonical campaign ID")],
    state: Annotated[Path, typer.Option("--state", help="Campaign SQLite state path")],
    session_id: Annotated[str, typer.Option("--session-id", help="Trusted session identity")],
    approval_id: Annotated[str | None, typer.Option("--approval-id")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Cancel only with the approval bound to the current campaign."""

    result = _run(
        lambda: _cancel(state, campaign_id, session_id, approval_id), output_json=output_json
    )
    assert isinstance(result, CampaignState)
    _emit(result.to_record(), output_json=output_json)


def _cancel(
    path: Path, campaign_id: str, session_id: str, approval_id: str | None
) -> CampaignState:
    with CampaignStateStore(path) as store:
        current = _state(store, campaign_id, session_id)
        _require_approval(current, approval_id)
        return store.cancel(campaign_id, session_id)


@campaign_app.command("explain")
def explain_command(
    campaign_id: Annotated[str, typer.Argument(help="Canonical campaign ID")],
    state: Annotated[Path, typer.Option("--state", help="Campaign SQLite state path")],
    session_id: Annotated[str, typer.Option("--session-id", help="Trusted session identity")],
    field: Annotated[list[str] | None, typer.Option("--field")] = None,
    outcome: Annotated[list[str] | None, typer.Option("--outcome")] = None,
    expected_state_digest: Annotated[str | None, typer.Option("--expected-state-digest")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Render an explanation from the canonical evidence query boundary."""

    result = _run(
        lambda: _explain(
            state,
            campaign_id,
            session_id,
            tuple(field or ()),
            tuple(outcome or ()),
            expected_state_digest,
        ),
        output_json=output_json,
    )
    _emit(result, output_json=output_json)


def _explain(
    path: Path,
    campaign_id: str,
    session_id: str,
    fields: tuple[str, ...],
    outcomes: tuple[str, ...],
    expected_state_digest: str | None,
) -> dict[str, object]:
    parsed_outcomes = tuple(
        sorted({EvidenceQueryOutcome(item) for item in outcomes}, key=lambda item: item.value)
    )
    with CampaignStateStore(path) as store:
        current = _state(store, campaign_id, session_id)
        snapshot = EvidenceSnapshot.from_store(store, campaign_id)
        request = EvidenceQuery(
            campaign_id,
            fields=fields,
            outcomes=parsed_outcomes,
            expected_state_digest=expected_state_digest,
        )
        response = EvidenceQueryEngine(snapshot).query_current(store, request)
    explanation = render_claim_evidence(response)
    record = explanation.to_record()
    record["campaign_state"] = current.to_record()
    record["query_status"] = response.status.value
    if response.status is EvidenceQueryStatus.PARTIAL:
        record["limitations"] = {
            "missing_fields": list(response.missing_fields),
            "unavailable_fields": list(response.unavailable_fields),
        }
    return record


__all__ = ["campaign_app"]
