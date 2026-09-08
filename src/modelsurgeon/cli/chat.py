"""Experimental, bounded conversational CLI entry point."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.config_io import load_settings
from modelsurgeon.conversation import (
    ChatOptimizeAdapter,
    ChatProgressEvent,
    ChatSessionError,
    ChatTurnResult,
    bootstrap_chat_session,
)
from modelsurgeon.experiments import ApprovalReuse
from modelsurgeon.provider_kind import ProviderKind


def _error_outcome(error: ChatSessionError) -> str:
    if error.code in {"unsupported_architecture", "provider_unavailable", "provider_unsupported"}:
        return "unknown"
    if error.code.startswith("unsupported_"):
        return "unsupported"
    return "failed"


def _emit(value: Mapping[str, object], *, output_json: bool, message: str | None = None) -> None:
    if output_json:
        typer.echo(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    elif message is not None:
        typer.echo(message)


def _render_turn(turn: ChatTurnResult, *, output_json: bool) -> None:
    if output_json:
        _emit(turn.to_record(), output_json=True)
        return
    result = turn.provider_result
    typer.echo(f"{turn.outcome} request={turn.request_id}")
    if result.failure is not None:
        typer.echo(f"reason: {result.failure.detail}")
    if turn.policy_decision is not None:
        typer.echo("interpreted OptimizationSpec preview (validated before any execution):")
        if turn.spec_preview is not None:
            typer.echo(turn.spec_preview.render())
        else:
            typer.echo(turn.policy_decision.canonical_json())
        if turn.execution is None:
            typer.echo("execution: not requested by modelsurgeon chat")
        else:
            execution = turn.execution
            typer.echo(
                f"execution: {execution.outcome.value} "
                f"plan={execution.plan_id or 'none'} "
                f"campaign={execution.campaign_id or 'none'}"
            )
            if execution.run_id is not None:
                typer.echo(f"run: {execution.run_id}")
            if execution.artifact_id is not None:
                typer.echo(f"artifact: {execution.artifact_id}")


def _render_progress(event: ChatProgressEvent, *, output_json: bool) -> None:
    if output_json:
        _emit(event.to_record(), output_json=True)
    else:
        typer.echo(f"progress: {event.stage} {event.status.value} outcome={event.outcome.value}")


def chat_command(
    model: Annotated[
        Path,
        typer.Argument(help="Local GGUF text-model path"),
    ],
    provider: Annotated[
        ProviderKind,
        typer.Option("--provider", help="Chat provider kind; only local and none are supported"),
    ] = ProviderKind.LOCAL,
    model_revision: Annotated[
        str | None,
        typer.Option("--model-revision", help="Expected SHA-256 revision of the local GGUF"),
    ] = None,
    runtime_revision: Annotated[
        str | None,
        typer.Option("--runtime-revision", help="Pinned local text-runtime revision"),
    ] = None,
    runtime_executable: Annotated[
        Path | None,
        typer.Option(
            "--runtime-executable",
            help="Explicit llama-cli executable for an external local GGUF runtime",
        ),
    ] = None,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", min=1, max=64, help="Maximum number of bounded requests"),
    ] = 8,
    max_input_tokens: Annotated[
        int,
        typer.Option("--max-input-tokens", min=1, help="Per-request input-token ceiling"),
    ] = 6144,
    max_output_tokens: Annotated[
        int,
        typer.Option("--max-output-tokens", min=1, help="Per-request output-token ceiling"),
    ] = 1024,
    max_wall_seconds: Annotated[
        float,
        typer.Option("--max-wall-seconds", min=0.001, help="Per-request wall-time ceiling"),
    ] = 60.0,
    request: Annotated[
        list[str] | None,
        typer.Option("--request", help="Process a request without entering interactive input"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit canonical JSON records"),
    ] = False,
    target_config: Annotated[
        Path | None,
        typer.Option("--target-config", help="Optimization settings for the target model"),
    ] = None,
    target_model: Annotated[
        str | None,
        typer.Option("--target-model", help="Target model path or immutable identifier"),
    ] = None,
    target_revision: Annotated[
        str | None,
        typer.Option("--target-revision", help="Immutable target model revision"),
    ] = None,
    preview_plan: Annotated[
        bool,
        typer.Option(
            "--preview-plan", help="Call the stable optimize planner after interpretation"
        ),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Execute an explicitly approved optimize plan"),
    ] = False,
    state: Annotated[
        Path | None,
        typer.Option("--state", help="Durable canonical campaign state database"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume the matching interrupted optimize workflow"),
    ] = False,
    approval_id: Annotated[
        str | None,
        typer.Option("--approval-id", help="Explicit approval bound to the exact interpreted spec"),
    ] = None,
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Approve a stable optimize plan boundary; may be repeated"),
    ] = None,
    approval_expires_at: Annotated[
        str | None,
        typer.Option("--approval-expires-at", help="Expiry for newly recorded approvals"),
    ] = None,
    approval_reuse: Annotated[
        list[str] | None,
        typer.Option(
            "--approval-reuse",
            help="Approval reuse as code=one_time or code=reusable; may be repeated",
        ),
    ] = None,
    preset: Annotated[
        str,
        typer.Option(help="Bounded optimization preset: fast, balanced, or quality"),
    ] = "balanced",
    hardware_profile: Annotated[
        str,
        typer.Option("--hardware-profile", help="Hardware envelope identifier"),
    ] = "cpu-small",
    quality_profile: Annotated[
        str | None,
        typer.Option("--quality-profile", help="Quality target profile"),
    ] = None,
) -> None:
    """Start an experimental interpretation-only chat session."""

    execution_adapter: ChatOptimizeAdapter | None = None
    if preview_plan or execute:
        overrides: dict[str, object] = {}
        if target_model is not None:
            overrides["model.path"] = target_model
        if target_revision is not None:
            overrides["model.revision"] = target_revision
        try:
            reuse_values: dict[str, str] = {}
            for item in approval_reuse or []:
                name, separator, value = item.partition("=")
                if not separator or not name.strip():
                    raise ValueError(
                        "--approval-reuse values must use code=one_time or code=reusable"
                    )
                reuse_values[name.strip()] = ApprovalReuse(value.strip()).value
            settings = load_settings(target_config, cli_overrides=overrides)
            execution_adapter = ChatOptimizeAdapter(
                settings,
                state_path=state,
                preset=preset,
                hardware_profile=hardware_profile,
                quality_profile=quality_profile,
                approvals=tuple(approve or ()),
                approval_expires_at=approval_expires_at,
                approval_reuse=reuse_values,
                resume=resume,
            )
        except (OSError, ValueError) as error:
            payload = {
                "record_type": "error",
                "category": "chat",
                "outcome": "failed",
                "code": "execution_configuration",
                "message": str(error),
            }
            _emit(payload, output_json=output_json)
            if not output_json:
                typer.echo(f"chat error: {error}", err=True)
            raise typer.Exit(2) from error
    try:
        session = bootstrap_chat_session(
            model,
            provider_kind=provider,
            model_revision=model_revision,
            runtime_revision=runtime_revision,
            runtime_executable=runtime_executable,
            max_turns=max_turns,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_wall_seconds=max_wall_seconds,
            execution_adapter=execution_adapter,
        )
    except (ChatSessionError, OSError, ValueError) as error:
        payload = {
            "record_type": "error",
            "category": "chat",
            "outcome": _error_outcome(error) if isinstance(error, ChatSessionError) else "failed",
            "code": error.code if isinstance(error, ChatSessionError) else "chat_error",
            "message": str(error),
        }
        _emit(payload, output_json=output_json)
        if not output_json:
            typer.echo(f"chat error: {error}", err=True)
        raise typer.Exit(2) from error

    try:
        _emit(
            session.bootstrap.to_record(),
            output_json=output_json,
            message=(
                f"chat session={session.bootstrap.session_id} "
                f"provider={session.provider.identity.provider_id} "
                f"model={session.provider.identity.model_id}"
            ),
        )
        requests = tuple(request or ())
        if len(requests) > session.bootstrap.max_turns:
            raise ChatSessionError("turn_budget", "--request count exceeds --max-turns")
        for item in requests:
            turn = session.interpret(item)
            if preview_plan or execute:
                turn = session.preview_plan(turn)
            if execute:
                turn = session.execute_plan(
                    turn,
                    approval_id,
                    resume=resume,
                    progress_callback=lambda event: _render_progress(
                        event, output_json=output_json
                    ),
                )
            _render_turn(turn, output_json=output_json)
        if requests:
            return
        if not output_json:
            typer.echo("Enter /exit to leave. No optimization or mutation is performed.")
        while session.turns_used < session.bootstrap.max_turns:
            try:
                prompt = input() if output_json else typer.prompt("you", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                session.cancel()
                if not output_json:
                    typer.echo("chat cancelled")
                raise typer.Exit(130) from None
            if prompt.strip().lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            try:
                turn = session.interpret(prompt)
                if preview_plan or execute:
                    turn = session.preview_plan(turn)
                if execute:
                    turn = session.execute_plan(
                        turn,
                        approval_id,
                        resume=resume,
                        progress_callback=lambda event: _render_progress(
                            event, output_json=output_json
                        ),
                    )
                _render_turn(turn, output_json=output_json)
            except ChatSessionError as error:
                payload = {
                    "record_type": "error",
                    "category": "chat",
                    "outcome": _error_outcome(error),
                    "code": error.code,
                    "message": str(error),
                }
                _emit(payload, output_json=output_json)
                raise typer.Exit(2) from error
        else:
            if not output_json:
                typer.echo("chat turn budget exhausted")
    except ChatSessionError as error:
        payload = {
            "record_type": "error",
            "category": "chat",
            "outcome": _error_outcome(error),
            "code": error.code,
            "message": str(error),
        }
        _emit(payload, output_json=output_json)
        if not output_json:
            typer.echo(f"chat error: {error}", err=True)
        raise typer.Exit(2) from error
    finally:
        session.close()


__all__ = ["chat_command"]
