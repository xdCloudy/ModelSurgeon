"""Experimental, bounded conversational CLI entry point."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.conversation import ChatSessionError, ChatTurnResult, bootstrap_chat_session
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
        typer.echo("execution: not requested by modelsurgeon chat")


def chat_command(
    model: Annotated[
        Path,
        typer.Argument(help="Local text-model path; the initial chat slice accepts GGUF only"),
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
        typer.Option("--runtime-revision", help="Pinned llama-cpp-python revision"),
    ] = None,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", min=1, max=64, help="Maximum number of bounded requests"),
    ] = 8,
    max_input_tokens: Annotated[
        int,
        typer.Option("--max-input-tokens", min=1, help="Per-request input-token ceiling"),
    ] = 2048,
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
) -> None:
    """Start an experimental interpretation-only chat session."""

    try:
        session = bootstrap_chat_session(
            model,
            provider_kind=provider,
            model_revision=model_revision,
            runtime_revision=runtime_revision,
            max_turns=max_turns,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_wall_seconds=max_wall_seconds,
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
            _render_turn(session.interpret(item), output_json=output_json)
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
                _render_turn(session.interpret(prompt), output_json=output_json)
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
