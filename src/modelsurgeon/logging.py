"""Explicit structured logging configuration and contextual binding."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

import structlog


class LogFormat(StrEnum):
    HUMAN = "human"
    JSON = "json"


def configure_logging(
    level: str = "INFO",
    output_format: LogFormat = LogFormat.HUMAN,
) -> None:
    """Configure application logging when explicitly called by an entry point."""
    normalized_level = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized_level)
    if numeric_level is None:
        allowed = ", ".join(sorted(logging.getLevelNamesMapping()))
        raise ValueError(f"unknown log level {level!r}; expected one of: {allowed}")

    logging.basicConfig(format="%(message)s", level=numeric_level, force=True)
    renderer: structlog.types.Processor
    if output_format is LogFormat.JSON:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def bind_log_context(
    *,
    run_id: str | None = None,
    model_id: str | None = None,
    component_id: str | None = None,
) -> None:
    """Bind non-null experiment context to the current context-local scope."""
    values = {
        "run_id": run_id,
        "model_id": model_id,
        "component_id": component_id,
    }
    structlog.contextvars.bind_contextvars(
        **{name: value for name, value in values.items() if value is not None}
    )


def clear_log_context() -> None:
    """Remove all context-local logging values."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(
    *,
    run_id: str | None = None,
    model_id: str | None = None,
    component_id: str | None = None,
) -> Iterator[None]:
    """Temporarily bind run/model/component context and restore prior values."""
    tokens = structlog.contextvars.bind_contextvars(
        **{
            name: value
            for name, value in {
                "run_id": run_id,
                "model_id": model_id,
                "component_id": component_id,
            }.items()
            if value is not None
        }
    )
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)

