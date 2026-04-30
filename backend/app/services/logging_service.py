"""
Structured logging configuration.

Uses `structlog` to emit JSON logs in production and human-readable
logs in development. Every agent / RAG / guardrail call should obtain
a logger via `get_logger(__name__)`.

Windows note:
  We deliberately route structlog through stdlib `logging` (not
  `PrintLoggerFactory`) because the latter calls `print(..., flush=True)`
  which, combined with colorama's stdout wrapper on Windows, can raise
  `OSError: [Errno 22] Invalid argument` mid-request. Routing through
  stdlib logging means flush errors are swallowed by the logging
  framework instead of propagating up into the FastAPI dependency
  layer and surfacing as 500s to the user.

  We also disable colorised output on Windows for the same reason.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import AppEnv, get_settings


def configure_logging() -> None:
    """Configure stdlib logging + structlog. Call once at startup."""
    settings = get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    is_windows = sys.platform.startswith("win")

    if settings.app_env == AppEnv.DEVELOPMENT and not is_windows:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    elif settings.app_env == AppEnv.DEVELOPMENT:
        # Plain (no-colour) console rendering for Windows -- avoids the
        # colorama stdout-flush crash described above.
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            renderer,
        ],
        # stdlib LoggerFactory routes through Python's `logging`, which
        # absorbs stream errors (like Windows EINVAL on flush) instead
        # of crashing the request handler.
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to `name`."""
    return structlog.get_logger(name)
