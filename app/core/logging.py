"""DevGuide logging configuration."""

import logging
from typing import Optional, Union

LOGGER_NAME = "devguide"


def configure_logging(level: Optional[Union[int, str]] = None) -> None:
    """Configure simple, consistent standard library logging for DevGuide.

    Ensures a sensible default logging level and readable format without
    introducing complex structured logging dependencies or overriding
    existing handlers configured by the host environment.

    Security & Privacy:
    Never log API keys, uploaded file bytes, raw document contents,
    or complete LLM prompts.
    """
    log_level = level if level is not None else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a named logger instance for the application."""
    return logging.getLogger(name)
