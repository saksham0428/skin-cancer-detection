"""
app/core/logging.py — Logging configuration for the FastAPI application.
"""

import logging
import sys


def setup_logging(debug: bool = False) -> None:
    """
    Configure root logger for the application.

    Args:
        debug: If True, sets level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
