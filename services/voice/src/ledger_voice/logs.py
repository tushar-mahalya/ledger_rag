"""Logging policy: no transcripts, answers or audio in logs.

Pipecat logs transcripts and TTS text at DEBUG through loguru, whose default sink
is DEBUG. ``configure_logging`` replaces that sink with one at INFO or above, and
the service's own events log counts and timings only.
"""

import sys

from loguru import logger

MINIMUM_LEVEL = "INFO"
_LEVELS = {"INFO": 20, "WARNING": 30, "ERROR": 40}


def configure_logging(level: str = MINIMUM_LEVEL) -> int:
    """Install the only loguru sink; refuse levels that would emit DEBUG content."""
    if level.upper() not in _LEVELS:
        raise ValueError(f"Voice log level must be one of {sorted(_LEVELS)}")
    logger.remove()
    return logger.add(sys.stderr, level=level.upper(), backtrace=False, diagnose=False)
