"""
Application logging configuration.
"""

import sys
import os
from loguru import logger

os.makedirs("logs", exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    format="{time} | {level} | {message}",
    level="INFO"
)

logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)