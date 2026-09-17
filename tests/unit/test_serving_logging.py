from __future__ import annotations

import logging

from relevanceflow.serving.logging import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("relevanceflow.test")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "relevanceflow.test"
