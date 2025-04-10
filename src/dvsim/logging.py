# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Logging config."""

import logging
from typing import cast

from logzero import setup_logger

__all__ = ("configure_logging",)


class DVSimLogger(logging.getLoggerClass()):
    """Logger class for DVSim."""

    # Log level for verbose logging between INFO (10) and DEBUG (20)
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    VERBOSE = 15
    DEBUG = logging.DEBUG

    def __init__(self, name: str) -> None:
        """Initialise a logger."""
        super().__init__(name=name)

        logging.addLevelName(DVSimLogger.VERBOSE, "VERBOSE")

    def verbose(self, msg: object, *args: object) -> None:
        """Log a verbose msg."""
        self.log(self.VERBOSE, msg, *args)


def _build_logger() -> DVSimLogger:
    """Build a DVSim logger."""
    logging.setLoggerClass(DVSimLogger)

    return cast("DVSimLogger", setup_logger("dvsim"))


# Logger to import
log = _build_logger()
VERBOSE = DVSimLogger.VERBOSE


def configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure the Logger."""
    # Add log level 'VERBOSE' between INFO and DEBUG
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = log.VERBOSE
    else:
        log_level = logging.INFO

    log.setLevel(log_level)
