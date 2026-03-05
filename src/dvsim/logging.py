# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Logging config."""

import logging
from pathlib import Path
from typing import cast

from logzero import DEFAULT_FORMAT, setup_logger

__all__ = (
    "configure_logging",
    "LOG_LEVELS",
)


LOG_LEVELS = ["DEBUG", "VERBOSE", "INFO", "WARNING", "ERROR", "CRITICAL"]


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

    def set_logfile(
        self,
        path: Path,
        *,
        level: int | None = None,
        mode: str = "w",
    ) -> None:
        """Set a logfile to save the logs to."""
        fh = logging.FileHandler(filename=path, mode=mode)

        fh.setLevel(level or self.DEBUG)
        fh.setFormatter(
            logging.Formatter(DEFAULT_FORMAT),
        )

        self.addHandler(fh)


def _build_logger() -> DVSimLogger:
    """Build a DVSim logger."""
    logging.setLoggerClass(DVSimLogger)

    return cast("DVSimLogger", setup_logger("dvsim"))


# Logger to import
log = _build_logger()


def configure_logging(*, verbose: bool, debug: bool, log_level: str | None, log_file: str) -> None:
    """Configure the Logger.

    Explicitly setting the log_level takes precedence. But if this is not set
    then the debug and verbose flags are checked in that order. The default
    logging level is INFO.
    """
    if log_level and log_level in LOG_LEVELS:
        new_log_level: int = getattr(log, log_level)
    elif debug:
        new_log_level = log.DEBUG
    elif verbose:
        new_log_level = log.VERBOSE
    else:
        new_log_level: int = log.INFO

    log.setLevel(new_log_level)

    # Push the same level down to the existing stream handler
    for handler in log.handlers:
        handler.setLevel(new_log_level)

    if log_file:
        log.set_logfile(
            path=Path(log_file),
            level=log_level,
        )
