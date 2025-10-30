# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Fake Launcher that returns random results."""

from random import choice, random
from typing import TYPE_CHECKING

from dvsim.launcher.base import ErrorMessage, Launcher

if TYPE_CHECKING:
    from dvsim.job.data import JobSpec, WorkspaceConfig


__all__ = ("FakeLauncher",)


def _run_test_handler(_: "JobSpec") -> str:
    """Handle a RunTest deploy job."""
    return choice(("P", "F"))


def _cov_report_handler(job_spec: "JobSpec") -> str:
    """Handle a CompileSim deploy job."""
    keys = [
        "score",
        "line",
        "cond",
        "toggle",
        "fsm",
        "branch",
        "assert",
        "group",
    ]

    job_spec.cov_results_dict = {k: f"{random() * 100:.2f} %" for k in keys}

    return "P"


_DEPLOY_HANDLER = {
    "RunTest": _run_test_handler,
    "CovReport": _cov_report_handler,
}


class FakeLauncher(Launcher):
    """Launch jobs and return fake results."""

    # Poll job's completion status every this many seconds
    poll_freq = 0

    def _do_launch(self) -> None:
        """Do the launch."""

    def poll(self) -> str | None:
        """Check status of the running process."""
        deploy_cls = self.deploy.job_type
        if deploy_cls in _DEPLOY_HANDLER:
            return _DEPLOY_HANDLER[deploy_cls](job_spec=self.deploy)

        # Default result is Pass
        return "P"

    def kill(self) -> None:
        """Kill the running process."""
        self._post_finish(
            "K",
            ErrorMessage(line_number=None, message="Job killed!", context=[]),
        )

    @staticmethod
    def prepare_workspace(cfg: "WorkspaceConfig") -> None:
        """Prepare the workspace based on the chosen launcher's needs.

        This is done once for the entire duration for the flow run.

        Args:
            cfg: workspace configuration

        """

    @staticmethod
    def prepare_workspace_for_cfg(cfg: "WorkspaceConfig") -> None:
        """Prepare the workspace for a cfg.

        This is invoked once for each cfg.

        Args:
            cfg: workspace configuration

        """
