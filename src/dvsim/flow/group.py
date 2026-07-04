# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Grouping and cross-cfg orchestration of flow configurations.

A primary config file (an hjson with a `use_cfgs` key) groups several child
configs to be run together; a standalone config forms a degenerate group of
one. The `FlowGroup` owns everything that spans configs — selection, job
aggregation and the scheduler run — while each config handles its own flow
logic. There is no flow base class: configs satisfy the `Flow` protocol.
"""

import asyncio
import json
import os
import sys
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from dvsim.job.data import CompletedJobStatus, JobSpec
from dvsim.job.status import JobStatus
from dvsim.logging import log
from dvsim.scheduler.resources import UnknownResourcePolicy
from dvsim.scheduler.runner import (
    build_default_scheduler_backend,
    build_resource_manager,
    run_scheduler,
)

__all__ = (
    "Flow",
    "FlowGroup",
)


class Flow(Protocol):
    """The contract a flow configuration provides to the group.

    Flows also share an informal bootstrap contract - see
    `dvsim.flow.bootstrap`.
    """

    name: str
    flow_cfg_file: str
    is_primary_cfg: bool
    errors_seen: bool
    deploy: Sequence[JobSpec]

    def print_list(self) -> None:
        """Print the list of available items that can be kicked off."""
        ...

    def purge(self) -> None:
        """Purge the scratch area in preparation for the new run."""
        ...

    def create_deploy_objects(self) -> None:
        """Create the job specs for the items slated to be run."""
        ...

    def fake_policy(self, job: JobSpec) -> JobStatus | None:
        """Tell the fake backend how to fake a job, or None if no opinion."""
        ...

    def gen_results(
        self,
        results: Sequence[CompletedJobStatus],
        cfgs: Sequence["Flow"],
    ) -> None:
        """Generate the flow results (and, for a primary cfg, the summary)."""
        ...


class FlowGroup:
    """A group of flow configurations that are run together."""

    def __init__(self, primary: Flow, cfgs: Sequence[Flow], args: Namespace) -> None:
        """Initialise the flow group.

        Args:
            primary: the config the tool was pointed at. For a primary cfg
                this only groups the child cfgs and generates the results
                summary; for a standalone cfg it is the (only) cfg itself.
            cfgs: the configs to run.
            args: arguments passed to dvsim.

        """
        self.primary = primary
        self.cfgs = list(cfgs)
        self.args = args

    def print_list(self) -> None:
        """Print the list of available items that can be kicked off."""
        for cfg in self.cfgs:
            cfg.print_list()

    def purge(self) -> None:
        """Purge the existing scratch areas in preparation for the new run."""
        for cfg in self.cfgs:
            cfg.purge()

    def _prune_selected_cfgs(self) -> None:
        """Prune the list of configs for a primary config file."""
        # This should run after self.cfgs has been set
        assert self.cfgs

        # If the user didn't pass --select-cfgs, we don't do anything.
        if self.args.select_cfgs is None:
            return

        # If the user passed --select-cfgs, but this isn't a primary config
        # file, we should probably complain.
        if not self.primary.is_primary_cfg:
            log.error(
                f"The configuration file at {self.primary.flow_cfg_file!r} is not a primary "
                "config, but --select-cfgs was passed on the command "
                "line.",
            )
            sys.exit(1)

        # Filter configurations
        self.cfgs = [c for c in self.cfgs if c.name in self.args.select_cfgs]

    def create_deploy_objects(self) -> None:
        """Create the job specs for all configs in the group."""
        self._prune_selected_cfgs()

        # GUI, GUI debug or Interactive mode is allowed only for one cfg.
        gui_or_interactive = self.args.gui or self.args.gui_debug or self.args.interactive
        if gui_or_interactive and len(self.cfgs) > 1:
            log.fatal("In GUI mode, only one cfg can be run.")
            sys.exit(1)

        for cfg in self.cfgs:
            cfg.create_deploy_objects()

    def deploy_objects(self) -> Sequence[CompletedJobStatus]:
        """Deploy all the job specs in the group.

        Runs each job and returns the completed job statuses.
        """
        jobs: list[JobSpec] = []
        for cfg in self.cfgs:
            jobs.extend(cfg.deploy)

        if not jobs:
            log.error("Nothing to run!")
            sys.exit(1)

        if os.environ.get("DVSIM_DEPLOY_DUMP", "true"):
            filename = f"deploy_{self.args.branch}_{self.args.timestamp}.json"
            (Path(self.args.scratch_root) / filename).write_text(
                json.dumps(
                    # Sort on full name to ensure consistent ordering
                    sorted(
                        [
                            j.model_dump(
                                # callback functions can't be serialised
                                exclude={"pre_launch", "post_finish"},
                                mode="json",
                            )
                            for j in jobs
                        ],
                        key=lambda j: j["full_name"],
                    ),
                    indent=2,
                ),
            )

        backend = build_default_scheduler_backend(
            fake_policy=self._fake_policy,
        )

        # TODO: For Python 3.11 make this a StrEnum, then this conversion is not needed.
        missing_policy = UnknownResourcePolicy(self.args.on_missing_resource)
        resource_manager = build_resource_manager(
            resource_limits=dict(self.args.resource_limits or ()),
            missing_policy=missing_policy,
        )

        return asyncio.run(
            run_scheduler(
                jobs=jobs,
                max_parallel=self.args.max_parallel,
                interactive=self.args.interactive,
                backend=backend,
                resource_manager=resource_manager,
            )
        )

    def _fake_policy(self, job: JobSpec) -> JobStatus:
        """Tell the fake backend how to fake jobs.

        The first config with an opinion on the job decides; jobs no config
        has an opinion on pass.
        """
        for cfg in self.cfgs:
            status = cfg.fake_policy(job)
            if status is not None:
                return status

        return JobStatus.PASSED

    def gen_results(self, results: Sequence[CompletedJobStatus]) -> None:
        """Generate flow results.

        Args:
            results: completed job status objects.

        """
        self.primary.gen_results(results, self.cfgs)

    def has_errors(self) -> bool:
        """Return error state."""
        return self.primary.errors_seen
