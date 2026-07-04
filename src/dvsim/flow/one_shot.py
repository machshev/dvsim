# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Class describing a one-shot build configuration object."""

import argparse
import pprint
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from dvsim.flow.bootstrap import (
    default_rel_path,
    expand_wildcards,
    finalize_flow_state,
    init_flow_state,
    merge_hjson,
    process_overrides,
)
from dvsim.job.data import CompletedJobStatus, JobSpec
from dvsim.job.factory import (
    construct_job_cmd,
    job_full_name,
    job_namespace,
    new_job_spec,
    resolve_wildcards,
)
from dvsim.job.status import JobStatus
from dvsim.logging import log
from dvsim.modes import BuildMode
from dvsim.utils import rm_path


def create_compile_one_shot_job(build_mode: BuildMode, sim_cfg: "OneShotCfg") -> JobSpec:
    """Create a job spec for building the design (used by non-DV flows).

    Args:
        build_mode: build mode instance
        sim_cfg: flow config object

    Returns:
        the job spec for the build job.

    """
    target = "build"
    cmd_attrs = (
        # tool srcs
        "proj_root",
        # Flist gen
        "sv_flist_gen_cmd",
        "sv_flist_gen_dir",
        "sv_flist_gen_opts",
        # Build
        "build_dir",
        "build_cmd",
        "build_opts",
        "build_log",
        "build_timeout_mins",
        "post_build_cmds",
        "post_build_opts",
        "pre_build_cmds",
        # Report processing
        "report_cmd",
        "report_opts",
    )

    name = build_mode.name
    ns = job_namespace(
        sim_cfg,
        attrs=(*cmd_attrs, "build_fail_patterns", "build_pass_patterns"),
        mode_dict=build_mode.__dict__,
        # 'build_mode' is used as a substitution variable in the HJson.
        build_mode=name,
        name=name,
        qual_name=name,
        full_name=job_full_name(sim_cfg, name),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}_{name}",
        odir="{build_dir}",
    )

    if sim_cfg.args.build_timeout_mins is not None:
        ns["build_timeout_mins"] = sim_cfg.args.build_timeout_mins

    timeout_mins = ns["build_timeout_mins"]
    if timeout_mins:
        log.debug('Timeout for job "%s" is %d minutes.', name, timeout_mins)

    return new_job_spec(
        sim_cfg,
        job_type="CompileOneShot",
        target=target,
        name=name,
        qual_name=name,
        cmd=construct_job_cmd(
            makefile=resolve_wildcards(ns["flow_makefile"], ns),
            target=target,
            dry_run=ns["dry_run"],
            cmd_attrs={attr: resolve_wildcards(ns[attr], ns) for attr in cmd_attrs},
        ),
        odir=resolve_wildcards(ns["odir"], ns),
        gui=sim_cfg.gui,
        dry_run=ns["dry_run"],
        exports=resolve_wildcards(ns["exports"], ns),
        # Limit build jobs to 60 minutes if the timeout is not set.
        timeout_mins=timeout_mins if timeout_mins is not None else 60,
        pass_patterns=resolve_wildcards(ns["build_pass_patterns"], ns),
        fail_patterns=resolve_wildcards(ns["build_fail_patterns"], ns),
    )


class OneShotCfg(ABC):
    """Simple one-shot build flow for non-simulation targets like linting, synthesis and FPV."""

    # The hjson 'flow' key that selects this flow config; set in subclasses.
    flow: ClassVar[str | None] = None

    ignored_wildcards: ClassVar[list[str]] = [
        "build_mode",
        "index",
        "test",
    ]

    def __str__(self) -> str:
        """Get string representation of the flow config."""
        return pprint.pformat(self.__dict__)

    def __init__(
        self,
        flow_cfg_file: str,
        hjson_data: Mapping[str, Any],
        args: argparse.Namespace,
    ) -> None:
        """Initialise OneShotCfg object.

        Args:
          flow_cfg_file: Path to hjson cfg that was loaded.
          hjson_data: The parsed hjson from flow_cfg_file.
          args: Arguments passed to dvsim

        """
        # Options set from command line
        self.verbose = args.verbose
        self.flist_gen_cmd = ""
        self.flist_gen_opts = []
        self.sv_flist_gen_dir = ""
        self.flist_file = ""
        self.build_cmd = ""
        self.build_opts = []
        self.build_log = ""
        self.report_cmd = ""
        self.report_opts = []
        self.build_opts.extend(args.build_opts)
        self.build_unique = args.build_unique
        self.build_only = args.build_only

        # Options built from cfg_file files
        self.project = ""
        self.flow = ""
        self.flow_makefile = ""
        self.scratch_path = ""
        self.build_dir = ""
        self.run_dir = ""
        self.pass_patterns = []
        self.fail_patterns = []
        self.name = ""
        self.dut = ""
        self.fusesoc_core = ""
        self.ral_spec = ""
        self.build_modes: Sequence[BuildMode] = []
        self.run_modes = []
        self.regressions = []
        self.max_msg_count = -1

        # Flow results
        self.result = OrderedDict()
        self.result_summary = OrderedDict()

        self.dry_run = args.dry_run

        # Not needed for this build
        self.verbosity = ""
        self.en_build_modes = []

        # Generated data structures
        self.build_list = []
        self.deploy = []
        self.cov = args.cov

        init_flow_state(self, flow_cfg_file, args)

        # If build_unique is set, then add current timestamp to uniquify it
        if self.build_unique:
            self.build_dir += "_" + self.timestamp

        # Merge in the values from the loaded hjson file.
        merge_hjson(self, hjson_data)

        # A primary cfg simply groups child cfgs (see dvsim.flow.group).
        self.is_primary_cfg = "use_cfgs" in hjson_data

        default_rel_path(self)

        # Process overrides before substituting wildcards.
        process_overrides(self)

        # Expand wildcards.
        expand_wildcards(self)

        # Stuff below only pertains to individual cfg (not primary cfg).
        if not self.is_primary_cfg and (not self.select_cfgs or self.name in self.select_cfgs):
            # Print scratch_path at the start:
            log.info("[scratch_path]: [%s] [%s]", self.name, self.scratch_path)

            # Use the default build mode for tests that do not specify it
            if not hasattr(self, "build_mode"):
                self.build_mode = "default"

            # Create objects from raw dicts - build_modes, sim_modes, run_modes,
            # tests and regressions, only if not a primary cfg obj
            self._create_objects()

        finalize_flow_state(self)

    def wildcard_namespace(self) -> Mapping:
        """Return the flat mapping used for wildcard substitution.

        This combines all config and runtime state, which for this flow is
        the instance `__dict__`.
        """
        return self.__dict__

    def apply_expansion(self, expanded: Mapping) -> None:
        """Store the result of wildcard expansion of `wildcard_namespace()`."""
        self.__dict__ = expanded

    def has_errors(self) -> bool:
        """Return error state."""
        return self.errors_seen

    def fake_policy(self, _job: JobSpec) -> JobStatus | None:
        """Tell the fake backend how to fake a job. No opinion by default."""
        return None

    # Purge the output directories. This operates on self.
    def purge(self) -> None:
        """Purge the scratch area in preparation for the new run."""
        if not self.scratch_path:
            raise RuntimeError("Scratch path is '', so cannot purge.")

        log.info("Purging scratch path %s", self.scratch_path)
        rm_path(Path(self.scratch_path))

    def _create_objects(self) -> None:
        # Create build modes objects based on the names that were stored in
        # self.build_modes by the superclass constructor's call to
        # _merge_hjson.
        #
        # We've lied to the type checker about the type of self.build_modes,
        # giving it the type that it will contain after this function is
        # complete.
        mode_dicts: list[dict[str, object]] = []
        for mode_dict in self.build_modes:
            if not isinstance(mode_dict, dict):
                msg = f"Found a build mode item of {type(mode_dict)} when we expected a dict."
                raise TypeError(msg)

            mode_dicts.append(mode_dict)

        self.build_modes = BuildMode.create_modes(mode_dicts)

        # Extend all of the build modes (that are being built) with the global build_opts.
        for build_mode in self.build_modes:
            build_mode.build_opts.extend(self.build_opts)

    def print_list(self) -> None:
        """Print the list of available items that can be kicked off."""
        for list_item in self.list_items:
            log.info("---- List of %s in %s ----", list_item, self.name)
            if hasattr(self, list_item):
                items = getattr(self, list_item)
                for item in items:
                    log.info(item)
            else:
                log.error("Item %s does not exist!", list_item)

    def create_deploy_objects(self) -> None:
        """Create job specs from build modes."""
        builds = [create_compile_one_shot_job(build, self) for build in self.build_modes]

        self.builds = builds
        self.deploy = builds

    @abstractmethod
    def _gen_results_for_cfg(self, results: Sequence[CompletedJobStatus]) -> None:
        """Generate results for this config."""

    @abstractmethod
    def gen_results_summary(self, cfgs: Sequence["OneShotCfg"]) -> str:
        """Gathers the aggregated results from all sub configs."""

    def gen_results(
        self,
        results: Sequence[CompletedJobStatus],
        cfgs: Sequence["OneShotCfg"],
    ) -> None:
        """Generate flow results.

        Args:
            results: completed job status objects.
            cfgs: the flow configs that were run (see dvsim.flow.group).

        """
        for item in cfgs:
            project = item.name

            # Children of this configuration should all be OneShotCfg objects
            # too. Check that they are.
            if not isinstance(item, OneShotCfg):
                msg = "sub-configs of a OneShotCfg should also be instances of OneShotCfg."
                raise TypeError(msg)

            # We want to report all the results with the correct block
            # (checking the name) and also with the correct variant.
            #
            # For the latter check, we don't necessarily know that item (the
            # configuration that is being run) is a type that defines variant
            # at all. If not, we just interpret the variant as None.
            item_variant = getattr(item, "variant", None)

            item_results = [
                res
                for res in results
                if res.block.name == item.name and res.block.variant == item_variant
            ]

            # Parse results from the parsed results.hjson for the given config.
            # The item_results list has been filtered so that it only contains
            # results from the runs corresponding to the config.
            #
            # The noqa line is to tell the linter that this isn't an external
            # use of an internal method: the prototype comes from this class.
            result = item._gen_results_for_cfg(item_results)  # noqa: SLF001

            log.info("[results]: [%s]:\n%s\n", project, result)
            log.info("[scratch_path]: [%s] [%s]", project, item.scratch_path)

            # TODO: Implement HTML report using templates
            #
            # This was previously implemented by rendering the markdown results for the item.

            results_dir = Path(self.results_dir)
            results_dir.mkdir(exist_ok=True, parents=True)

            log.verbose("[report]: [%s] [%s/report.html]", project, item.results_dir)

            self.errors_seen |= item.errors_seen

        if self.is_primary_cfg:
            self.gen_results_summary(cfgs)
            # TODO: Write a combined HTML report to self.results_html_name
