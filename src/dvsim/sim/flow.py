# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Class describing simulation configuration object."""

import fnmatch
import pprint
import random
import shlex
import shutil
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar

import hjson
from pydantic import ValidationError

from dvsim.flow.bootstrap import (
    default_rel_path,
    expand_wildcards,
    finalize_flow_state,
    init_flow_state,
)
from dvsim.job.data import CompletedJobStatus, JobSpec
from dvsim.job.factory import (
    construct_job_cmd,
    is_equivalent_job_spec,
    job_full_name,
    job_namespace,
    new_job_spec,
    resolve_wildcards,
)
from dvsim.job.status import JobStatus
from dvsim.logging import log
from dvsim.modes import BuildMode, Mode, RunMode, find_mode
from dvsim.regression import Regression
from dvsim.sim.config import SimFlowConfig, load_sim_flow_config
from dvsim.sim.data import (
    IPMeta,
    SimFlowResults,
    SimFlowSummary,
    SimResultsSummary,
    Testpoint,
    TestResult,
    TestStage,
    ToolMeta,
)
from dvsim.sim.report import gen_reports
from dvsim.sim_results import BucketedFailures, SimResults
from dvsim.test import Test
from dvsim.testplan import Testplan
from dvsim.tool.utils import get_sim_tool_plugin
from dvsim.utils import TS_FORMAT, clean_odirs, rm_path, subst_wildcards
from dvsim.utils.fs import relative_to
from dvsim.utils.git import git_https_url_with_commit

__all__ = ("SimCfg",)

# This affects the bucketizer failure report.
_MAX_UNIQUE_TESTS = 5
_MAX_TEST_RESEEDS = 2


# Custom seeds for tests, registered from the command line (--seeds). These
# are consumed first. If --fixed-seed <val> is also passed, the subsequent
# tests (once the custom seeds are consumed) will be run with the fixed seed.
_test_seeds: list[int] = []
_fixed_seed: int | None = None


def set_test_seeds(seeds: list[int] | None, fixed_seed: int | None) -> None:
    """Register the test seeds passed on the command line."""
    global _test_seeds, _fixed_seed  # noqa: PLW0603
    _test_seeds = seeds if seeds is not None else []
    _fixed_seed = fixed_seed


def _get_test_seed() -> int:
    """Get the test random seed."""
    if not _test_seeds:
        if _fixed_seed is not None:
            return _fixed_seed
        _test_seeds.extend(random.getrandbits(256) for _ in range(1000))
    return _test_seeds.pop(0)


def _apply_tool_plugin(ns: dict, sim_cfg: "SimCfg", target: str) -> None:
    """Mutate a job's wildcard namespace based on any tool plugins."""
    try:
        plugin = get_sim_tool_plugin(sim_cfg.tool)
    except NotImplementedError as e:
        log.debug("Could not find sim tool for %s: %s", sim_cfg.tool, str(e))
        return

    plugin.set_additional_attrs(ns, sim_cfg, target)


def create_compile_sim_job(build_mode: BuildMode, sim_cfg: "SimCfg") -> JobSpec:
    """Create a job spec for building the simulation executable.

    Args:
        build_mode: build mode instance
        sim_cfg: simulation config object

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
        "pre_build_cmds",
        "build_cmd",
        "build_dir",
        "build_opts",
        "post_build_cmds",
        "post_build_opts",
    )

    name = build_mode.name
    ns = job_namespace(
        sim_cfg,
        attrs=(
            *cmd_attrs,
            "build_fail_patterns",
            "build_pass_patterns",
            "build_timeout_mins",
            "cov_db_dir",
        ),
        mode_dict=build_mode.__dict__,
        # Dont run the compile job in GUI mode.
        gui=False,
        # 'build_mode' is used as a substitution variable in the HJson.
        build_mode=name,
        name=name,
        seed=sim_cfg.build_seed,
        qual_name=name,
        full_name=job_full_name(sim_cfg, name),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}_{name}",
        odir="{build_dir}",
    )

    if sim_cfg.args.build_timeout_mins is not None:
        ns["build_timeout_mins"] = sim_cfg.args.build_timeout_mins

    _apply_tool_plugin(ns, sim_cfg, target)

    timeout_mins = ns["build_timeout_mins"]
    if timeout_mins:
        log.debug('Timeout for job "%s" is %d minutes.', name, timeout_mins)

    cov_db_dir = Path(resolve_wildcards(ns["cov_db_dir"], ns))

    def pre_launch() -> None:
        """Perform pre-launch tasks."""
        # Delete old coverage database directories before building again. We
        # need to do this because the build directory is not 'renewed'.
        rm_path(cov_db_dir)

    return new_job_spec(
        sim_cfg,
        job_type="CompileSim",
        target=target,
        name=name,
        qual_name=name,
        cmd=construct_job_cmd(
            makefile=resolve_wildcards(ns["flow_makefile"], ns),
            target=target,
            dry_run=ns["dry_run"],
            cmd_attrs={attr: resolve_wildcards(ns[attr], ns) for attr in cmd_attrs},
            cmds_list_vars=("pre_build_cmds", "post_build_cmds"),
        ),
        odir=resolve_wildcards(ns["odir"], ns),
        gui=False,
        dry_run=ns["dry_run"],
        exports=resolve_wildcards(ns["exports"], ns),
        seed=sim_cfg.build_seed,
        weight=5,
        # Limit build jobs to 60 minutes if the timeout is not set.
        timeout_mins=timeout_mins if timeout_mins is not None else 60,
        pass_patterns=resolve_wildcards(ns["build_pass_patterns"], ns),
        fail_patterns=resolve_wildcards(ns["build_fail_patterns"], ns),
        pre_launch=pre_launch,
    )


def create_run_test_job(
    index: int,
    test: Test,
    build_job: JobSpec,
    sim_cfg: "SimCfg",
) -> JobSpec:
    """Create a job spec for running a test. There is one of these per seed.

    Args:
        index: reseed index of this run.
        test: the test to run.
        build_job: the build job this run depends on.
        sim_cfg: simulation config object

    Returns:
        the job spec for the run job.

    """
    target = "run"
    seed = _get_test_seed()
    log.debug(
        "Creating run job for %s test %s no. %d with seed %s",
        sim_cfg.name,
        getattr(test, "name", "[unknown]"),
        index,
        seed,
    )

    extracted_cmd_attrs = (
        # tool srcs
        "proj_root",
        "uvm_test",
        "uvm_test_seq",
        "sw_images",
        "sw_build_device",
        "sw_build_cmd",
        "sw_build_opts",
        "run_dir",
        "pre_run_cmds",
        "run_cmd",
        "run_opts",
        "post_run_cmds",
    )
    # 'build_seed' and 'seed' are set directly below, but are also make
    # variables on the run command.
    cmd_attrs = (*extracted_cmd_attrs, "build_seed", "seed")

    build_mode = test.build_mode.name
    qual_name = "{run_dir_name}." + str(seed)

    ns = job_namespace(
        sim_cfg,
        attrs=(
            *extracted_cmd_attrs,
            "cov_db_dir",
            "cov_db_test_dir",
            "run_dir_name",
            "run_fail_patterns",
            "run_pass_patterns",
            "run_timeout_mins",
            "run_timeout_multiplier",
        ),
        mode_dict=test.__dict__,
        index=index,
        build_seed=sim_cfg.build_seed,
        seed=seed,
        # Systemverilog accepts seeds with a maximum size of 32 bits.
        svseed=int(seed) & 0xFFFFFFFF,
        # 'test' is used as a substitution variable in the HJson.
        test=test.name,
        name=test.name,
        build_mode=build_mode,
        qual_name=qual_name,
        full_name=job_full_name(sim_cfg, qual_name),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}_{build_mode}",
        odir="{run_dir}",
    )

    if sim_cfg.args.run_timeout_mins is not None:
        ns["run_timeout_mins"] = sim_cfg.args.run_timeout_mins

    if sim_cfg.args.run_timeout_multiplier is not None:
        ns["run_timeout_multiplier"] = sim_cfg.args.run_timeout_multiplier

    if ns["run_timeout_mins"] and ns["run_timeout_multiplier"]:
        ns["run_timeout_mins"] = int(ns["run_timeout_mins"] * ns["run_timeout_multiplier"])

    _apply_tool_plugin(ns, sim_cfg, target)

    qual_name = resolve_wildcards(qual_name, ns)
    full_name = job_full_name(sim_cfg, qual_name)

    if ns["run_timeout_multiplier"]:
        log.debug(
            'Timeout multiplier for job "%s" is %f.',
            full_name,
            ns["run_timeout_multiplier"],
        )

    if ns["run_timeout_mins"]:
        log.debug('Timeout for job "%s" is %d minutes.', full_name, ns["run_timeout_mins"])

    # We did something wrong if build_mode is not the same as the build_job
    # arg's name.
    if build_mode != build_job.name:
        msg = (
            f"Created a build job with name {build_job.name}, when we "
            f"expected the name to be {build_mode}."
        )
        raise AssertionError(msg)

    cov_db_test_dir = Path(resolve_wildcards(ns["cov_db_test_dir"], ns))

    def post_finish(status: JobStatus) -> None:
        """Perform tidy up tasks."""
        if status != JobStatus.PASSED:
            # Delete the coverage data if available.
            rm_path(cov_db_test_dir)

    return new_job_spec(
        sim_cfg,
        job_type="RunTest",
        target=target,
        name=test.name,
        qual_name=qual_name,
        cmd=construct_job_cmd(
            makefile=resolve_wildcards(ns["flow_makefile"], ns),
            target=target,
            dry_run=ns["dry_run"],
            cmd_attrs={attr: resolve_wildcards(ns[attr], ns) for attr in cmd_attrs},
            cmds_list_vars=("pre_run_cmds", "post_run_cmds"),
        ),
        odir=resolve_wildcards(ns["odir"], ns),
        gui=sim_cfg.gui,
        dry_run=ns["dry_run"],
        exports=resolve_wildcards(ns["exports"], ns),
        seed=seed,
        dependencies=([build_job] if build_job is not None and not sim_cfg.run_only else []),
        # Limit run jobs to 60 minutes if the timeout is not set.
        timeout_mins=(ns["run_timeout_mins"] if ns["run_timeout_mins"] is not None else 60),
        # When running a test, we should always renew the output directory.
        renew_odir=True,
        # In GUI mode, the log file is not updated; hence, nothing to check.
        pass_patterns=([] if sim_cfg.gui else resolve_wildcards(ns["run_pass_patterns"], ns)),
        fail_patterns=([] if sim_cfg.gui else resolve_wildcards(ns["run_fail_patterns"], ns)),
        post_finish=post_finish,
    )


def create_cov_merge_job(
    run_jobs: Sequence[JobSpec],
    run_build_modes: Sequence[str],
    sim_cfg: "SimCfg",
) -> JobSpec:
    """Create a job spec for merging the coverage databases of the run jobs.

    Args:
        run_jobs: the run jobs whose coverage is to be merged.
        run_build_modes: the build mode name of each run job (in the same
            order as run_jobs).
        sim_cfg: simulation config object

    Returns:
        the job spec for the coverage merge job.

    """
    target = "cov_merge"
    cfg_namespace = sim_cfg.wildcard_namespace()

    # Construct the cov_db_dirs right away from the run jobs. This is a
    # special variable used in the HJson. The coverage associated with
    # the primary build mode needs to be first in the list.
    cov_db_dirs = []
    for build_mode in run_build_modes:
        cov_db_dir = subst_wildcards(
            "{cov_db_dir}",
            {**cfg_namespace, "build_mode": build_mode},
        )
        if cov_db_dir not in cov_db_dirs:
            if sim_cfg.primary_build_mode == build_mode:
                cov_db_dirs.insert(0, cov_db_dir)
            else:
                cov_db_dirs.append(cov_db_dir)

    # Sort the cov_db_dirs except for the first directory.
    if len(cov_db_dirs) > 1:
        cov_db_dirs = [cov_db_dirs[0], *sorted(cov_db_dirs[1:])]

    # Early lookup the cov_merge_db_dir, which is a mandatory misc
    # attribute anyway. We need it to compute additional cov db dirs.
    cov_merge_db_dir = subst_wildcards("{cov_merge_db_dir}", cfg_namespace)

    # Prune previous merged cov directories, keeping past 7 dbs.
    prev_cov_db_dirs = clean_odirs(odir=Path(cov_merge_db_dir), max_odirs=7)

    # If the --cov-merge-previous command line switch is passed, then
    # merge coverage with the previous runs.
    if sim_cfg.cov_merge_previous:
        cov_db_dirs += [str(item) for item in prev_cov_db_dirs]

    cmd_attrs = ("cov_merge_cmd", "cov_merge_opts")
    ns = job_namespace(
        sim_cfg,
        attrs=(*cmd_attrs, "cov_merge_dir", "cov_merge_db_dir"),
        cov_db_dirs=cov_db_dirs,
        qual_name=target,
        full_name=job_full_name(sim_cfg, target),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}",
        # For merging coverage db, the precise output dir is set in the HJson.
        odir="{cov_merge_db_dir}",
    )

    _apply_tool_plugin(ns, sim_cfg, target)

    return new_job_spec(
        sim_cfg,
        job_type="CovMerge",
        target=target,
        name=ns["name"],
        qual_name=target,
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
        weight=10,
        dependencies=run_jobs,
        # Run coverage merge even if just one test passes.
        needs_all_dependencies_passing=False,
        # Append cov_db_dirs to the list of exports.
        extra_exports={"cov_db_dirs": shlex.quote(" ".join(cov_db_dirs))},
    )


def create_cov_report_job(merge_job: JobSpec, sim_cfg: "SimCfg") -> JobSpec:
    """Create a job spec for generating a coverage report.

    Args:
        merge_job: the coverage merge job this one depends on.
        sim_cfg: simulation config object

    Returns:
        the job spec for the coverage report job.

    """
    target = "cov_report"
    cmd_attrs = ("cov_report_cmd", "cov_report_opts")

    ns = job_namespace(
        sim_cfg,
        attrs=(*cmd_attrs, "cov_report_dir", "cov_merge_db_dir", "cov_report_txt"),
        qual_name=target,
        full_name=job_full_name(sim_cfg, target),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}",
        odir="{cov_report_dir}",
    )

    _apply_tool_plugin(ns, sim_cfg, target)

    cov_report_txt = Path(resolve_wildcards(ns["cov_report_txt"], ns))
    dry_run = ns["dry_run"]

    def post_finish(status: JobStatus) -> None:
        """Extract the coverage results summary for the dashboard.

        The results are stored on the cfg (see SimCfg.cov_report_results),
        which is where the report generation looks for them.

        If the extraction fails, an appropriate exception is raised, which must
        be caught by the caller to mark the job as a failure.
        """
        if dry_run or status != JobStatus.PASSED or not cov_report_txt.exists():
            return

        # At this point, we have finished running a tool, so we know that
        # sim_cfg.tool must have been set.
        if sim_cfg.tool is None:
            raise RuntimeError("sim_cfg.tool cannot be None now.")

        plugin = get_sim_tool_plugin(tool=sim_cfg.tool)

        results, _cov_total = plugin.get_cov_summary_table(
            cov_report_path=cov_report_txt,
        )

        sim_cfg.cov_report_results = {tup[0]: tup[1] for tup in zip(*results, strict=False)}

    return new_job_spec(
        sim_cfg,
        job_type="CovReport",
        target=target,
        name=ns["name"],
        qual_name=target,
        cmd=construct_job_cmd(
            makefile=resolve_wildcards(ns["flow_makefile"], ns),
            target=target,
            dry_run=dry_run,
            cmd_attrs={attr: resolve_wildcards(ns[attr], ns) for attr in cmd_attrs},
        ),
        odir=resolve_wildcards(ns["odir"], ns),
        gui=sim_cfg.gui,
        dry_run=dry_run,
        exports=resolve_wildcards(ns["exports"], ns),
        weight=10,
        dependencies=[merge_job],
        post_finish=post_finish,
    )


def create_cov_unr_job(sim_cfg: "SimCfg") -> JobSpec:
    """Create a job spec for the coverage UNR flow.

    Args:
        sim_cfg: simulation config object

    Returns:
        the job spec for the UNR coverage calculation job.

    """
    target = "cov_unr"
    cmd_attrs = (
        # tool srcs
        "proj_root",
        # Need to generate filelist based on build mode
        "sv_flist_gen_cmd",
        "sv_flist_gen_dir",
        "sv_flist_gen_opts",
        "build_dir",
        "cov_unr_build_cmd",
        "cov_unr_build_opts",
        "cov_unr_run_cmd",
        "cov_unr_run_opts",
    )

    ns = job_namespace(
        sim_cfg,
        attrs=(*cmd_attrs, "cov_unr_dir", "cov_merge_db_dir", "build_fail_patterns"),
        qual_name=target,
        full_name=job_full_name(sim_cfg, target),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}",
        odir="{cov_unr_dir}",
    )

    _apply_tool_plugin(ns, sim_cfg, target)

    return new_job_spec(
        sim_cfg,
        job_type="CovUnr",
        target=target,
        name=ns["name"],
        qual_name=target,
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
        # Reuse the build_fail_patterns set in the HJson.
        fail_patterns=resolve_wildcards(ns["build_fail_patterns"], ns),
    )


def create_cov_analyze_job(sim_cfg: "SimCfg") -> JobSpec:
    """Create a job spec for running the coverage analysis tool.

    Args:
        sim_cfg: simulation config object

    Returns:
        the job spec for the coverage analysis job.

    """
    # Enforce GUI mode for coverage analysis.
    sim_cfg.gui = True

    target = "cov_analyze"
    cmd_attrs = (
        # tool srcs
        "proj_root",
        "cov_analyze_cmd",
        "cov_analyze_opts",
    )

    ns = job_namespace(
        sim_cfg,
        attrs=(*cmd_attrs, "cov_analyze_dir", "cov_merge_db_dir"),
        qual_name=target,
        full_name=job_full_name(sim_cfg, target),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}",
        odir="{cov_analyze_dir}",
    )

    _apply_tool_plugin(ns, sim_cfg, target)

    return new_job_spec(
        sim_cfg,
        job_type="CovAnalyze",
        target=target,
        name=ns["name"],
        qual_name=target,
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
    )


def create_cov_vplan_job(report_job: JobSpec, sim_cfg: "SimCfg") -> JobSpec:
    """Create a job spec for generating a Verification Plan report using DVPlan.

    Args:
        report_job: the coverage report job this one depends on.
        sim_cfg: simulation config object

    Returns:
        the job spec for the vPlan report job.

    """
    target = "cov_vplan"

    ns = job_namespace(
        sim_cfg,
        attrs=("proj_root", "vplan", "dut_instance"),
        cov_vplan_dir=f"{sim_cfg.scratch_path}/{target}",
        qual_name=target,
        full_name=job_full_name(sim_cfg, target),
        job_name=f"{Path(sim_cfg.scratch_path).name}_{target}",
        odir="{cov_vplan_dir}",
    )

    _apply_tool_plugin(ns, sim_cfg, target)

    odir = resolve_wildcards(ns["odir"], ns)
    vplan = resolve_wildcards(ns["vplan"], ns)
    dut_instance = resolve_wildcards(ns["dut_instance"], ns)
    prepare_opts = resolve_wildcards(sim_cfg.cov_vplan_prepare_opts, ns)
    process_opts = resolve_wildcards(sim_cfg.cov_vplan_process_opts, ns)

    # Calculate IP root.
    ip_root = str(Path(vplan).parent.parent)

    # Use fixed output filenames so the report location is always predictable.
    annotated_hjson = f"{odir}/vplan_annotated.hjson"
    gen_html = f"{odir}/vplan_annotated.html"

    # Construct the pure bash shell command, bypassing the Makefile
    # convention used by the other job types.
    if shutil.which("dvplan") is None:
        fallback = (
            "echo 'WARNING: dvplan tool not installed in PATH. Skipping vPlan generation.'"
        )
        cmd = f"/usr/bin/env bash -c {shlex.quote(fallback)}"
    else:

        def format_opts(opts: list[str] | str) -> str:
            return " ".join(opts) if isinstance(opts, list) else str(opts)

        prepare_cmd = " ".join(
            f"dvplan prepare_vplan {format_opts(prepare_opts)} "
            f"{ip_root} {vplan} {annotated_hjson}".split(),
        )

        vendor_tool = f"{sim_cfg.tool}_report"
        process_cmd = " ".join(
            f"dvplan process_results {format_opts(process_opts)} "
            f"--coverage {vendor_tool} {report_job.odir} "
            f"-R {gen_html} -s {sim_cfg.name} {dut_instance} {annotated_hjson}".split(),
        )

        full_command = f"set -e; mkdir -p {odir}; {prepare_cmd} && {process_cmd}"
        cmd = f"/usr/bin/env bash -c {shlex.quote(full_command)}"

    annotated_hjson_path = Path(annotated_hjson)
    dry_run = ns["dry_run"]

    def post_finish(status: JobStatus) -> None:
        """Extract the overall vPlan normalised coverage from the annotated HJSON.

        The coverage is stored on the cfg (see SimCfg.vplan_coverage), which
        is where the report generation looks for it.
        """
        if dry_run or status != JobStatus.PASSED:
            return
        if not annotated_hjson_path.exists():
            return
        try:
            with annotated_hjson_path.open() as f:
                data = hjson.load(f)
            # HJSON vPlans are keyed: {dut_name: {fields...}}
            root_node = next(iter(data.values()), {})
            raw = root_node.get("Normalized_Coverage")
            if raw is not None:
                sim_cfg.vplan_coverage = float(str(raw).rstrip(" %"))
        except Exception:  # noqa: BLE001
            log.debug("Could not extract vPlan coverage from '%s'.", annotated_hjson_path)

    return new_job_spec(
        sim_cfg,
        job_type="CovVPlan",
        target=target,
        name=ns["name"],
        qual_name=target,
        cmd=cmd,
        odir=odir,
        gui=sim_cfg.gui,
        dry_run=dry_run,
        exports=resolve_wildcards(ns["exports"], ns),
        weight=10,
        dependencies=[report_job],
        post_finish=post_finish,
    )


class SimCfg:
    """Simulation configuration object.

    A simulation configuration class holds key information required for building
    a DV regression framework.

    Unlike the dict-based flows, the sim flow keeps its config-file state in
    `self.config`, a validated `SimFlowConfig` model. Attribute reads fall
    through to the model (see `__getattr__`), while runtime state lives on the
    instance as usual and shadows config values where the names collide (e.g.
    `testplan` is rebound to a `Testplan` object once parsed).
    """

    flow = "sim"

    # TODO: Find a way to set these in sim cfg instead
    ignored_wildcards: ClassVar = [
        "build_mode",
        "index",
        "test",
        "seed",
        "svseed",
        "uvm_test",
        "uvm_test_seq",
        "cov_db_dirs",
        "sw_images",
        "sw_build_device",
        "sw_build_cmd",
        "sw_build_opts",
    ]

    def __str__(self) -> str:
        """Get string representation of the flow config."""
        return pprint.pformat(self.__dict__)

    def __init__(self, flow_cfg_file, hjson_data, args) -> None:
        # Options set from command line
        self.build_unique = args.build_unique
        self.build_seed = args.build_seed
        self.build_only = args.build_only
        self.run_only = args.run_only
        self.reseed_ovrd = args.reseed
        self.reseed_multiplier = args.reseed_multiplier
        # Waves must be of type string, since it may be used as substitution
        # variable in the HJson cfg files.
        self.waves = args.waves or "none"
        self.max_waves = args.max_waves
        self.cov = args.cov
        self.cov_merge_previous = args.cov_merge_previous
        self.profile = args.profile or "(cfg uses profile without --profile)"
        self.xprop_off = args.xprop_off
        self.verbose = args.verbose
        self.dry_run = args.dry_run
        self.map_full_testplan = args.map_full_testplan

        # Set default sim modes for unpacking
        en_build_modes = args.build_modes.copy()
        if args.gui:
            en_build_modes.append("gui")
        if args.gui_debug:
            en_build_modes.append("gui_debug")
        if args.waves is not None:
            en_build_modes.append("waves")
        else:
            en_build_modes.append("waves_off")
        if self.cov is True:
            en_build_modes.append("cov")
        if args.profile is not None:
            en_build_modes.append("profile")
        if self.xprop_off is not True:
            en_build_modes.append("xprop")
        if self.build_seed:
            en_build_modes.append("build_seed")

        # Command-line options that seed config list values. These are folded
        # into the config model in _merge_hjson: the command-line values come
        # first and the values from the hjson config files are appended.
        self._cli_config_seeds = {
            "build_opts": list(args.build_opts),
            "en_build_modes": en_build_modes,
            "run_opts": list(args.run_opts),
            "en_run_modes": list(args.run_modes),
        }

        # Generated data structures
        self.variant_name = ""
        self.build_list = []
        self.run_list = []
        self.cov_merge_job: JobSpec | None = None
        self.cov_report_job: JobSpec | None = None
        self.cov_vplan_job: JobSpec | None = None

        # Results written back by the job post_finish callbacks (see
        # create_cov_report_job / create_cov_vplan_job).
        self.cov_report_results: dict[str, str] = {}
        self.vplan_coverage: float | None = None

        self.results_summary = OrderedDict()

        init_flow_state(self, flow_cfg_file, args)

        # Merge in the values from the loaded hjson file (into the typed
        # config model - see _merge_hjson).
        self._merge_hjson(hjson_data)

        # A primary cfg simply groups child cfgs (see dvsim.flow.group).
        self.is_primary_cfg = "use_cfgs" in hjson_data

        default_rel_path(self)

        # Process overrides before substituting wildcards.
        self._process_overrides()

        # Expand wildcards.
        self._expand()

        finalize_flow_state(self)

    def __getattr__(self, name: str):
        """Fall back to the config model for attribute reads.

        Only invoked when normal attribute lookup fails, so runtime instance
        attributes (including ones shadowing config keys) take precedence.
        Only config keys are delegated - the model's own API is not exposed.
        """
        config = self.__dict__.get("config")
        if (
            config is not None
            and not (name.startswith("__") and name.endswith("__"))
            and (name in SimFlowConfig.model_fields or name in (config.model_extra or {}))
        ):
            return getattr(config, name)

        msg = f"{type(self).__name__!r} object has no attribute {name!r}"
        raise AttributeError(msg)

    def _is_config_key(self, name: str) -> bool:
        """Whether `name` is a key managed by the config model."""
        return name in SimFlowConfig.model_fields or name in (self.config.model_extra or {})

    def _merge_hjson(self, hjson_data: Mapping) -> None:
        """Load the hjson data into the typed config model.

        Unlike the base class, the sim flow does not merge the hjson data
        into the instance `__dict__`: the validated `SimFlowConfig` model is
        the config state and attribute reads fall through to it (see
        `__getattr__`), with the schema field defaults serving as the config
        defaults.
        """
        try:
            self.config = load_sim_flow_config(self.flow_cfg_file, hjson_data)
        except RuntimeError as err:
            log.error(str(err))
            sys.exit(1)

        # Drop the instance defaults set by init_flow_state for keys the
        # config model manages - they would otherwise shadow the config.
        for key in [k for k in self.__dict__ if self._is_config_key(k)]:
            del self.__dict__[key]

        # Fold the command-line seeded options into the config. CLI values
        # come first, matching the historic merge order where hjson values
        # were appended to the CLI-seeded lists.
        for key, seed in self._cli_config_seeds.items():
            setattr(self.config, key, [*seed, *getattr(self.config, key)])
        del self._cli_config_seeds

    def _process_overrides(self) -> None:
        """Apply the typed overrides from the config model."""
        overrides_seen = {}
        for override in self.config.overrides:
            if override.name in overrides_seen:
                log.error(
                    'Override for key "%s" already exists!\nOld: %s\nNew: %s',
                    override.name,
                    overrides_seen[override.name],
                    override.value,
                )
                sys.exit(1)
            overrides_seen[override.name] = override.value
            self._do_override(override.name, override.value)

    def _do_override(self, ov_name: str, ov_value: object) -> None:
        """Override a single attribute, preferring runtime state over config."""
        in_instance = ov_name in self.__dict__
        if in_instance:
            orig_value = self.__dict__[ov_name]
        elif self._is_config_key(ov_name):
            orig_value = getattr(self.config, ov_name)
        else:
            log.error('Override key "%s" not found in the cfg!', ov_name)
            sys.exit(1)

        if not isinstance(ov_value, type(orig_value)):
            log.error(
                'The type of override value "%s" for "%s" '
                'doesn\'t match the type of original value "%s"',
                ov_value,
                ov_name,
                orig_value,
            )
            sys.exit(1)

        log.debug('Overriding "%s" value "%s" with "%s"', ov_name, orig_value, ov_value)
        if in_instance:
            self.__dict__[ov_name] = ov_value
        else:
            setattr(self.config, ov_name, ov_value)

    def wildcard_namespace(self) -> dict:
        """Merge the config model into the wildcard substitution namespace.

        Runtime instance attributes shadow config values of the same name.
        """
        namespace = self.config.model_dump()
        namespace.update(self.__dict__)
        return namespace

    def apply_expansion(self, expanded: Mapping) -> None:
        """Split the expanded namespace back into config and runtime state.

        Keys that live in the instance `__dict__` (runtime state, including
        shadowed config keys) are updated in place; everything else is config
        data and is re-validated into a fresh config model.
        """
        instance_keys = set(self.__dict__)
        cfg_data = {k: v for k, v in expanded.items() if k not in instance_keys}
        self.__dict__.update((k, v) for k, v in expanded.items() if k in instance_keys)

        try:
            self.config = SimFlowConfig.model_validate(cfg_data)
        except ValidationError as err:
            log.error(
                "%r: config is no longer schema-valid after wildcard expansion:\n%s",
                self.flow_cfg_file,
                err,
            )
            sys.exit(1)

    def _expand(self) -> None:
        # Choose a wave format now. Note that this has to happen after parsing
        # the configuration format because our choice might depend on the
        # chosen tool.
        self.waves = self._resolve_waves()

        # If build_unique is set, then add current timestamp to uniquify it
        if self.build_unique:
            self.build_dir += "_" + self.timestamp

        # If the user specified a verbosity on the command line then
        # self.args.verbosity will be n, l, m, h or d. Set self.verbosity now.
        # We will actually have loaded some other verbosity level from the
        # config file, but that won't have any effect until expansion so we can
        # safely switch it out now.
        if self.args.verbosity is not None:
            self.verbosity = self.args.verbosity

        expand_wildcards(self)

        if self.variant:
            self.variant_name = self.name + "/" + self.variant
        else:
            self.variant_name = self.name

        # Set the title for simulation results.
        self.results_title = self.variant_name.upper() + " Simulation Results"

        # Stuff below only pertains to individual cfg (not primary cfg)
        # or individual selected cfgs (if select_cfgs is configured via command line)
        # TODO: find a better way to support select_cfgs
        if not self.is_primary_cfg and (not self.select_cfgs or self.name in self.select_cfgs):
            # If self.tool is None at this point, there was no --tool argument on
            # the command line, and there is no default tool set in the config
            # file. That's ok if this is a primary config (where the
            # sub-configurations can choose tools themselves), but not otherwise.
            if self.tool is None:
                log.error(
                    "Config file does not specify a default tool, "
                    "and there was no --tool argument on the command line.",
                )
                sys.exit(1)

            # Print scratch_path at the start:
            log.info("[scratch_path]: [%s] [%s]", self.name, self.scratch_path)

            # Use the default build mode for tests that do not specify it
            if not self.build_mode:
                self.build_mode = "default"

            # Set the primary build mode. The coverage associated to this build
            # is the main coverage. Some tools need this information. This is
            # of significance only when there are multiple builds. If there is
            # only one build, and its not the primary_build_mode, then we
            # update the primary_build_mode to match what is built.
            if not self.primary_build_mode:
                self.primary_build_mode = self.build_mode

            # Create objects from raw dicts - build_modes, sim_modes, run_modes,
            # tests and regressions, only if not a primary cfg obj
            self._create_objects()

    def _resolve_waves(self):
        """Choose and return a wave format, if waves are enabled.

        This is called after reading the config file. This method is used to
        update the value of class member 'waves', which must be of type string,
        since it is used as a substitution variable in the parsed HJson dict.
        If waves are not enabled, or if this is a primary cfg, then return
        'none'. 'tool', which must be set at this point, supports a limited
        list of wave formats (supplied with 'supported_wave_formats' key).
        """
        if self.waves == "none" or self.is_primary_cfg:
            return "none"

        assert self.tool is not None

        # If the user has specified their preferred wave format, use it. As
        # a sanity check, error out if the chosen tool doesn't support the
        # format, but only if we know about the tool. If not, we'll just assume
        # they know what they're doing.
        if self.supported_wave_formats and self.waves not in self.supported_wave_formats:
            log.error(
                f"Chosen tool ({self.tool}) does not support wave format {self.waves!r}.",
            )
            sys.exit(1)

        return self.waves

    # Purge the output directories. This operates on self.
    def purge(self) -> None:
        """Purge the scratch area in preparation for the new run."""
        assert self.scratch_path
        log.info("Purging scratch path %s", self.scratch_path)
        rm_path(self.scratch_path)

    def _create_objects(self) -> None:
        # Create build and run modes objects
        self.build_modes = BuildMode.create_modes(self.build_modes)
        self.run_modes = RunMode.create_modes(self.run_modes)

        # Walk through build modes enabled on the CLI and append the opts
        for en_build_mode in self.en_build_modes:
            build_mode_obj = find_mode(en_build_mode, self.build_modes)
            if build_mode_obj is not None:
                self.pre_build_cmds.extend(build_mode_obj.pre_build_cmds)
                self.post_build_cmds.extend(build_mode_obj.post_build_cmds)
                self.build_opts.extend(build_mode_obj.build_opts)
                self.post_build_opts.extend(build_mode_obj.post_build_opts)
                self.pre_run_cmds.extend(build_mode_obj.pre_run_cmds)
                self.post_run_cmds.extend(build_mode_obj.post_run_cmds)
                self.run_opts.extend(build_mode_obj.run_opts)
                self.sw_images.extend(build_mode_obj.sw_images)
                self.sw_build_opts.extend(build_mode_obj.sw_build_opts)
            else:
                log.error(
                    'Mode "%s" enabled on the command line is not defined',
                    en_build_mode,
                )
                sys.exit(1)

        # Walk through run modes enabled on the CLI and append the opts
        for en_run_mode in self.en_run_modes:
            run_mode_obj = find_mode(en_run_mode, self.run_modes)
            if run_mode_obj is not None:
                self.pre_run_cmds.extend(run_mode_obj.pre_run_cmds)
                self.post_run_cmds.extend(run_mode_obj.post_run_cmds)
                self.run_opts.extend(run_mode_obj.run_opts)
                self.sw_images.extend(run_mode_obj.sw_images)
                self.sw_build_opts.extend(run_mode_obj.sw_build_opts)
            else:
                log.error('Mode "%s" enabled on the command line is not defined', en_run_mode)
                sys.exit(1)

        # Create tests from given list of items
        self.tests = Test.create_tests(self.tests, self)

        # Regressions
        # Parse testplan if provided.
        if self.testplan != "":
            self.testplan = Testplan(
                self.testplan,
                repo_top=Path(self.proj_root),
                name=self.variant_name,
            )
            # Extract tests in each stage and add them as regression target.
            self.regressions.extend(self.testplan.get_stage_regressions())
        else:
            # Create a dummy testplan with no entries.
            self.testplan = Testplan(
                "<dummy testplan>", repo_top=Path(self.proj_root), name=self.name
            )

        # Create regressions
        self.regressions = Regression.create_regressions(self.regressions, self, self.tests)

    def print_list(self) -> None:
        """Print the list of available items that can be kicked off."""
        for list_item in self.list_items:
            log.info("---- List of %s in %s ----", list_item, self.variant_name)
            items = getattr(self, list_item, None)
            if items is None:
                log.error("No %s defined for %s.", list_item, self.variant_name)

            for item in items:
                # Convert the item into something that can be printed in the
                # list. Some modes are specified as strings themselves (so
                # there's no conversion needed). Others should be subclasses of
                # Mode, which has a name field that we can use.
                if isinstance(item, str):
                    mode_name = item
                else:
                    assert isinstance(item, Mode)
                    mode_name = item.name

                log.info(mode_name)

    def _create_build_and_run_list(self) -> None:
        """Generate a list of deployable objects from the provided items.

        Tests to be run are provided with --items switch. These can be glob-
        style patterns. This method finds regressions and tests that match
        these patterns.
        """

        def _match_items(items: list, patterns: list):
            hits = []
            matched = set()
            for pattern in patterns:
                item_hits = fnmatch.filter(items, pattern)
                if item_hits:
                    hits += item_hits
                    matched.add(pattern)
            return hits, matched

        # Process regressions first.
        regr_map = {regr.name: regr for regr in self.regressions}
        regr_hits, items_matched = _match_items(regr_map.keys(), self.items)
        regrs = [regr_map[regr] for regr in regr_hits]
        for regr in regrs:
            overlap = bool([t for t in regr.tests if t in self.run_list])
            if overlap:
                log.warning(
                    f"Regression {regr.name} added to be run has tests that "
                    "overlap with other regressions also being run. This can "
                    "result in conflicting build / run time opts to be set, "
                    "resulting in unexpected results. Skipping.",
                )
                continue

            self.run_list += regr.tests
            # Merge regression's build and run opts with its tests and their
            # build_modes.
            regr.merge_regression_opts()

        # Process individual tests, skipping the ones already added from
        # regressions.
        test_map = {test.name: test for test in self.tests if test not in self.run_list}
        test_hits, items_matched_ = _match_items(test_map.keys(), self.items)
        self.run_list += [test_map[test] for test in test_hits]
        items_matched |= items_matched_

        # Check if all items have been processed.
        for item in set(self.items) - items_matched:
            log.warning(
                f"Item {item} did not match any regressions or tests in {self.flow_cfg_file}.",
            )

        # Merge the global build and run opts
        Test.merge_global_opts(
            self.run_list,
            self.pre_build_cmds,
            self.post_build_cmds,
            self.build_opts,
            self.post_build_opts,
            self.pre_run_cmds,
            self.post_run_cmds,
            self.run_opts,
            self.sw_images,
            self.sw_build_opts,
        )

        # Process reseed override and create the build_list
        build_list_names = []
        for test in self.run_list:
            # Override reseed if available.
            if self.reseed_ovrd is not None:
                test.reseed = self.reseed_ovrd

            # Apply reseed multiplier if set on the command line. This is
            # always positive but might not be an integer. Round to nearest,
            # but make sure there's always at least one iteration.
            scaled = round(test.reseed * self.reseed_multiplier)
            test.reseed = max(1, scaled)

            # Create the unique set of builds needed.
            if test.build_mode.name not in build_list_names:
                self.build_list.append(test.build_mode)
                build_list_names.append(test.build_mode.name)

    def _expand_run_list(self, build_map):
        """Generate a list of tests to be run.

        For each test in tests, we add it test.reseed times. The ordering is
        interleaved so that we run through all of the tests as soon as
        possible. If there are multiple tests and they have different reseed
        values, they are "fully interleaved" at the start (so if there are
        tests A, B with reseed values of 5 and 2, respectively, then the list
        will be ABABAAA).

        build_map is a dictionary mapping a build mode to its build job spec.
        """
        tagged = []

        for test in self.run_list:
            build_job = build_map[test.build_mode]
            tagged.extend(
                (idx, create_run_test_job(idx, test, build_job, self))
                for idx in range(test.reseed)
            )

        # Stably sort the tagged list by the 1st coordinate.
        tagged.sort(key=lambda x: x[0])

        # Return the sorted list of RunTest objects, discarding the indices by
        # which we sorted it.
        return [run for _, run in tagged]

    def create_deploy_objects(self) -> None:
        """Create deploy objects from the build and run lists."""
        # Create the build and run list first
        self._create_build_and_run_list()

        self.builds = []
        build_map = {}
        for i, build_mode_obj in enumerate(self.build_list, start=1):
            log.debug(
                "Creating build mode obj %d/%d: %s",
                i,
                len(self.build_list),
                build_mode_obj.name,
            )
            new_build = create_compile_sim_job(build_mode_obj, self)

            # It is possible for tests to supply different build modes, but
            # those builds may differ only under specific circumstances,
            # such as coverage being enabled. If coverage is not enabled,
            # then they may be completely identical. In that case, we can
            # save compute resources by removing the extra duplicated
            # builds. We discard the new_build if it is equivalent to an
            # existing one.
            is_unique = True
            for build in self.builds:
                if is_equivalent_job_spec(build, new_build):
                    # Discard `new_build` since build implements the same
                    # thing. If `new_build` is the same as
                    # `primary_build_mode`, update `primary_build_mode` to
                    # match `build`.
                    if new_build.name == self.primary_build_mode:
                        self.primary_build_mode = build.name
                    new_build = build
                    is_unique = False
                    break

            if is_unique:
                self.builds.append(new_build)
            build_map[build_mode_obj] = new_build

        # If there is only one build, set primary_build_mode to it.
        if len(self.builds) == 1:
            self.primary_build_mode = self.builds[0].name

        # Check self.primary_build_mode is set correctly.
        build_mode_names = {b.name for b in self.builds}
        if not self.build_list and not self.run_list:
            log.error("Nothing to do as no matching jobs could be found.")
            sys.exit(1)
        if self.primary_build_mode not in build_mode_names:
            log.error(
                f'"primary_build_mode: {self.primary_build_mode}" '
                f"in {self.name} cfg is invalid. Please pick from "
                f"{build_mode_names}.",
            )
            sys.exit(1)

        # Update all tests to use the updated (uniquified) build modes.
        for test in self.run_list:
            if test.build_mode.name != build_map[test.build_mode].name:
                test.build_mode = find_mode(build_map[test.build_mode].name, self.build_modes)

        self.runs = [] if self.build_only else self._expand_run_list(build_map)

        # In GUI mode or GUI with debug mode, only allow one test to run.
        if self.gui and len(self.runs) > 1:
            self.runs = self.runs[:1]
            log.warning(
                f"In GUI mode, only one test is allowed to run. Picking {self.runs[0].full_name}",
            )

        # GUI mode is only available for Xcelium for the moment.
        if (self.gui_debug) and (self.tool != "xcelium"):
            log.error(
                "GUI debug mode is only available for Xcelium, please remove "
                "--gui_debug / -gd option or switch to Xcelium tool.",
            )
            sys.exit(1)

        # Add builds to the list of things to run, only if --run-only switch
        # is not passed.
        self.deploy = []
        if not self.run_only:
            self.deploy += self.builds

        if not self.build_only:
            self.deploy += self.runs

            # Create cov_merge and cov_report jobs, so long as we've got at
            # least one run to do.
            if self.cov and self.runs:
                # The build mode name of each run, used to derive the set of
                # coverage databases to merge.
                mode_by_test = {test.name: test.build_mode.name for test in self.run_list}
                run_build_modes = [mode_by_test[run.name] for run in self.runs]

                self.cov_merge_job = create_cov_merge_job(self.runs, run_build_modes, self)
                self.cov_report_job = create_cov_report_job(self.cov_merge_job, self)
                self.deploy += [self.cov_merge_job, self.cov_report_job]

                if getattr(self, "vplan", False):
                    self.cov_vplan_job = create_cov_vplan_job(self.cov_report_job, self)
                    self.deploy.append(self.cov_vplan_job)

    def cov_analyze(self) -> None:
        """Open GUI tool for coverage analysis.

        Use the last regression coverage data to open up the GUI tool to analyze
        the coverage.
        """
        self.deploy = [create_cov_analyze_job(self)]

    def cov_unr(self) -> None:
        """Generate unreachable coverage exclusions.

        Use the last regression coverage data to generate unreachable coverage
        exclusions.
        """
        # TODO, Only support VCS
        if self.tool not in ["vcs", "xcelium"]:
            log.error("Only VCS and Xcelium are supported for the UNR flow.")
            sys.exit(1)

        self.deploy = [create_cov_unr_job(self)]

    def has_errors(self) -> bool:
        """Return error state."""
        return self.errors_seen

    def gen_results(
        self,
        results: Sequence[CompletedJobStatus],
        cfgs: Sequence["SimCfg"],
    ) -> None:
        """Generate flow results.

        Args:
            results: completed job status objects.
            cfgs: the flow configs that were run (see dvsim.flow.group).

        """
        repo_root = Path(self.proj_root)
        reports_dir = Path(self.scratch_base_path) / "reports"
        url = git_https_url_with_commit(path=repo_root)
        build_seed = self.build_seed if not self.run_only else None

        try:
            dvsim_version = version("dvsim").strip()

        except PackageNotFoundError as e:
            log.debug("DVSim package not found: %s", str(e))
            dvsim_version = None

        all_flow_results: Mapping[str, SimFlowResults] = {}
        flow_summaries: Mapping[str, SimFlowSummary] = {}

        for item in cfgs:
            item_results = [
                res
                for res in results
                if res.block.name == item.name and res.block.variant == item.variant
            ]

            flow_results: SimFlowResults = item._gen_json_results(
                run_results=item_results,
                url=url,
            )

            # Convert to lowercase to match filename
            block_result_index = item.variant_name.lower().replace("/", "_")

            all_flow_results[block_result_index] = flow_results
            flow_summaries[block_result_index] = flow_results.summary()

            self.errors_seen |= item.errors_seen

        # The timestamp for this run has been taken with `utcnow()` and is
        # stored in a custom format.  Store it in standard ISO format with
        # explicit timezone annotation.
        timestamp = (
            datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S")
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )

        # If this is a primary config, attach the "top" information. Even if it isn't,
        # we still want to potentially generate a summary to attach other metadata.
        top = None
        if self.is_primary_cfg:
            top = IPMeta(
                name=self.name,
                variant=self.variant,
                commit=self.commit,
                commit_short=self.commit_short,
                branch=self.branch,
                url=url,
                revision_info=self.revision,
            )

        results_summary = SimResultsSummary(
            top=top,
            version=dvsim_version,
            timestamp=timestamp,
            build_seed=build_seed,
            flow_results=flow_summaries,
            report_path=reports_dir,
        )

        # Generate all the JSON/HTML reports to the report area.
        gen_reports(
            summary=results_summary,
            flow_results=all_flow_results,
            path=reports_dir,
        )

    def _gen_json_results(
        self,
        run_results: Sequence[CompletedJobStatus],
        url: str,
    ) -> SimFlowResults:
        """Generate structured SimFlowResults from simulation run data.

        Args:
            run_results: completed job status.
            url: for the IP source

        Returns:
            Flow results object.

        """
        sim_results = SimResults(results=run_results)
        if not self.testplan.test_results_mapped:
            self.testplan.map_test_results(sim_results.table)

        # --- Metadata ---
        timestamp = datetime.strptime(self.timestamp, TS_FORMAT).replace(tzinfo=timezone.utc)

        block = IPMeta(
            name=self.name.lower(),
            variant=(self.variant or "").lower() or None,
            commit=self.commit,
            commit_short=self.commit_short,
            branch=self.branch or "",
            url=url,
            revision_info=self.revision,
        )
        tool = ToolMeta(name=self.tool.lower(), version="unknown")

        build_seed = self.build_seed if not self.run_only else None

        # Build up a reference to the testplan, which might be overridden.
        if self.testplan_doc_path:
            rel_path = relative_to(
                Path(self.testplan_doc_path),
                Path(self.proj_root),
            )

        else:
            # TODO: testplan variants frequently override `rel_path` for reporting
            # and build reasons, but do not update the `testplan_doc_path`, meaning
            # that they point to a variant testplan path that is not available
            # in the book, unlike the original (non-variant).
            rel_path = Path(self.rel_path).parent / "data" / f"{self.name}_testplan.hjson"

        if self.book:
            testplan_ref = "https://{}/{}".format(self.book, str(rel_path.with_suffix(".html")))
        else:
            testplan_ref = str(rel_path)

        # --- Build stages only from testpoints that have at least one executed test ---
        stage_to_tps: defaultdict[str, dict[str, Testpoint]] = defaultdict(dict)
        stage_to_trs: defaultdict[str, dict[str, TestResult]] = defaultdict(dict)
        all_trs: defaultdict[str, TestResult] = {}

        def make_test_result(tr) -> TestResult | None:
            if tr.total == 0 and not self.map_full_testplan:
                return None

            return TestResult(
                max_time=tr.job_runtime,
                sim_time=tr.simulated_time,
                passed=tr.passing,
                total=tr.total,
                percent=100.0 * tr.passing / (tr.total or 1),
            )

        # 1. Mapped testpoints — only include if at least one test ran
        for tp in self.testplan.testpoints:
            if tp.name in {"Unmapped tests", "N.A."}:
                continue

            test_results: dict[str, TestResult] = {}
            for tr in tp.test_results:
                if test := make_test_result(tr):
                    test_results[tr.name] = test

            # Critical: skip entire testpoint if no tests actually ran
            if not test_results and not self.map_full_testplan:
                continue

            # Aggregate testpoint stats
            tp_passed = sum(t.passed for t in test_results.values())
            tp_total = sum(t.total for t in test_results.values())

            stage_to_tps[tp.stage][tp.name] = Testpoint(
                tests=test_results,
                passed=tp_passed,
                total=tp_total,
                percent=100.0 * tp_passed / tp_total if tp_total else 0.0,
            )

            for name, tr in test_results.items():
                stage_to_trs[tp.stage][name] = tr
                all_trs[name] = tr

        # 2. Unmapped tests — only if they actually ran
        unmapped_tests: dict[str, TestResult] = {}
        for tr in sim_results.table:
            if not tr.mapped and (test := make_test_result(tr)):
                unmapped_tests[tr.name] = test

        if unmapped_tests:
            tp_passed = sum(t.passed for t in unmapped_tests.values())
            tp_total = sum(t.total for t in unmapped_tests.values())
            stage_to_tps["unmapped"]["Unmapped"] = Testpoint(
                tests=unmapped_tests,
                passed=tp_passed,
                total=tp_total,
                percent=100.0 * tp_passed / tp_total if tp_total else 0.0,
            )

            for name, tr in unmapped_tests.items():
                stage_to_trs["unmapped"][name] = tr
                all_trs[name] = tr

        # --- Final stage aggregation ---
        stages: dict[str, TestStage] = {}

        for stage_name, testpoints in stage_to_tps.items():
            stage_passed = sum(tr.passed for tr in stage_to_trs[stage_name].values())
            stage_total = sum(tr.total for tr in stage_to_trs[stage_name].values())

            stages[stage_name] = TestStage(
                testpoints=testpoints,
                passed=stage_passed,
                total=stage_total,
                percent=100.0 * stage_passed / stage_total if stage_total else 0.0,
            )

        total_passed = sum(tr.passed for tr in all_trs.values())
        total_runs = sum(tr.total for tr in all_trs.values())

        # --- Coverage ---
        coverage: dict[str, float | None] = {}
        coverage_model = None
        for k, v in self.cov_report_results.items():
            try:
                coverage[k.lower()] = float(v.rstrip("% "))
            except (ValueError, TypeError, AttributeError):
                coverage[k.lower()] = None

        coverage_model = get_sim_tool_plugin(self.tool).get_coverage_metrics(
            raw_metrics=coverage,
        )

        # Link to the coverage report page, if one exists
        cov_report_page = None
        if self.cov_report_page:
            cov_report_dir = self.cov_report_dir or "cov_report"
            cov_report_page = Path(cov_report_dir, self.cov_report_page)

        vplan_report_page = None
        vplan_coverage = None
        if self.cov_vplan_job:
            vplan_report_page = Path(self.scratch_path) / "cov_vplan" / "vplan_annotated.html"
            vplan_coverage = self.vplan_coverage

        failures = BucketedFailures.from_job_status(results=run_results)
        if failures.buckets:
            self.errors_seen = True

        # --- Final result ---
        return SimFlowResults(
            block=block,
            tool=tool,
            timestamp=timestamp,
            build_seed=build_seed,
            testplan_ref=testplan_ref,
            stages=stages,
            coverage=coverage_model,
            cov_report_page=cov_report_page,
            vplan_report_page=vplan_report_page,
            vplan_coverage=vplan_coverage,
            failed_jobs=failures,
            passed=total_passed,
            total=total_runs,
            percent=100.0 * total_passed / total_runs if total_runs else 0.0,
        )

    def fake_policy(self, job: JobSpec) -> JobStatus | None:
        """Tell the fake backend how to fake jobs for this flow.

        Currently randomly returns 50% pass / 50% fail for RunTest jobs, and fakes injecting
        randomized 0-100% coverage results into the cfg's coverage report results. Returns
        None for jobs this cfg has no opinion on (see dvsim.flow.group).
        """
        if job.job_type == "RunTest":
            return random.choice((JobStatus.PASSED, JobStatus.FAILED))

        # TODO: hack, try to remove. Annotate the cfg with some faked
        # coverage results. Just allows us to fake coverage results for now
        # without needing to create a fake result file or significantly refactor.
        if (
            job.job_type == "CovReport"
            and self.cov_report_job
            and self.cov_report_job.full_name == job.full_name
        ):
            fake_keys = [
                "score",
                "assert",
                "group",
                "block",
                "line",
                "branch",
                "cond",
                "toggle",
                "fsm",
            ]
            # Approximate the correct keys; in reality some jobs might not
            # have certain types of coverage (e.g. FSM) depending on the RTL
            if job.tool == "vcs":
                fake_keys.remove("block")
            elif job.tool == "xcelium":
                fake_keys.remove("cond")
            self.cov_report_results = {k: f"{random.random() * 100:.2f} %" for k in fake_keys}
            return JobStatus.PASSED

        return None
