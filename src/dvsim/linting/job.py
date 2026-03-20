# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lint job builder for creating executable lint jobs."""

from __future__ import annotations

import re
import shlex
import sys
from typing import TYPE_CHECKING

from dvsim.job.data import JobSpec, WorkspaceConfig
from dvsim.linting.config import LintBlockConfig  # noqa: TC001
from dvsim.logging import log
from dvsim.report.data import IPMeta, ToolMeta
from dvsim.utils import subst_wildcards

if TYPE_CHECKING:
    from pathlib import Path


def create_lint_job(
    config: LintBlockConfig,
    args,
    proj_root: Path,
    scratch_root: Path,
    timestamp: str,
    branch: str,
) -> JobSpec:
    """Create a JobSpec for a lint block config.

    Args:
        config: Lint block configuration
        args: Command line arguments
        proj_root: Project root directory
        scratch_root: Scratch root directory
        timestamp: Timestamp for this run
        branch: Git branch name

    Returns:
        JobSpec ready for execution

    """
    tool = config.tool or args.tool

    # Standardized scratch directory structure: scratch/<branch>/<block>/lint/<tool>/
    lint_base = scratch_root / branch / config.name / "lint"
    scratch_path = lint_base / tool
    build_dir = scratch_path
    build_log = scratch_path / "lint.log"  # Generic log name for makefile output

    # Wildcard values for substituting into build command and options
    wildcard_values = {
        "scratch_root": str(scratch_root),
        "proj_root": str(proj_root),
        "branch": branch,
        "tool": tool,
        "build_mode": "default",  # Not used for directory structure, but may be in build_opts
        "build_dir": str(build_dir),
        "job_prefix": args.job_prefix if hasattr(args, "job_prefix") else "",
    }

    # Build makefile-based command like the old flow
    flow_makefile = subst_wildcards(
        getattr(config, "flow_makefile", "").strip(),
        wildcard_values,
    )

    if not flow_makefile:
        msg = f"Missing flow_makefile in config for {config.name}"
        raise ValueError(msg)

    # Build the command using make with all required parameters
    build_cmd_value = subst_wildcards(config.build_cmd, wildcard_values).strip()
    build_opts_value = " ".join(subst_wildcards(opt, wildcard_values) for opt in config.build_opts)

    # Fix the fusesoc work-root to use our build_dir instead of pre-resolved path
    # Replace any --work-root=<path> with our standardized location
    build_opts_value = re.sub(
        r"--work-root=[^\s]+", f"--work-root={build_dir}/fusesoc-work", build_opts_value
    )

    # Use DVSim's built-in tool parser instead of external scripts
    # Use the same Python interpreter that's running DVSim
    report_cmd = f"{sys.executable} -m dvsim.linting.tool.{tool}"

    # Construct report_opts based on our build_dir
    # The parser expects --repfile and --outfile
    report_opts_value = f"--repfile={build_log} --outfile={build_dir}/results.hjson"

    # Get pre/post build commands
    pre_build_cmds = getattr(config, "pre_build_cmds", [])
    post_build_cmds = getattr(config, "post_build_cmds", [])
    pre_build_cmds_value = (
        " && ".join(subst_wildcards(cmd, wildcard_values) for cmd in pre_build_cmds)
        if pre_build_cmds
        else ""
    )
    post_build_cmds_value = (
        " && ".join(subst_wildcards(cmd, wildcard_values) for cmd in post_build_cmds)
        if post_build_cmds
        else ""
    )

    # Get flist generation fields - handle both string and list
    sv_flist_gen_cmd_raw = getattr(config, "sv_flist_gen_cmd", "")
    sv_flist_gen_cmd = subst_wildcards(
        sv_flist_gen_cmd_raw if isinstance(sv_flist_gen_cmd_raw, str) else "",
        wildcard_values,
    )

    sv_flist_gen_dir_raw = getattr(config, "sv_flist_gen_dir", "")
    sv_flist_gen_dir = subst_wildcards(
        sv_flist_gen_dir_raw if isinstance(sv_flist_gen_dir_raw, str) else "",
        wildcard_values,
    )

    sv_flist_gen_opts_raw = getattr(config, "sv_flist_gen_opts", "")
    if isinstance(sv_flist_gen_opts_raw, list):
        sv_flist_gen_opts = " ".join(
            subst_wildcards(opt, wildcard_values) for opt in sv_flist_gen_opts_raw
        )
    else:
        sv_flist_gen_opts = subst_wildcards(
            sv_flist_gen_opts_raw if isinstance(sv_flist_gen_opts_raw, str) else "",
            wildcard_values,
        )

    # Build timeout
    build_timeout_mins = getattr(config, "build_timeout_mins", None)

    # Construct make command with all parameters
    cmd_parts = [
        "make",
        "-f",
        shlex.quote(flow_makefile),
        "build",
        f"build_cmd={shlex.quote(build_cmd_value)}",
        f"build_dir={shlex.quote(str(build_dir))}",
        f"build_log={shlex.quote(str(build_log))}",
        f"build_opts={shlex.quote(build_opts_value)}",
        f"build_timeout_mins={build_timeout_mins}",
        f"post_build_cmds={shlex.quote(post_build_cmds_value)}",
        f"pre_build_cmds={shlex.quote(pre_build_cmds_value)}",
        f"proj_root={shlex.quote(str(proj_root))}",
        f"report_cmd={shlex.quote(report_cmd)}",
        f"report_opts={shlex.quote(report_opts_value)}",
        f"sv_flist_gen_cmd={shlex.quote(sv_flist_gen_cmd)}",
        f"sv_flist_gen_dir={shlex.quote(sv_flist_gen_dir)}",
        f"sv_flist_gen_opts={shlex.quote(sv_flist_gen_opts)}",
    ]
    cmd = " ".join(cmd_parts)

    # Set up exports
    exports = {
        "SCRATCH_PATH": str(scratch_path.parent),  # Parent of tool directory
        "proj_root": str(proj_root),
    }

    log.debug(f"Lint job command for {config.name}: {cmd}")
    log.debug(f"Build dir: {build_dir}")
    log.debug(f"Build log: {build_log}")

    return JobSpec(
        name=config.name,
        job_type="LintJob",
        target="build",
        backend=None,
        seed=None,
        full_name=f"{config.name}:default",
        qual_name=config.name,
        block=IPMeta(
            name=config.name,
            variant="",
            commit="",
            commit_short="",
            branch=branch,
            url="",
            revision_info="",
        ),
        tool=ToolMeta(
            name=tool,
            version="",
        ),
        workspace_cfg=WorkspaceConfig(
            timestamp=timestamp,
            project_root=proj_root,
            scratch_root=scratch_root,
            scratch_path=scratch_path,
        ),
        dependencies=[],
        needs_all_dependencies_passing=True,
        weight=1,
        timeout_mins=60,  # Default 60 minute timeout for lint jobs
        cmd=cmd,
        exports=exports,
        dry_run=args.dry_run if hasattr(args, "dry_run") else False,
        interactive=args.interactive if hasattr(args, "interactive") else False,
        odir=build_dir,
        renew_odir=False,
        log_path=build_log,
        pre_launch=lambda: None,
        post_finish=lambda _: None,
        pass_patterns=config.pass_patterns if hasattr(config, "pass_patterns") else [],
        fail_patterns=config.fail_patterns if hasattr(config, "fail_patterns") else [],
    )
