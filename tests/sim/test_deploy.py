# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test job spec factories."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from hamcrest import assert_that, equal_to

from dvsim.job.data import JobSpec, WorkspaceConfig
from dvsim.sim.flow import create_compile_sim_job

__all__ = ()


class FakeCliArgs:
    """Fake CLI args."""

    def __init__(self) -> None:
        """Initialise fake command line arguments."""
        self.build_timeout_mins = None
        self.timestamp = "timestamp"


class FakeSimCfg:
    """Fake sim configuration."""

    flow = "fake"

    def __init__(self) -> None:
        """Initialise fake sim configuration."""
        self.name = "flow_name"
        self.variant = "variant"

        self.args = FakeCliArgs()
        self.dry_run = True
        self.gui = False
        self.interactive = False

        self.scratch_path = "/scratch_path"
        self.scratch_root = "/scratch_root"
        self.proj_root = "/project"

        self.tool = "faketool"
        self.commit = "0123456789abcdef"
        self.commit_short = "0123456"
        self.branch = "main"
        self.revision = "revision"
        self.workspace_cfg = WorkspaceConfig(
            timestamp="timestamp",
            project_root=Path("/project"),
            scratch_root=Path("/scratch_root"),
            scratch_path=Path("/scratch_path"),
        )

        self.exports = []

        self.flow_makefile = "path/to/makefile"
        self.build_cmd = "path/to/{build_mode}/build_cmd"
        self.pre_build_cmds = ["A", "B"]
        self.post_build_cmds = ["C", "D"]
        self.build_dir = "build/dir"
        self.build_pass_patterns = []
        self.build_fail_patterns = []
        self.build_seed = 123

        self.sv_flist_gen_cmd = "gen_cmd"
        self.sv_flist_gen_opts = []
        self.sv_flist_gen_dir = "path/to/gen"

        self.cov_db_dir = "path"

    def wildcard_namespace(self) -> dict:
        """Return the wildcard substitution namespace (see dvsim.flow.bootstrap)."""
        return self.__dict__


class FakeBuildMode:
    """Fake BuildMode."""

    def __init__(self) -> None:
        """Initialise fake BuildMode."""
        self.name = "build_name"
        self.build_timeout_mins = 500
        self.build_mode = "build_mode"
        self.build_opts = ["-b path/here", '-a "Quoted"']
        self.post_build_opts = ["E"]


def _create_compile_sim_job(
    *,
    build_overrides: Mapping | None = None,
    sim_overrides: Mapping | None = None,
    cli_args_overrides: Mapping | None = None,
) -> JobSpec:
    """Create a build job spec.

    Test helper that takes overrides to apply on top of the default values for
    the BuildMode and SimCfg fake objects.
    """
    cli_args = FakeCliArgs()
    if cli_args_overrides:
        for arg, value in cli_args_overrides.items():
            setattr(cli_args, arg, value)

    build_mode_obj = FakeBuildMode()
    if build_overrides:
        for arg, value in build_overrides.items():
            setattr(build_mode_obj, arg, value)

    sim_cfg = FakeSimCfg()
    if sim_overrides:
        for arg, value in sim_overrides.items():
            setattr(sim_cfg, arg, value)

    # Override the cli args in the sim configuration
    sim_cfg.args = cli_args

    return create_compile_sim_job(
        build_mode=build_mode_obj,
        sim_cfg=sim_cfg,
    )


class TestCreateCompileSimJob:
    """Test the create_compile_sim_job factory."""

    @staticmethod
    @pytest.mark.parametrize(
        ("build_overrides", "sim_overrides", "exp_cmd"),
        [
            (
                {"dry_run": True},
                {},
                "make -f path/to/makefile build "
                "-n "
                "build_cmd=path/to/build_name/build_cmd "
                "build_dir=build/dir "
                "build_opts='-b path/here -a \"Quoted\"' "
                "post_build_cmds='C && D' "
                "post_build_opts=E "
                "pre_build_cmds='A && B' "
                "proj_root=/project "
                "sv_flist_gen_cmd=gen_cmd "
                "sv_flist_gen_dir=path/to/gen "
                "sv_flist_gen_opts=''",
            ),
            (
                {"dry_run": False},
                {},
                "make -f path/to/makefile build "
                "build_cmd=path/to/build_name/build_cmd "
                "build_dir=build/dir "
                "build_opts='-b path/here -a \"Quoted\"' "
                "post_build_cmds='C && D' "
                "post_build_opts=E "
                "pre_build_cmds='A && B' "
                "proj_root=/project "
                "sv_flist_gen_cmd=gen_cmd "
                "sv_flist_gen_dir=path/to/gen "
                "sv_flist_gen_opts=''",
            ),
        ],
    )
    def test_cmd(build_overrides: Mapping, sim_overrides: Mapping, exp_cmd: str) -> None:
        """Test that a build job spec has the expected cmd."""
        job = _create_compile_sim_job(
            build_overrides=build_overrides,
            sim_overrides=sim_overrides,
        )

        assert_that(job.cmd, equal_to(exp_cmd))

    @staticmethod
    @pytest.mark.parametrize(
        ("build_overrides", "sim_overrides", "name", "full_name"),
        [
            ({"name": "fred"}, {"variant": None}, "fred", "flow_name:fred"),
            ({"name": "fred"}, {"variant": "v1"}, "fred", "flow_name_v1:fred"),
            ({"name": "fred"}, {"name": "flow", "variant": None}, "fred", "flow:fred"),
            ({"name": "george"}, {"variant": None}, "george", "flow_name:george"),
            ({"name": "george"}, {"variant": "v2"}, "george", "flow_name_v2:george"),
        ],
    )
    def test_names(
        build_overrides: Mapping,
        sim_overrides: Mapping,
        name: str,
        full_name: str,
    ) -> None:
        """Test that a build job spec ends up with the expected names."""
        job = _create_compile_sim_job(
            build_overrides=build_overrides,
            sim_overrides=sim_overrides,
        )

        assert_that(job.name, equal_to(name))
        assert_that(job.qual_name, equal_to(name))
        assert_that(job.full_name, equal_to(full_name))

    @staticmethod
    @pytest.mark.parametrize(
        ("sim_overrides", "seed"),
        [
            ({"build_seed": 123}, 123),
            ({"build_seed": 631}, 631),
        ],
    )
    def test_seed(
        sim_overrides: Mapping,
        seed: int,
    ) -> None:
        """Test that a build job spec ends up with the expected seed."""
        job = _create_compile_sim_job(
            sim_overrides=sim_overrides,
        )

        assert_that(job.seed, equal_to(seed))

    @staticmethod
    @pytest.mark.parametrize(
        ("cli_args_overrides", "build_overrides", "timeout"),
        [
            ({"build_timeout_mins": 111}, {}, 111),
            ({}, {"build_timeout_mins": 112}, 112),
        ],
    )
    def test_timeout(
        cli_args_overrides: Mapping,
        build_overrides: Mapping,
        timeout: int,
    ) -> None:
        """Test that a build job spec ends up with the expected timeout."""
        job = _create_compile_sim_job(
            build_overrides=build_overrides,
            cli_args_overrides=cli_args_overrides,
        )

        assert_that(job.timeout_mins, equal_to(timeout))
