# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test Job deployment models."""

from hamcrest import assert_that, equal_to

from dvsim.job.deploy import CompileSim


class FakeCliArgs:
    """Fake CLI args."""

    def __init__(self) -> None:
        """Initialise fake command line arguments."""
        self.build_timeout_mins = 500
        self.timestamp = "timestamp"


class FakeSimCfg:
    """Fake sim configuration."""

    def __init__(self) -> None:
        """Initialise fake sim configuration."""
        self.name = "flow_name"

        self.args = FakeCliArgs()
        self.dry_run = True
        self.gui = False

        self.scratch_path = "/scratch_path"
        self.scratch_root = "/scratch_root"
        self.proj_root = "/project"

        self.exports = []

        self.flow_makefile = "path/to/makefile"
        self.build_cmd = "build_cmd"
        self.pre_build_cmds = ["A", "B"]
        self.post_build_cmds = ["C", "D"]
        self.build_dir = "build/dir"
        self.build_pass_patterns = None
        self.build_fail_patterns = None
        self.build_seed = None

        self.sv_flist_gen_cmd = "gen_cmd"
        self.sv_flist_gen_opts = []
        self.sv_flist_gen_dir = "path/to/gen"

        self.cov = True
        self.cov_db_dir = "path"


class FakeBuildMode:
    """Fake BuildMode."""

    def __init__(self) -> None:
        """Initialise fake BuildMode."""
        self.name = "build_name"
        self.build_timeout_mins = 500
        self.build_mode = "build_mode"
        self.build_opts = []


class TestCompileSim:
    """Test CompileSim."""

    @staticmethod
    def test_new() -> None:
        """Test that a CompileSim can be constructed with new()."""
        job = CompileSim.new(
            build_mode_obj=FakeBuildMode(),
            sim_cfg=FakeSimCfg(),
        )

        assert_that(job.name, equal_to("build_name"))
        assert_that(
            job.cmd,
            equal_to(
                "make -f path/to/makefile build "
                "-n "
                "build_cmd=build_cmd "
                "build_dir=build/dir "
                "build_opts='' "
                "post_build_cmds='C && D' "
                "pre_build_cmds='A && B' "
                "proj_root=/project "
                "sv_flist_gen_cmd=gen_cmd "
                "sv_flist_gen_dir=path/to/gen "
                "sv_flist_gen_opts=''"
            ),
        )
