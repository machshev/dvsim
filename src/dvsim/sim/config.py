# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models describing the sim flow configuration schema.

`SimFlowConfig` validates the merged hjson data for a sim flow config (the
output of `dvsim.flow.hjson.load_hjson`) before it is merged into a `SimCfg`.
The mode / test / regression sub-schemas mirror the attributes of the
corresponding classes in `dvsim.modes`, `dvsim.test` and `dvsim.regression`
and reject unknown keys, catching typos at load time instead of deep inside
mode merging.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from dvsim.flow.config import FlowConfig

__all__ = (
    "BuildModeConfig",
    "RegressionConfig",
    "RunModeConfig",
    "SimFlowConfig",
    "TestConfig",
    "validate_sim_cfg_data",
)


class BuildModeConfig(BaseModel):
    """Schema for an entry of `build_modes` (see `dvsim.modes.BuildMode`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    is_sim_mode: int = 0
    pre_build_cmds: Sequence[str] = ()
    post_build_cmds: Sequence[str] = ()
    en_build_modes: Sequence[str] = ()
    build_opts: Sequence[str] = ()
    post_build_opts: Sequence[str] = ()
    build_timeout_mins: int | None = None
    pre_run_cmds: Sequence[str] = ()
    post_run_cmds: Sequence[str] = ()
    run_opts: Sequence[str] = ()
    sw_images: Sequence[str] = ()
    sw_build_opts: Sequence[str] = ()


class RunModeConfig(BaseModel):
    """Schema for an entry of `run_modes` (see `dvsim.modes.RunMode`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    reseed: int | None = None
    pre_run_cmds: Sequence[str] = ()
    post_run_cmds: Sequence[str] = ()
    en_run_modes: Sequence[str] = ()
    run_opts: Sequence[str] = ()
    uvm_test: str = ""
    uvm_test_seq: str = ""
    build_mode: str = ""
    run_timeout_mins: int | None = None
    run_timeout_multiplier: float | None = None
    sw_images: Sequence[str] = ()
    sw_build_device: str = ""
    sw_build_opts: Sequence[str] = ()


class TestConfig(RunModeConfig):
    """Schema for an entry of `tests` (see `dvsim.test.Test`).

    A test is a run mode with the same set of attributes.
    """


class RegressionConfig(BaseModel):
    """Schema for an entry of `regressions` (see `dvsim.regression.Regression`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tests: Sequence[str] | None = None
    """Tests in this regression; absent/None means "run all available tests"."""
    excl_tests: Sequence[str] = ()
    reseed: int | None = None
    en_sim_modes: Sequence[str] = ()
    en_run_modes: Sequence[str] = ()
    pre_build_cmds: Sequence[str] = ()
    post_build_cmds: Sequence[str] = ()
    pre_run_cmds: Sequence[str] = ()
    post_run_cmds: Sequence[str] = ()
    build_opts: Sequence[str] = ()
    post_build_opts: Sequence[str] = ()
    run_opts: Sequence[str] = ()


class SimFlowConfig(FlowConfig):
    """Schema for a sim flow config.

    Typed fields cover the keys that dvsim itself reads; further
    project-specific keys (wildcard substitution variables) are allowed and
    checked by the base class. As with `FlowConfig`, defaults document the
    effective `SimCfg` defaults - dump with `exclude_unset=True` so only the
    keys actually present in the hjson data get merged.
    """

    model_config = ConfigDict(extra="allow")

    # Structural sections.
    build_modes: Sequence[BuildModeConfig] = ()
    run_modes: Sequence[RunModeConfig] = ()
    tests: Sequence[TestConfig] = ()
    regressions: Sequence[RegressionConfig] = ()
    en_build_modes: Sequence[str] = ()
    en_run_modes: Sequence[str] = ()

    # Testbench / DUT.
    dut: str = ""
    dut_instance: str = ""
    tb: str = ""
    testplan: str = ""
    testplan_doc_path: str = ""
    vplan: str = ""
    fusesoc_core: str = ""
    ral_spec: str = ""
    sim_tops: Sequence[str] = ()
    timescale: str = ""

    # Build.
    build_mode: str = ""
    primary_build_mode: str = ""
    reseed: int | None = None
    flow_makefile: str = ""
    build_cmd: str = ""
    build_dir: str = ""
    build_ex: str = ""
    build_db_dir: str = ""
    build_seed_file_path: str = ""
    pre_build_cmds: Sequence[str] = ()
    post_build_cmds: Sequence[str] = ()
    build_opts: Sequence[str] = ()
    post_build_opts: Sequence[str] = ()
    build_pass_patterns: Sequence[str] = ()
    build_fail_patterns: Sequence[str] = ()

    # Run.
    run_cmd: str = ""
    run_dir: str = ""
    run_dir_name: str = ""
    run_script: str = ""
    pre_run_cmds: Sequence[str] = ()
    post_run_cmds: Sequence[str] = ()
    run_opts: Sequence[str] = ()
    run_pass_patterns: Sequence[str] = ()
    run_fail_patterns: Sequence[str] = ()
    pass_patterns: Sequence[str] = ()
    fail_patterns: Sequence[str] = ()
    verbosity: str = ""
    supported_wave_formats: Sequence[str] = ()

    # Software build collateral.
    sw_root_dir: str = ""
    sw_images: Sequence[str] = ()
    sw_build_device: str = ""
    sw_build_opts: Sequence[str] = ()
    sw_build_cmd: str | Sequence[str] = ""

    # File list generation.
    sv_flist: str = ""
    sv_flist_gen_cmd: str = ""
    sv_flist_gen_dir: str = ""
    sv_flist_gen_flags: Sequence[str] = ()
    sv_flist_gen_opts: Sequence[str] = ()
    fusesoc_cores_root_dirs: Sequence[str] = ()
    post_flist_opts: str = ""

    # Coverage collection, merging and reporting.
    cov_metrics: str = ""
    cov_db_dir: str = ""
    cov_db_test_dir: str = ""
    cov_db_test_dir_name: str = ""
    cov_work_dir: str = ""
    cov_merge_cmd: str = ""
    cov_merge_dir: str = ""
    cov_merge_db_dir: str = ""
    cov_merge_opts: Sequence[str] = ()
    cov_report_cmd: str = ""
    cov_report_dir: str = ""
    cov_report_opts: Sequence[str] = ()
    cov_report_page: str = ""
    cov_report_txt: str = ""
    cov_analyze_cmd: str = ""
    cov_analyze_dir: str = ""
    cov_analyze_opts: Sequence[str] = ()
    cov_unr_dir: str = ""
    cov_unr_metrics: str = ""
    cov_unr_build_cmd: str | Sequence[str] = ""
    cov_unr_build_opts: Sequence[str] = ()
    cov_unr_common_build_opts: Sequence[str] = ()
    cov_unr_run_cmd: str | Sequence[str] = ""
    cov_unr_run_opts: Sequence[str] = ()
    cov_vplan_prepare_opts: Sequence[str] = ()
    cov_vplan_process_opts: Sequence[str] = ()


def validate_sim_cfg_data(path: str, hjson_data: Mapping) -> dict:
    """Validate merged sim cfg hjson data against the schema.

    Args:
        path: config file the data was loaded from (used in error messages).
        hjson_data: merged hjson data as returned by `load_hjson`.

    Returns:
        Only the keys actually present in `hjson_data`, with the structural
        sections (tests, modes, regressions, overrides) validated and
        normalised back to plain dicts/lists.

    Raises:
        RuntimeError: if the data does not match the schema.

    """
    try:
        model = SimFlowConfig.model_validate(dict(hjson_data))
    except ValidationError as err:
        msg = f"{path!r}: sim flow config does not match the schema:\n{err}"
        raise RuntimeError(msg) from err

    return model.model_dump(exclude_unset=True)
