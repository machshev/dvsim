# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models describing the sim flow configuration schema.

`SimFlowConfig` validates the merged hjson data for a sim flow config (the
output of `dvsim.flow.hjson.load_hjson`); `SimCfg` holds the validated model
as its config state. The mode / test / regression sub-schemas are defined next
to the classes they configure (`dvsim.modes`, `dvsim.test`, `dvsim.regression`)
and reject unknown keys, catching typos at load time instead of deep inside
mode merging.
"""

from collections.abc import Mapping

from pydantic import ConfigDict, Field, ValidationError

from dvsim.flow.config import FlowConfig
from dvsim.modes import BuildModeConfig, RunModeConfig
from dvsim.regression import RegressionConfig
from dvsim.test import TestConfig

__all__ = (
    "BuildModeConfig",
    "RegressionConfig",
    "RunModeConfig",
    "SimFlowConfig",
    "TestConfig",
    "load_sim_flow_config",
)


class SimFlowConfig(FlowConfig):
    """Schema for a sim flow config.

    Typed fields cover the keys that dvsim itself reads; further
    project-specific keys (wildcard substitution variables) are allowed and
    checked by the base class. `SimCfg` holds an instance of this model as its
    config state, with the field defaults serving as the config defaults.
    """

    model_config = ConfigDict(extra="allow")

    # Structural sections.
    build_modes: list[BuildModeConfig] = Field(default_factory=list)
    run_modes: list[RunModeConfig] = Field(default_factory=list)
    tests: list[TestConfig] = Field(default_factory=list)
    regressions: list[RegressionConfig] = Field(default_factory=list)
    en_build_modes: list[str] = Field(default_factory=list)
    en_run_modes: list[str] = Field(default_factory=list)

    # Testbench / DUT.
    dut: str = ""
    dut_instance: str = ""
    tb: str = ""
    testplan: str = ""
    testplan_doc_path: str = ""
    vplan: str = ""
    fusesoc_core: str = ""
    ral_spec: str = ""
    sim_tops: list[str] = Field(default_factory=list)
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
    pre_build_cmds: list[str] = Field(default_factory=list)
    post_build_cmds: list[str] = Field(default_factory=list)
    build_opts: list[str] = Field(default_factory=list)
    post_build_opts: list[str] = Field(default_factory=list)
    build_pass_patterns: list[str] = Field(default_factory=list)
    build_fail_patterns: list[str] = Field(default_factory=list)

    # Run.
    run_cmd: str = ""
    run_dir: str = ""
    run_dir_name: str = ""
    run_script: str = ""
    pre_run_cmds: list[str] = Field(default_factory=list)
    post_run_cmds: list[str] = Field(default_factory=list)
    run_opts: list[str] = Field(default_factory=list)
    run_pass_patterns: list[str] = Field(default_factory=list)
    run_fail_patterns: list[str] = Field(default_factory=list)
    pass_patterns: list[str] = Field(default_factory=list)
    fail_patterns: list[str] = Field(default_factory=list)
    verbosity: str = ""
    supported_wave_formats: list[str] = Field(default_factory=list)

    # Software build collateral.
    sw_root_dir: str = ""
    sw_images: list[str] = Field(default_factory=list)
    sw_build_device: str = ""
    sw_build_opts: list[str] = Field(default_factory=list)
    sw_build_cmd: str | list[str] = ""

    # File list generation.
    sv_flist: str = ""
    sv_flist_gen_cmd: str = ""
    sv_flist_gen_dir: str = ""
    sv_flist_gen_flags: list[str] = Field(default_factory=list)
    sv_flist_gen_opts: list[str] = Field(default_factory=list)
    fusesoc_cores_root_dirs: list[str] = Field(default_factory=list)
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
    cov_merge_opts: list[str] = Field(default_factory=list)
    cov_report_cmd: str = ""
    cov_report_dir: str = ""
    cov_report_opts: list[str] = Field(default_factory=list)
    cov_report_page: str = ""
    cov_report_txt: str = ""
    cov_analyze_cmd: str = ""
    cov_analyze_dir: str = ""
    cov_analyze_opts: list[str] = Field(default_factory=list)
    cov_unr_dir: str = ""
    cov_unr_metrics: str = ""
    cov_unr_build_cmd: str | list[str] = ""
    cov_unr_build_opts: list[str] = Field(default_factory=list)
    cov_unr_common_build_opts: list[str] = Field(default_factory=list)
    cov_unr_run_cmd: str | list[str] = ""
    cov_unr_run_opts: list[str] = Field(default_factory=list)
    cov_vplan_prepare_opts: list[str] = Field(default_factory=list)
    cov_vplan_process_opts: list[str] = Field(default_factory=list)


def load_sim_flow_config(path: str, hjson_data: Mapping) -> SimFlowConfig:
    """Validate merged sim cfg hjson data into a `SimFlowConfig` model.

    Args:
        path: config file the data was loaded from (used in error messages).
        hjson_data: merged hjson data as returned by `load_hjson`.

    Returns:
        The validated config model.

    Raises:
        RuntimeError: if the data does not match the schema.

    """
    try:
        return SimFlowConfig.model_validate(dict(hjson_data))
    except ValidationError as err:
        msg = f"{path!r}: sim flow config does not match the schema:\n{err}"
        raise RuntimeError(msg) from err
