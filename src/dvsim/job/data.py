# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Job data models."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dvsim.launcher.base import ErrorMessage, Launcher

__all__ = (
    "CompletedJobStatus",
    "JobSpec",
    "WorkspaceConfig",
)


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str
    timestamp: str

    project_root: Path
    scratch_root: Path
    scratch_path: Path


class JobSpec(BaseModel):
    """Job specification."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # required for JobTime
    )

    job_type: str  # Deployment type

    target: str  # run phase [build, run, ...]
    flow: str  # Name of the flow config (e.g. tl_agent)
    tool: str

    name: str
    seed: int | None

    # Full name disambiguates across multiple cfg being run (example:
    # 'aes:default', 'uart:default' builds.
    full_name: str

    # Qualified name disambiguates the instance name with other instances
    # of the same class (example: 'uart_smoke' reseeded multiple times
    # needs to be disambiguated using the index -> '0.uart_smoke'.
    qual_name: str

    workspace_cfg: WorkspaceConfig

    dependencies: list[str]
    needs_all_dependencies_passing: bool
    weight: int
    timeout_mins: int | None

    cmd: str
    exports: Mapping[str, str]
    dry_run: bool
    interactive: bool
    gui: bool

    odir: Path  # Output directory
    log_path: Path
    links: Mapping[str, Path]

    # TODO: remove the need for these callables here
    pre_launch: Callable[[Launcher], None]
    post_finish: Callable[[str], None]

    pass_patterns: Sequence[str]
    fail_patterns: Sequence[str]


class CompletedJobStatus(BaseModel):
    """Job status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_type: str  # Deployment type
    name: str
    seed: int | None

    # Full name disambiguates across multiple cfg being run (example:
    # 'aes:default', 'uart:default' builds.
    full_name: str

    # Qualified name disambiguates the instance name with other instances
    # of the same class (example: 'uart_smoke' reseeded multiple times
    # needs to be disambiguated using the index -> '0.uart_smoke'.
    qual_name: str

    target: str  # run phase [build, run, ...]

    log_path: Path

    job_runtime: float
    simulated_time: float

    status: str
    fail_msg: ErrorMessage
