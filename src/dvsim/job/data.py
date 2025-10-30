# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Job data models."""

from collections.abc import Callable
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_type: str  # Deployment type

    target: str  # run phase [build, run, ...]
    flow: str  # Name of the flow config (e.g. tl_agent)

    full_name: str  # Full name (e.g. tl_agent:default)

    workspace_cfg: WorkspaceConfig

    # TODO: use ID rather than full JobSpec object
    dependencies: list["JobSpec"]
    needs_all_dependencies_passing: bool
    weight: int

    odir: Path  # Output directory

    pre_launch: Callable[[Launcher], None]
    post_finish: Callable[[str], None]


class CompletedJobStatus(BaseModel):
    """Job status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    fail_msg: ErrorMessage
