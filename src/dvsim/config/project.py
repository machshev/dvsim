# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Project configuration for template variable resolution."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    """Project configuration for template variable resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proj_root: Path = Field(..., description="Project root directory")
    tool: str = Field(..., description="Tool name (e.g., ascentlint)")
    scratch_path: Path = Field(..., description="Scratch directory path")

    def to_context(self) -> dict[str, str]:
        """Convert to a context dictionary for template resolution.

        Returns:
            Dictionary mapping template variable names to their string values

        """
        return {
            "proj_root": str(self.proj_root),
            "tool": self.tool,
            "scratch_path": str(self.scratch_path),
        }
