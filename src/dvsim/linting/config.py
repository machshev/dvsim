# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for lint flow configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MessageBucket(BaseModel):
    """Categorization for lint messages."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="Message category (e.g., 'flow', 'lint')")
    severity: str = Field(..., description="Message severity (e.g., 'info', 'warning', 'error')")
    label: str = Field(default="", description="Label for the bucket")


class LintBlockConfig(BaseModel):
    """Configuration for an individual lint block/job.

    This model represents a complete, resolved configuration after all imports
    have been merged. All required fields must be present.
    """

    # Block identification
    name: str = Field(..., description="Name of the lint block")
    dut: str = Field(..., description="Design under test name")

    # FuseSoC configuration
    fusesoc_core: str = Field(..., description="FuseSoC core to use")
    additional_fusesoc_argument: str | None = Field(
        default=None, description="Additional FuseSoC argument (e.g., mapping)"
    )

    # Build configuration
    build_dir: str = Field(..., description="Build directory path")
    build_log: str = Field(..., description="Build log file path")
    build_cmd: str = Field(..., description="Build command")
    build_opts: list[str] = Field(..., description="Build options")

    # Lint-specific configuration
    is_style_lint: bool | None = Field(default=None, description="Whether this is a style lint run")
    report_severities: list[str] = Field(
        ...,
        description="Message severities to include in reports",
    )
    fail_severities: list[str] = Field(
        ...,
        description="Message severities that cause the flow to fail",
    )
    message_buckets: list[MessageBucket] = Field(..., description="Message bucket configuration")

    # Tool configuration
    tool: str | None = Field(default=None, description="Lint tool to use")

    # Paths
    rel_path: str = Field(..., description="Relative path for results")

    # Import configurations (empty after merging)
    import_cfgs: list[str] | None = Field(default=None, description="Configuration files to import")

    model_config = ConfigDict(frozen=True, extra="allow")


class LintBatchConfig(BaseModel):
    """Top-level configuration for a lint batch containing multiple blocks.

    This model represents a complete, resolved configuration after all imports
    have been merged. All required fields must be present.
    """

    # Batch identification
    name: str = Field(..., description="Name of the lint batch")

    # Batch-level paths
    rel_path: str = Field(..., description="Relative path for results")

    # Batch-level lint configuration (optional, can be overridden by blocks)
    report_severities: list[str] | None = Field(
        default=None,
        description="Message severities to include in reports",
    )
    fail_severities: list[str] | None = Field(
        default=None,
        description="Message severities that cause the flow to fail",
    )

    # Import configurations (empty after merging)
    import_cfgs: list[str] | None = Field(default=None, description="Configuration files to import")

    # Individual lint blocks
    use_cfgs: list[LintBlockConfig | dict] = Field(
        ..., description="Individual lint blocks to execute"
    )

    model_config = ConfigDict(frozen=True, extra="ignore")
