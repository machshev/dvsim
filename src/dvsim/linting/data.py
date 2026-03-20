# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic data models for lint flow results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from dvsim.job.status import JobStatus  # noqa: TC001
from dvsim.report.data import IPMeta, ToolMeta  # noqa: TC001


class LintMessageBucket(BaseModel):
    """Bucket of lint messages by severity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flow_error: list[str] = []
    flow_warning: list[str] = []
    flow_info: list[str] = []
    lint_error: list[str] = []
    lint_warning: list[str] = []
    lint_info: list[str] = []

    @property
    def total_errors(self) -> int:
        """Total number of errors (flow + lint)."""
        return len(self.flow_error) + len(self.lint_error)

    @property
    def total_warnings(self) -> int:
        """Total number of warnings (flow + lint)."""
        return len(self.flow_warning) + len(self.lint_warning)

    @property
    def total_infos(self) -> int:
        """Total number of infos (flow + lint)."""
        return len(self.flow_info) + len(self.lint_info)

    @property
    def total_messages(self) -> int:
        """Total number of messages."""
        return self.total_errors + self.total_warnings + self.total_infos


class LintJobResult(BaseModel):
    """Results for a single lint job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Job name (config name)."""

    messages: LintMessageBucket
    """Categorized lint messages."""

    job_runtime: float
    """Job runtime in seconds."""

    log_path: str
    """Path to the job log file."""

    status: JobStatus
    """Job execution status."""

    @property
    def passed(self) -> bool:
        """Check if job passed (execution succeeded and no errors)."""
        return self.status == JobStatus.PASSED and self.messages.total_errors == 0


class LintFlowResults(BaseModel):
    """Complete lint flow results for a single block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: IPMeta
    """Block metadata."""

    tool: ToolMeta
    """Tool metadata."""

    timestamp: str
    """Timestamp for this run."""

    branch: str
    """Git branch name."""

    name: str
    """Flow configuration name."""

    jobs: list[LintJobResult]
    """Results for all lint jobs."""

    @property
    def total_jobs(self) -> int:
        """Total number of jobs."""
        return len(self.jobs)

    @property
    def passed_jobs(self) -> int:
        """Number of jobs that passed."""
        return sum(1 for job in self.jobs if job.passed)

    @property
    def failed_jobs(self) -> int:
        """Number of jobs that failed."""
        return self.total_jobs - self.passed_jobs

    @property
    def total_errors(self) -> int:
        """Total errors across all jobs."""
        return sum(job.messages.total_errors for job in self.jobs)

    @property
    def total_warnings(self) -> int:
        """Total warnings across all jobs."""
        return sum(job.messages.total_warnings for job in self.jobs)

    @property
    def total_infos(self) -> int:
        """Total infos across all jobs."""
        return sum(job.messages.total_infos for job in self.jobs)

    @property
    def overall_status(self) -> Literal["PASSED", "FAILED"]:
        """Overall status of the lint flow."""
        return "PASSED" if self.failed_jobs == 0 else "FAILED"


class LintFlowSummary(BaseModel):
    """Summary of lint flow results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: IPMeta
    """Primary block metadata."""

    tool: ToolMeta
    """Tool metadata."""

    timestamp: str
    """Timestamp for this run."""

    branch: str
    """Git branch name."""

    name: str
    """Flow configuration name."""

    total_jobs: int
    """Total number of jobs."""

    passed_jobs: int
    """Number of jobs that passed."""

    failed_jobs: int
    """Number of jobs that failed."""

    total_errors: int
    """Total errors across all jobs."""

    total_warnings: int
    """Total warnings across all jobs."""

    total_infos: int
    """Total infos across all jobs."""

    overall_status: Literal["PASSED", "FAILED"]
    """Overall status of the lint flow."""

    @classmethod
    def from_flow_results(cls, results: LintFlowResults) -> LintFlowSummary:
        """Create summary from full results."""
        return cls(
            block=results.block,
            tool=results.tool,
            timestamp=results.timestamp,
            branch=results.branch,
            name=results.name,
            total_jobs=results.total_jobs,
            passed_jobs=results.passed_jobs,
            failed_jobs=results.failed_jobs,
            total_errors=results.total_errors,
            total_warnings=results.total_warnings,
            total_infos=results.total_infos,
            overall_status=results.overall_status,
        )
