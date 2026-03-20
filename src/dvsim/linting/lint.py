# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lint flow handler."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from dvsim.config import ProjectConfig
from dvsim.job.data import CompletedJobStatus, JobSpec  # noqa: TC001
from dvsim.job.status import JobStatus
from dvsim.linting.config import LintBatchConfig, LintBlockConfig
from dvsim.linting.job import create_lint_job
from dvsim.linting.parser import parse_lint_flow_config
from dvsim.logging import log
from dvsim.scheduler.runner import build_default_scheduler_backend, run_scheduler

if TYPE_CHECKING:
    from collections.abc import Sequence


class LintFlow:
    """Executes lint checks on hardware designs."""

    def __init__(self, config_path: str | Path, args, proj_root: str | Path) -> None:
        """Initialize the lint flow.

        Args:
            config_path: Path to the lint flow configuration file
            args: Command line arguments
            proj_root: Project root directory

        """
        self._config_path = Path(config_path)
        self._args = args
        self._proj_root = Path(proj_root)

        log.info(f"Loading lint config from {self._config_path}")

        # Create ProjectConfig for template resolution
        project_config = ProjectConfig(
            proj_root=self._proj_root,
            tool=args.tool,
            scratch_path=Path(args.scratch_root),
        )

        self._config: LintBatchConfig | LintBlockConfig = parse_lint_flow_config(
            self._config_path, project_config
        )

        log.info(f"Loaded lint config: {self._config.name}")

        # Store additional context needed for job creation
        self._timestamp = getattr(args, "timestamp", "unknown")
        self._branch = getattr(args, "branch", "unknown")
        self._configs_to_run: list[LintBlockConfig] = []
        self._job_specs: list = []
        self._errors_seen = False

    def run(self) -> None:
        """Execute the lint flow.

        This is the main entry point for running the lint flow.
        """
        log.info("Starting lint flow execution")

        # Create deploy objects
        self._create_deploy_objects()

        # Deploy and execute
        results = self._deploy_objects()

        # Generate results
        self._gen_results(results)

    def has_errors(self) -> bool:
        """Check if lint found errors.

        Returns:
            True if errors found, False otherwise

        """
        return self._errors_seen

    def _create_deploy_objects(self) -> None:
        """Prepare lint jobs for execution."""
        log.info("Creating lint jobs")

        # Determine which configs to process
        if isinstance(self._config, LintBatchConfig):
            log.info(f"Processing batch config with {len(self._config.use_cfgs)} blocks")
            configs_to_run = self._config.use_cfgs
        else:
            log.info("Processing single block config")
            configs_to_run = [self._config]

        # Filter by --select-cfgs if specified
        if self._args.select_cfgs:
            selected = set(self._args.select_cfgs)
            configs_to_run = [cfg for cfg in configs_to_run if cfg.name in selected]
            log.info(
                f"Filtered to {len(configs_to_run)} selected configs: {[c.name for c in configs_to_run]}"
            )

        if not configs_to_run:
            log.warning("No configs selected to run")

        self._configs_to_run = configs_to_run
        log.info(f"Will create {len(configs_to_run)} lint job(s)")

        # Create job specs for each config
        self._job_specs = []
        for cfg in configs_to_run:
            job_spec = create_lint_job(
                config=cfg,
                args=self._args,
                proj_root=self._proj_root,
                scratch_root=Path(self._args.scratch_root),
                timestamp=self._timestamp,
                branch=self._branch,
            )

            self._job_specs.append(job_spec)

            log.info(f"Created job: {job_spec.name}")
            log.debug(f"  Command: {job_spec.cmd}")
            log.debug(f"  Working dir: {job_spec.odir}")
            log.debug(f"  Log file: {job_spec.log_path}")

        log.info(f"Created {len(self._job_specs)} job specification(s)")

    def _deploy_objects(self) -> Sequence[CompletedJobStatus]:
        """Execute lint jobs.

        Returns:
            List of completed job statuses

        """
        if not self._job_specs:
            log.warning("No jobs to execute")
            return []

        log.info(f"Executing {len(self._job_specs)} lint job(s)")

        (
            Path(self._args.scratch_root) / f"deploy_{self._branch}_{self._timestamp}.json"
        ).write_text(
            json.dumps(
                # Sort on full name to ensure consistent ordering
                sorted(
                    [
                        j.model_dump(
                            # callback functions can't be serialised
                            exclude={"pre_launch", "post_finish"},
                            mode="json",
                        )
                        for j in self._job_specs
                    ],
                    key=lambda j: j["full_name"],
                ),
                indent=2,
            ),
        )

        backend = build_default_scheduler_backend(
            fake_policy=self._fake_policy,
        )

        return asyncio.run(
            run_scheduler(
                jobs=self._job_specs,
                max_parallel=self._args.max_parallel,
                interactive=self._args.interactive,
                backend=backend,
            )
        )

    def _gen_results(self, results: Sequence[CompletedJobStatus]) -> None:
        """Generate results report.

        Args:
            results: Job results from execution

        """
        import hjson

        from dvsim.linting.data import (
            LintFlowResults,
            LintFlowSummary,
            LintJobResult,
            LintMessageBucket,
        )
        from dvsim.linting.report import gen_reports
        from dvsim.report.data import IPMeta, ToolMeta

        log.info("=" * 80)
        log.info(f"LINT RESULTS: {self._config.name}")
        log.info("=" * 80)

        if not results:
            log.warning("No results to report")
            return

        # Parse results.hjson files and build Pydantic models
        job_results = []
        for result in results:
            # Try to read results.hjson for this job
            results_file = Path(result.log_path).parent / "results.hjson"
            lint_messages = {}

            if results_file.exists():
                try:
                    with results_file.open() as f:
                        lint_messages = hjson.load(f)
                except Exception as e:
                    log.warning(f"Failed to parse {results_file}: {e}")

            # Create message bucket
            message_bucket = LintMessageBucket(
                flow_error=lint_messages.get("flow_error", []),
                flow_warning=lint_messages.get("flow_warning", []),
                flow_info=lint_messages.get("flow_info", []),
                lint_error=lint_messages.get("lint_error", []),
                lint_warning=lint_messages.get("lint_warning", []),
                lint_info=lint_messages.get("lint_info", []),
            )

            # Create job result
            job_result = LintJobResult(
                name=result.name,
                messages=message_bucket,
                job_runtime=result.job_runtime,
                log_path=str(result.log_path),
                status=result.status,
            )

            job_results.append(job_result)

            # Log each result
            status_symbol = "✓" if job_result.passed else "✗"
            log.info(f"{status_symbol} {result.name}:")
            log.info(f"  Log: {result.log_path}")
            log.info(f"  Runtime: {result.job_runtime:.2f}s")
            log.info(f"  Status: {result.status.name}")
            log.info(
                f"  Messages: {message_bucket.total_errors} errors, "
                f"{message_bucket.total_warnings} warnings, {message_bucket.total_infos} info"
            )

            # Show failure message if job execution failed
            if result.status != JobStatus.PASSED and result.fail_msg:
                log.error(f"  Failure: {result.fail_msg.message}")

            # Show sample error messages if present
            if message_bucket.total_errors > 0:
                sample_errors = message_bucket.flow_error[:3] + message_bucket.lint_error[:3]
                if sample_errors:
                    log.error("  Sample errors:")
                    for msg in sample_errors[:5]:
                        log.error(f"    {msg}")
                    if message_bucket.total_errors > 5:
                        log.error(f"    ... and {message_bucket.total_errors - 5} more")

        # Get tool from first result
        tool_name = results[0].tool.name if results and results[0].tool else "unknown"
        tool_version = results[0].tool.version if results and results[0].tool else ""

        # Create flow results model
        flow_results = LintFlowResults(
            block=IPMeta(
                name=self._config.name,
                variant="",
                commit="",
                commit_short="",
                branch=self._branch,
                url="",
                revision_info="",
            ),
            tool=ToolMeta(
                name=tool_name,
                version=tool_version,
            ),
            timestamp=self._timestamp,
            branch=self._branch,
            name=self._config.name,
            jobs=job_results,
        )

        # Create summary
        summary = LintFlowSummary.from_flow_results(flow_results)

        # Update errors_seen flag based on fail_severities (warnings don't cause failures)
        fail_severities = getattr(self._config, "fail_severities", None) or ["error"]
        if "error" in fail_severities and flow_results.total_errors > 0:
            self._errors_seen = True

        # Generate reports in scratch root directory
        report_path = Path(self._proj_root) / "scratch" / self._branch / "reports"
        gen_reports(summary, flow_results, report_path)

        # Print summary
        log.info("=" * 80)
        log.info("SUMMARY:")
        log.info(f"  Total jobs: {summary.total_jobs}")
        log.info(f"  Passed jobs: {summary.passed_jobs}")
        log.info(f"  Failed jobs: {summary.failed_jobs}")
        log.info(f"  Total errors: {summary.total_errors}")
        log.info(f"  Total warnings: {summary.total_warnings}")
        log.info(f"  Total info: {summary.total_infos}")
        log.info(f"  Overall status: {summary.overall_status}")

        if self._errors_seen:
            log.error("Lint found errors")
        else:
            log.info("All lint jobs passed!")

        log.info("=" * 80)

    def _fake_policy(self, _job: JobSpec) -> JobStatus:
        """Tell the fake backend how to fake jobs for this flow. Default flow always passes."""
        return JobStatus.PASSED
