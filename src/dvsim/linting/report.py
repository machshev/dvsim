# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lint flow report generation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from tabulate import tabulate

from dvsim.logging import log
from dvsim.report.artifacts import ReportArtifacts, render_static_content
from dvsim.templates.render import render_template

if TYPE_CHECKING:
    from pathlib import Path

    from dvsim.linting.data import LintFlowResults, LintFlowSummary, LintJobResult


class JsonReportRenderer:
    """Renderer for JSON lint reports."""

    format_name = "json"

    def render(
        self,
        summary: LintFlowSummary,
        flow_results: LintFlowResults,
        outdir: Path | None = None,
    ) -> ReportArtifacts:
        """Generate JSON report artifacts.

        Args:
            summary: Lint flow summary
            flow_results: Complete lint flow results
            outdir: Output directory (not used, for API compatibility)

        Returns:
            Dictionary of file paths to JSON content

        """
        artifacts = {}

        # Block-level results
        block_file = "lint.json"
        artifacts[block_file] = flow_results.model_dump_json(indent=2)

        # Summary
        artifacts["index.json"] = summary.model_dump_json(indent=2)

        return artifacts


class MarkdownReportRenderer:
    """Renderer for Markdown lint reports."""

    format_name = "md"

    def render(
        self,
        summary: LintFlowSummary,
        flow_results: LintFlowResults,
        outdir: Path | None = None,
    ) -> ReportArtifacts:
        """Generate Markdown report artifacts.

        Args:
            summary: Lint flow summary
            flow_results: Complete lint flow results
            outdir: Output directory (not used, for API compatibility)

        Returns:
            Dictionary of file paths to Markdown content

        """
        artifacts = {}

        # Block-level markdown report
        block_file = f"{flow_results.name}_lint_report.md"
        artifacts[block_file] = self._render_block_report(flow_results)

        # Summary markdown report
        artifacts["index.md"] = self._render_summary_report(summary, flow_results)

        return artifacts

    def _render_block_report(self, results: LintFlowResults) -> str:
        """Render block-level markdown report.

        Args:
            results: Complete lint flow results

        Returns:
            Markdown content

        """
        lines = []

        # Header
        lines.append(f"## {results.name.upper()} Lint Results\n")
        lines.append(f"### {datetime.now().strftime('%A %B %d %Y %H:%M:%S UTC')}")
        lines.append(f"### Branch: {results.branch}")
        lines.append(f"### Tool: {results.tool.name.upper()}\n")

        # Summary table
        lines.append(self._render_summary_table(results))
        lines.append("")

        # Message details for each job
        lines.extend(self._render_job_messages(job) for job in results.jobs)

        return "\n".join(lines)

    def _render_summary_report(
        self, summary: LintFlowSummary, results: LintFlowResults
    ) -> str:
        """Render summary markdown report.

        Args:
            summary: Lint flow summary
            results: Complete lint flow results (for detailed info)

        Returns:
            Markdown content

        """
        lines = []

        # Header
        lines.append("# Lint Results Summary\n")
        lines.append(f"**Config:** {summary.name}")
        lines.append(f"**Branch:** {summary.branch}")
        lines.append(f"**Tool:** {summary.tool.name}")
        lines.append(f"**Status:** {summary.overall_status}\n")

        # Overall statistics
        lines.append("## Overall Statistics\n")
        lines.append(f"- Total jobs: {summary.total_jobs}")
        lines.append(f"- Passed: {summary.passed_jobs}")
        lines.append(f"- Failed: {summary.failed_jobs}")
        lines.append(f"- Total errors: {summary.total_errors}")
        lines.append(f"- Total warnings: {summary.total_warnings}")
        lines.append(f"- Total infos: {summary.total_infos}\n")

        # Job results table
        lines.append("## Job Results\n")
        lines.append(self._render_summary_table(results))

        return "\n".join(lines)

    def _render_summary_table(self, results: LintFlowResults) -> str:
        """Render summary table of all jobs.

        Args:
            results: Complete lint flow results

        Returns:
            Markdown table

        """
        headers = [
            "Build Mode",
            "Flow Infos",
            "Flow Warnings",
            "Flow Errors",
            "Lint Infos",
            "Lint Warnings",
            "Lint Errors",
        ]

        rows = [[
                    job.name,
                    f"{len(job.messages.flow_info)} I",
                    f"{len(job.messages.flow_warning)} W",
                    f"{len(job.messages.flow_error)} E",
                    f"{len(job.messages.lint_info)} I",
                    f"{len(job.messages.lint_warning)} W",
                    f"{len(job.messages.lint_error)} E",
                ] for job in results.jobs]

        return tabulate(rows, headers=headers, tablefmt="pipe", stralign="center")

    def _render_job_messages(self, job: LintJobResult) -> str:
        """Render message details for a job.

        Args:
            job: Lint job result

        Returns:
            Markdown section with messages

        """
        lines = []

        lines.append(f"### Messages for Build Mode `{job.name}`\n")

        # Lint errors
        if job.messages.lint_error:
            lines.append("#### Lint Errors\n```")
            for msg in job.messages.lint_error[:50]:  # Limit to 50
                lines.append(msg)
            if len(job.messages.lint_error) > 50:
                lines.append(f"... and {len(job.messages.lint_error) - 50} more")
            lines.append("```\n")

        # Lint warnings
        if job.messages.lint_warning:
            lines.append("#### Lint Warnings\n```")
            for msg in job.messages.lint_warning[:50]:
                lines.append(msg)
            if len(job.messages.lint_warning) > 50:
                lines.append(f"... and {len(job.messages.lint_warning) - 50} more")
            lines.append("```\n")

        # Lint infos
        if job.messages.lint_info:
            lines.append("#### Lint Infos\n```")
            for msg in job.messages.lint_info[:50]:
                lines.append(msg)
            if len(job.messages.lint_info) > 50:
                lines.append(f"... and {len(job.messages.lint_info) - 50} more")
            lines.append("```\n")

        # Flow errors (if any)
        if job.messages.flow_error:
            lines.append("#### Flow Errors\n```")
            for msg in job.messages.flow_error:
                lines.append(msg)
            lines.append("```\n")

        # Flow warnings (if any)
        if job.messages.flow_warning:
            lines.append("#### Flow Warnings\n```")
            for msg in job.messages.flow_warning:
                lines.append(msg)
            lines.append("```\n")

        return "\n".join(lines)


class HtmlReportRenderer:
    """Renderer for HTML lint reports."""

    format_name = "html"

    def render(
        self,
        summary: LintFlowSummary,
        flow_results: LintFlowResults,
        outdir: Path | None = None,
    ) -> ReportArtifacts:
        """Generate HTML report artifacts.

        Args:
            summary: Lint flow summary
            flow_results: Complete lint flow results
            outdir: Output directory (for static assets)

        Returns:
            Dictionary of file paths to HTML/CSS/JS content

        """
        artifacts = {}

        # Block-level HTML report
        block_file = "lint.html"
        artifacts[block_file] = render_template(
            "lint/block_report.html",
            {
                "results": flow_results,
                "summary": summary,
            },
        )

        # Summary HTML report
        artifacts["index.html"] = render_template(
            "lint/summary_report.html",
            {
                "summary": summary,
                "results": flow_results,
            },
        )

        # Static assets (CSS, JS)
        if outdir:
            artifacts.update(
                render_static_content(
                    static_files=[
                        "css/style.css",
                        "css/bootstrap.min.css",
                        "js/bootstrap.bundle.min.js",
                        "js/htmx.min.js",
                    ],
                    outdir=outdir,
                )
            )

        return artifacts


def gen_reports(
    summary: LintFlowSummary,
    flow_results: LintFlowResults,
    report_path: Path,
) -> None:
    """Generate all lint reports.

    Args:
        summary: Lint flow summary
        flow_results: Complete lint flow results
        report_path: Base path for reports

    """
    # Generate JSON and HTML reports (write to disk)
    report_path.mkdir(parents=True, exist_ok=True)

    file_renderers = [
        JsonReportRenderer(),
        HtmlReportRenderer(),
    ]

    for renderer in file_renderers:
        log.info(f"Generating {renderer.format_name.upper()} reports...")

        artifacts = renderer.render(summary, flow_results, report_path)

        # Write artifacts to disk
        for filename, content in artifacts.items():
            filepath = report_path / filename
            filepath.write_text(content)
            log.info(f"  {filepath}")

    # Generate markdown report (print to console)
    log.info("Generating MARKDOWN report...")
    md_renderer = MarkdownReportRenderer()
    md_artifacts = md_renderer.render(summary, flow_results, None)

    # Print markdown to console
    for filename, content in md_artifacts.items():
        log.info(f"\n{'='*80}")
        log.info(f"Markdown Report: {filename}")
        log.info(f"{'='*80}\n")
        print(content)
        log.info(f"\n{'='*80}\n")

    log.info(f"Reports generated in {report_path}")
