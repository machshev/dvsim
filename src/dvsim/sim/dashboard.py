# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Generate dashboard.

The dashboard is a cut down version of the full report where a simpler summary
is required than the full report simulation summary. This is intended to be used
on a separate website and links back to the detailed report if required.

This is intended to generate a dashboard that could be used on the OpenTitan
and automatically i.e. https://opentitan.org/dashboard/index.html
"""

from pathlib import Path

from anybadge import Badge

from dvsim.logging import log
from dvsim.report.artifacts import render_static_content
from dvsim.sim.data import SimResultsSummary
from dvsim.templates.render import render_template

__all__ = ("gen_dashboard",)


def gen_dashboard(
    summary: SimResultsSummary,
    path: Path,
    base_url: str | None = None,
) -> None:
    """Generate a summary dashboard.

    Args:
        summary: overview of the block results
        path: output directory path
        base_url: override the base URL for links

    """
    log.debug("generating results dashboard")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the JS and CSS files
    render_static_content(
        static_files=[
            "css/style.css",
            "css/bootstrap.min.css",
            "js/bootstrap.bundle.min.js",
            "js/htmx.min.js",
        ],
        outdir=path,
    )

    (path / "dashboard.html").write_text(
        render_template(
            path="dashboard/dashboard.html",
            data={
                "summary": summary,
                "base_url": base_url,
            },
        )
    )

    (path / "dashboard.json").write_text(summary.model_dump_json())


def _badge_write(badge: Badge, path: Path) -> None:
    """Write a badge to file.

    Args:
        badge: the badge to write
        path: file path to write the badge to

    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        path.unlink()

    badge.write_badge(path)


def _badge_write_int(
    label: str,
    value: int | None,
    path: Path,
) -> None:
    """Write a badge to file.

    Args:
        label: badge text label
        value: badge value as int or
        path: file path to write the badge to

    """
    _badge_write(
        badge=Badge(
            label=label,
            value=value,
            default_color="blue",
        ),
        path=path,
    )


_COLOR_THRESHOLDS = {
    0.01: "#EF5757",
    10: "#EF6D57",
    20: "#EF8357",
    30: "#EF9957",
    40: "#EFAF57",
    50: "#EFC557",
    60: "#EFDB57",
    70: "#ECEF57",
    80: "#D6EF57",
    90: "#C0EF57",
    100: "#57EF57",
}
_TEXT_THRESHOLD = 20


def _badge_write_percent(
    label: str,
    value: float,
    path: Path,
) -> None:
    """Write a badge to file.

    Args:
        label: badge text label
        value: badge value as int or
        path: file path to write the badge to

    """
    _badge_write(
        badge=Badge(
            label=label,
            value=f"{value:.1f}" if value else "0",
            value_suffix="%",
            thresholds=_COLOR_THRESHOLDS,
            text_color="#fff," + ("#000" if value > _TEXT_THRESHOLD else "#fff"),
        ),
        path=path,
    )


def gen_badges(
    summary: SimResultsSummary,
    path: Path,
) -> None:
    """Generate a dashboard badges.

    Args:
        summary: overview of the block results
        path: output directory path
        base_url: override the base URL for links

    """
    base_path = path / "badge"

    for block, results in summary.flow_results.items():
        block_base = base_path / block

        _badge_write_int(
            label="Tests Running",
            value=results.total,
            path=block_base / "test.svg",
        )

        _badge_write_percent(
            label="Tests Passing",
            value=100 * results.passed / results.total,
            path=block_base / "passing.svg",
        )

        # Coverage
        if results.coverage is None:
            continue

        functional = results.coverage.functional or 0
        _badge_write_percent(
            label="Functional Coverage",
            value=functional,
            path=block_base / "functional.svg",
        )

        code = (
            results.coverage.code.average
            if results.coverage.code and results.coverage.code.average
            else 0
        )
        _badge_write_percent(
            label="Code Coverage",
            value=code,
            path=block_base / "code.svg",
        )

        assertion = results.coverage.assertion or 0
        _badge_write_percent(
            label="Assertion Coverage",
            value=assertion,
            path=block_base / "assertion.svg",
        )
