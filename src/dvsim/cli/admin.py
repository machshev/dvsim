# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""DVSim CLI main entry point."""

import sys
from importlib.metadata import version
from pathlib import Path

import click


@click.group()
@click.version_option(version("dvsim"))
def cli() -> None:
    """DVSim Administration tool.

    Temporary tool for administration tasks for a DVSim project. The commands
    here are experimental and may change at any time. As functionality
    stabilises it will be moved over to the main `dvsim` command.
    """


@cli.group()
def dashboard() -> None:
    """Dashboard helper commands."""


@dashboard.command("gen")
@click.argument(
    "json_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--base-url",
    default=None,
    type=str,
)
def dashboard_gen(json_path: Path, output_dir: Path, base_url: str | None) -> None:
    """Generate a dashboard from a existing results JSON."""
    from dvsim.sim.dashboard import gen_badges, gen_dashboard  # noqa: PLC0415
    from dvsim.sim.data import SimResultsSummary  # noqa: PLC0415

    results: SimResultsSummary = SimResultsSummary.load(path=json_path)

    gen_dashboard(
        summary=results,
        path=output_dir,
        base_url=base_url,
    )

    gen_badges(
        summary=results,
        path=output_dir,
    )


@cli.command()
@click.argument(
    "hjson_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--proj-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Project root directory (default: infer from git root)",
)
def check(hjson_file: Path, proj_root: Path | None) -> None:
    """Check a flow configuration file for validity."""
    from dvsim.check.flow import check_flow_config  # noqa: PLC0415
    from dvsim.config import ProjectConfig  # noqa: PLC0415
    from dvsim.linting.config import LintBatchConfig  # noqa: PLC0415
    from dvsim.utils.git import repo_root  # noqa: PLC0415

    # Infer proj_root if not provided
    if proj_root is None:
        proj_root = repo_root(hjson_file.parent)
        if proj_root is None:
            proj_root = hjson_file.parent

    # Ensure proj_root is absolute
    proj_root = proj_root.resolve()

    # Use placeholder values for check validation
    project_config = ProjectConfig(
        proj_root=proj_root,
        tool="ascentlint",
        scratch_path=Path("/tmp/scratch"),  # noqa: S108
    )

    success, message, flow_type, config = check_flow_config(hjson_file, project_config)

    if flow_type:
        click.echo(f"Flow type: {flow_type}")

    if success:
        click.secho(f"✓ {message}", fg="green")

        # If it's a batch config, list the child configs
        if isinstance(config, LintBatchConfig) and config.use_cfgs:
            click.echo(f"\nChild configurations ({len(config.use_cfgs)}):")
            for i, block_config in enumerate(config.use_cfgs, 1):
                click.secho(f"  {i}. ✓ {block_config.name}", fg="green")

        sys.exit(0)
    else:
        click.secho(f"✗ {message}", fg="red")
        sys.exit(1)


@cli.group()
def report() -> None:
    """Reporting helper commands."""


@report.command("gen")
@click.argument(
    "json_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
def report_gen(json_path: Path, output_dir: Path) -> None:
    """Generate a report from a existing results JSON."""
    from dvsim.sim.data import SimResultsSummary  # noqa: PLC0415
    from dvsim.sim.report import gen_reports  # noqa: PLC0415

    summary: SimResultsSummary = SimResultsSummary.load(path=json_path)
    flow_results = summary.load_flow_results(
        base_path=json_path.parent,
    )

    gen_reports(
        summary=summary,
        flow_results=flow_results,
        path=output_dir,
    )


if __name__ == "__main__":
    sys.exit(cli())
