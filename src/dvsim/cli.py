# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""dvsim is a tool to deploy ASIC tool flows.

It supports flows such as regressions for design verification (DV), formal
property verification (FPV), linting and synthesis.

It uses hjson as the format for specifying what to build and run. It is an
end-to-end regression manager that can deploy multiple builds (where some tests
might need different set of compile time options requiring a uniquely build sim
executable) in parallel followed by tests in parallel using the load balancer
of your choice.

dvsim is built to be tool-agnostic so that you can easily switch between the
tools at your disposal. dvsim uses fusesoc as the starting step to resolve all
inter-package dependencies and provide us with a filelist that will be consumed
by the sim tool.
"""

from collections.abc import Sequence
from pathlib import Path

import click


@click.group()
def main() -> None:
    """DVSim is a command line tool to deploy ASIC tool flows.

    Example flows are regressions for design verification (DV), formal property
    verification (FPV), linting and synthesis.

    It uses hjson as the format for specifying what to build and run. It is an
    end-to-end regression manager that can deploy multiple builds (where some tests
    might need different set of compile time options requiring a uniquely build sim
    executable) in parallel followed by tests in parallel using the load balancer
    of your choice.

    dvsim is built to be tool-agnostic so that you can easily switch between the
    tools at your disposal. dvsim uses fusesoc as the starting step to resolve all
    inter-package dependencies and provide us with a filelist that will be consumed
    by the sim tool.
    """


# Pass through for the old DVSim argparse cli
@main.command(
    context_settings={"ignore_unknown_options": True},
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=str)
def run(args: Sequence[str]) -> None:
    """Run DVSim."""
    from dvsim.cli_run import dvsim_run

    dvsim_run(args_list=args)


@main.group()
def report() -> None:
    """DVSim report commands."""


@report.command()
@click.argument(
    "run_path",
    default=None,
    type=click.Path(),
)
def generate(*, run_path: Path) -> None:
    """Generate a report."""
    from dvsim.report.generate import generate_report

    generate_report(run_path=Path(run_path))


if __name__ == "__main__":
    main()
