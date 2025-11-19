# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""DVSim CLI main entry point."""

import click


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx) -> None:
    """Entry point for DVSim."""
    if ctx.invoked_subcommand is None:
        from dvsim.cli.run import run  # noqa: PLC0415

        run()


if __name__ == "__main__":
    main()
