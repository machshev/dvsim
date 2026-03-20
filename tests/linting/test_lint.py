# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test new lint flow implementation."""

import argparse
from pathlib import Path

from hamcrest import assert_that, equal_to, is_

from dvsim.linting.lint import LintFlow


def _create_test_args() -> argparse.Namespace:
    """Create test arguments."""
    args = argparse.Namespace()
    args.tool = None
    args.items = []
    args.list = None
    return args


def test_lint_flow_init() -> None:
    """Test LintFlow can be initialized."""
    fixtures_dir = Path(__file__).parent.parent / "flow" / "fixtures"
    config_file = fixtures_dir / "example_lint.hjson"

    args = _create_test_args()
    lint_flow = LintFlow(config_file, args, "/tmp/proj")  # noqa: S108

    assert_that(lint_flow, is_(LintFlow))


def test_lint_flow_has_errors() -> None:
    """Test has_errors returns False."""
    fixtures_dir = Path(__file__).parent.parent / "flow" / "fixtures"
    config_file = fixtures_dir / "example_lint.hjson"

    args = _create_test_args()
    lint_flow = LintFlow(config_file, args, "/tmp/proj")  # noqa: S108

    assert_that(lint_flow.has_errors(), is_(False))


def test_lint_flow_deploy_objects() -> None:
    """Test deploy_objects returns empty list."""
    fixtures_dir = Path(__file__).parent.parent / "flow" / "fixtures"
    config_file = fixtures_dir / "example_lint.hjson"

    args = _create_test_args()
    lint_flow = LintFlow(config_file, args, "/tmp/proj")  # noqa: S108

    results = lint_flow.deploy_objects()

    assert_that(results, equal_to([]))


def test_lint_flow_gen_results() -> None:
    """Test gen_results can be called without errors."""
    fixtures_dir = Path(__file__).parent.parent / "flow" / "fixtures"
    config_file = fixtures_dir / "example_lint.hjson"

    args = _create_test_args()
    lint_flow = LintFlow(config_file, args, "/tmp/proj")  # noqa: S108

    # Should not raise an exception
    lint_flow.gen_results([])
