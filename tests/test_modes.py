# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test mode / test / regression creation from validated configs."""

import pytest

from dvsim.modes import BuildMode, BuildModeConfig, RunMode
from dvsim.regression import Regression
from dvsim.test import Test as SimTest

__all__ = ()


@pytest.fixture(autouse=True)
def _reset_item_names():
    """Reset the class-level name registries between tests."""
    for cls in (BuildMode, RunMode, SimTest, Regression):
        cls.item_names = []


def test_build_mode_defaults_come_from_schema() -> None:
    """A mode created from a minimal dict has the schema's default attributes."""
    mode = next(m for m in BuildMode.create_modes([{"name": "waves"}]) if m.name == "waves")

    assert mode.name == "waves"
    assert mode.is_sim_mode == 0
    assert mode.build_opts == []
    assert mode.build_timeout_mins is None


def test_default_build_mode_is_created() -> None:
    """create_modes prepends the "default" build mode."""
    modes = BuildMode.create_modes([{"name": "waves"}])

    assert [m.name for m in modes] == ["default", "waves"]


def test_same_name_modes_are_merged() -> None:
    """Two entries with the same name merge their list options."""
    modes = RunMode.create_modes(
        [
            {"name": "gui", "run_opts": ["-gui"]},
            {"name": "gui", "run_opts": ["-verdi"]},
        ],
    )

    (gui,) = modes
    assert gui.run_opts == ["-gui", "-verdi"]


def test_sub_modes_are_merged() -> None:
    """A mode pulling in another via en_build_modes inherits its options."""
    modes = BuildMode.create_modes(
        [
            {"name": "cov", "build_opts": ["-cm line"]},
            {"name": "waves", "en_build_modes": ["cov"], "build_opts": ["-debug"]},
        ],
    )

    waves = next(m for m in modes if m.name == "waves")
    assert waves.build_opts == ["-debug", "-cm line"]


def test_invalid_mode_key_exits() -> None:
    """An unknown key in a mode dict is a fatal, user-facing error."""
    with pytest.raises(SystemExit):
        BuildMode.create_modes([{"name": "waves", "biuld_opts": ["-debug"]}])


def test_mode_without_name_exits() -> None:
    """A mode dict without a name is a fatal, user-facing error."""
    with pytest.raises(SystemExit):
        RunMode.create_modes([{"run_opts": ["-gui"]}])


def test_regression_accepts_testplan_stage_shape() -> None:
    """Regressions produced by Testplan.get_stage_regressions validate."""
    reg = Regression.mode_from_dict({"name": "V1", "tests": ["uart_smoke"]})

    assert reg.tests == ["uart_smoke"]
    assert reg.test_names == []


def test_regression_tests_default_to_none() -> None:
    """An absent tests key means "run all available tests" (None sentinel)."""
    reg = Regression.mode_from_dict({"name": "nightly"})

    assert reg.tests is None


def test_get_default_mode() -> None:
    """The default build mode is constructed from the schema."""
    default = BuildMode.get_default_mode()

    assert default.name == "default"
    assert default.en_build_modes == []


def test_mode_from_config_model() -> None:
    """Modes can be constructed directly from a config model."""
    mode = BuildMode(BuildModeConfig(name="waves", build_opts=["-debug"]))

    assert mode.name == "waves"
    assert mode.build_opts == ["-debug"]
