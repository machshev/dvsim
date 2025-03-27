# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test ConfigStore."""

from pprint import pformat

from hamcrest import assert_that, calling, equal_to, raises

from dvsim.config.store import ConfigStore


class TestConfigStore:
    """Test ConfigStore.

    This is a flexible KV config store to substitute for the flow cfg objects
    to facilitate a move towards more strongly typed config objects.
    """

    @staticmethod
    def test_initialise_with_nothing() -> None:
        """Test that a ConfigStore object can be created."""
        ConfigStore()

    @staticmethod
    def test_initialise_with_data() -> None:
        """Test that a ConfigStore object can be created."""
        cfg = ConfigStore({"a": 1})

        assert_that(cfg.a, equal_to(1))

    @staticmethod
    def test_attributes_can_be_created_and_deleted() -> None:
        """Test that a ConfigStore object can be created and deleted."""
        cfg = ConfigStore({"a": 1})
        cfg.b = 3

        assert_that(cfg.a, equal_to(1))
        assert_that(cfg.b, equal_to(3))

        del cfg.b

        assert_that(not hasattr(cfg, "b"))

    @staticmethod
    def test_muitable_mapping() -> None:
        """Test that a ConfigStore object can be created."""
        cfg = ConfigStore({"c": 10})

        assert_that(cfg.c, equal_to(10))
        assert_that(cfg["c"], equal_to(10))
        assert_that(len(cfg), equal_to(1))

        cfg["b"] = 2

        assert_that(cfg.b, equal_to(2))
        assert_that(cfg["b"], equal_to(2))
        assert_that(len(cfg), equal_to(2))

        del cfg["c"]

        assert_that("c" in cfg, equal_to(False))
        assert_that(len(cfg), equal_to(1))

        cfg.update({"d": 20, "e": 30})

        assert_that(cfg.d, equal_to(20))
        assert_that(cfg.e, equal_to(30))
        assert_that(len(cfg), equal_to(3))

        assert_that("d" in cfg, equal_to(True))
        assert_that(set(cfg.keys()), equal_to({"b", "d", "e"}))
        assert_that(set(cfg.values()), equal_to({2, 20, 30}))
        assert_that(
            set(cfg.items()),
            equal_to({("b", 2), ("d", 20), ("e", 30)}),
        )

    @staticmethod
    def test_str() -> None:
        """Test ConfigStore str format."""
        data = {"a": 1, "b": [1, 2, "test", ("xyz", 23)]}
        assert_that(str(ConfigStore(data)), equal_to(pformat(data)))

    @staticmethod
    def test_invalid_attribute_raises() -> None:
        """Test getattr with invalid attribute raises AttributeError."""
        cfg = ConfigStore({"c": 10})
        assert_that(
            calling(getattr).with_args(cfg, "invalid"),
            raises(AttributeError),
        )

    @staticmethod
    def test_getting_and_setting_private_attributes() -> None:
        """Test getting and setting private attributes still works."""
        cfg = ConfigStore({"c": 10})
        cfg._test = "value"

        assert_that(cfg._test, equal_to("value"))
        assert_that(cfg._data, equal_to({"c": 10}))

        del cfg._test

        assert_that(
            calling(getattr).with_args(cfg, "_test"),
            raises(AttributeError),
        )
