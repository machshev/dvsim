# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Modes are an abstraction for a collection of options and configuration for a dvsim job."""

import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import Self

from dvsim.logging import log


class BuildModeConfig(BaseModel):
    """Schema for an entry of `build_modes`.

    This is the single source of truth for the attributes of `BuildMode`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    is_sim_mode: int = 0
    pre_build_cmds: list[str] = []
    post_build_cmds: list[str] = []
    en_build_modes: list[str] = []
    build_opts: list[str] = []
    post_build_opts: list[str] = []
    build_timeout_mins: int | None = None
    pre_run_cmds: list[str] = []
    post_run_cmds: list[str] = []
    run_opts: list[str] = []
    sw_images: list[str] = []
    sw_build_opts: list[str] = []


class RunModeConfig(BaseModel):
    """Schema for an entry of `run_modes`.

    This is the single source of truth for the attributes of `RunMode`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    reseed: int | None = None
    pre_run_cmds: list[str] = []
    post_run_cmds: list[str] = []
    en_run_modes: list[str] = []
    run_opts: list[str] = []
    uvm_test: str = ""
    uvm_test_seq: str = ""
    build_mode: str = ""
    run_timeout_mins: int | None = None
    run_timeout_multiplier: float | None = None
    sw_images: list[str] = []
    sw_build_device: str = ""
    sw_build_opts: list[str] = []


class Mode:
    """A collection of options that represents a single mode.

    This might be a run mode (options for an EDA tool?), a build mode, a test
    or a regression. The mode's attributes are defined by its config schema
    (`config_cls`), which the raw hjson dicts are validated against.
    """

    # Set in subclasses: the schema describing this mode's attributes.
    config_cls: ClassVar[type[BaseModel]]

    def __init__(self, cfg: BaseModel) -> None:
        """Initialise mode attributes from a validated config model."""
        for key, value in cfg.model_dump().items():
            setattr(self, key, value)

    @classmethod
    def mode_from_dict(cls, mdict: Mapping[str, Any]) -> Self:
        """Create a mode from a raw dict, validating it against the mode's schema."""
        try:
            cfg = cls.config_cls.model_validate(mdict)
        except ValidationError as err:
            log.error(
                "Invalid %s entry %s:\n%s",
                cls.__name__,
                dict(mdict),
                err,
            )
            sys.exit(1)

        return cls(cfg)

    def get_sub_modes(self) -> Sequence[str]:
        # Default behaviour is not to have sub-modes
        return []

    def set_sub_modes(self, sub_modes: Sequence[str]) -> None:
        # Default behaviour is not to have sub-modes
        return None

    def merge_mode(self, mode: "Mode") -> None:
        """Update this object by merging it with mode."""
        sub_modes = self.get_sub_modes()
        is_sub_mode = mode.name in sub_modes

        # If the mode to be merged in is not known as a sub-mode of this mode
        # then something has gone wrong. Generate an error.
        if mode.name != self.name and not is_sub_mode:
            log.error(
                "Cannot merge mode %s with %s: it is not a sub-mode and they are not equal.",
                self.name,
                mode.name,
            )
            sys.exit(1)

        # Merge attributes in self with attributes in mode arg, since they are
        # the same mode but set in separate files, or a sub-mode.
        for attr, self_attr_val in self.__dict__.items():
            mode_attr_val = getattr(mode, attr, None)

            # If sub-mode, skip the name: it could differ.
            if is_sub_mode and attr == "name":
                continue

            # If the incoming  value is None, then nothing to do here.
            if mode_attr_val is None:
                continue

            # If the current value is None, then replace with the incoming value.
            if self_attr_val is None:
                setattr(self, attr, mode_attr_val)
                continue

            # If both values are equal, there is nothing to do.
            if self_attr_val == mode_attr_val:
                continue

            # If we have genuine types (because neither value is None), check
            # that the values are compatible.
            if not isinstance(mode_attr_val, type(self_attr_val)):
                log.error(
                    "Cannot merge %s with mode %s: the incoming values for "
                    "attribute %s are not of the same type.",
                    self.name,
                    mode.name,
                    attr,
                )
                sys.exit(1)

            # If the current value is a list, the incoming one must be as well.
            # Append that to the current list.
            if isinstance(self_attr_val, list):
                assert isinstance(mode_attr_val, list)
                self_attr_val.extend(mode_attr_val)
                continue

            # The types that we support other than lists are "scalar" types,
            # which each have a default value. The idea is that a default value
            # gets overridden by anything else.
            scalar_types = {str: "", int: -1}
            default_val = scalar_types.get(type(self_attr_val))

            # If the incoming value is the type's default value, it will have
            # no effect.
            if mode_attr_val == default_val:
                continue

            # If the existing value is the type's default value, it will be
            # overridden by the incoming value.
            if self_attr_val == default_val:
                setattr(self, attr, mode_attr_val)
                continue

            # If we get to here then neither value is the default value and
            # they are not equal. Raise an error because we don't know how to
            # merge them.
            log.error(
                "Cannot merge mode %s into %s because they have conflicting "
                "values for attribute %s: %s and %s.",
                mode.name,
                self.name,
                attr,
                mode_attr_val,
                self_attr_val,
            )
            sys.exit(1)

        # Check newly appended sub_modes, remove 'self' and duplicates
        sub_modes = self.get_sub_modes()

        if sub_modes != []:
            new_sub_modes = []
            for sub_mode in sub_modes:
                if self.name != sub_mode and sub_mode not in new_sub_modes:
                    new_sub_modes.append(sub_mode)
            self.set_sub_modes(new_sub_modes)

        return True

    @classmethod
    def create_modes(cls, mdicts: Iterable[Mapping[str, Any]]) -> list[Self]:
        """Create modes of type cls.

        Use the given list of raw dicts Process dependencies.

        Return a list of created objects.
        """

        def merge_sub_modes(mode, parent, objs) -> None:
            # Check if there are modes available to merge
            sub_modes = mode.get_sub_modes()
            if sub_modes == []:
                return

            # Set parent if it is None. If not, check cyclic dependency
            if parent is None:
                parent = mode
            elif mode.name == parent.name:
                log.error('Cyclic dependency when processing mode "%s"', mode.name)
                sys.exit(1)

            for sub_mode in sub_modes:
                # Find the sub_mode obj from str
                found = False
                for obj in objs:
                    if sub_mode == obj.name:
                        # First recursively merge the sub_modes
                        merge_sub_modes(obj, parent, objs)

                        # Now merge the sub mode with mode
                        mode.merge_mode(obj)
                        found = True
                        break
                if not found:
                    log.error(
                        'Sub mode "%s" added to mode "%s" was not found!',
                        sub_mode,
                        mode.name,
                    )
                    sys.exit(1)

        modes_objs = []
        # create a default mode if available
        default_mode = cls.get_default_mode()
        if default_mode is not None:
            modes_objs.append(default_mode)

        # Process list of raw dicts that represent the modes
        # Pass 1: Create unique set of modes by merging modes with the same name
        for mdict in mdicts:
            # Create a new item
            new_mode_merged = False
            new_mode = cls.mode_from_dict(mdict)
            for mode in modes_objs:
                # Merge new one with existing if available
                if mode.name == new_mode.name:
                    mode.merge_mode(new_mode)
                    new_mode_merged = True
                    break

            # Add the new mode to the list if not already appended
            if not new_mode_merged:
                modes_objs.append(new_mode)
                cls.item_names.append(new_mode.name)

        # Pass 2: Recursively expand sub modes within parent modes
        for mode in modes_objs:
            merge_sub_modes(mode, None, modes_objs)

        # Return the list of objects
        return modes_objs

    @staticmethod
    def get_default_mode(mode_type) -> None:
        return None


def find_mode(mode_name: str, modes: Sequence[Mode]) -> Mode | None:
    """Search through a list of modes and return the one with the given name.

    Return None if nothing was found.
    """
    for mode in modes:
        if mode_name == mode.name:
            return mode
    return None


def find_mode_list(mode_names: Sequence[str], modes: Sequence[Mode]) -> Sequence[Mode]:
    """Find modes matching a list of names."""
    found_list = []
    for mode_name in mode_names:
        mode = find_mode(mode_name, modes)
        if mode is None:
            log.error(
                "Cannot find requested mode (%s) in list. Known names: %s",
                mode_name,
                [m.name for m in modes],
            )
            sys.exit(1)

        found_list.append(mode)

    return found_list


class BuildMode(Mode):
    """Build modes."""

    # Maintain a list of build_modes str
    item_names = []

    config_cls = BuildModeConfig

    def __init__(self, cfg: BuildModeConfig) -> None:
        """Initialise a build mode from its validated config."""
        super().__init__(cfg)
        self.en_build_modes = list(set(self.en_build_modes))

    def get_sub_modes(self) -> Sequence[str]:
        return self.en_build_modes

    def set_sub_modes(self, sub_modes: Sequence[str]) -> None:
        self.en_build_modes = sub_modes

    @staticmethod
    def get_default_mode():
        return BuildMode(BuildModeConfig(name="default"))


class RunMode(Mode):
    """A collection of options for running a test."""

    # Maintain a list of run_modes str
    item_names = []

    config_cls = RunModeConfig

    def __init__(self, cfg: RunModeConfig) -> None:
        """Initialise a run mode from its validated config."""
        super().__init__(cfg)
        self.en_run_modes = list(set(self.en_run_modes))

    def get_sub_modes(self) -> list[str]:
        return self.en_run_modes

    def set_sub_modes(self, sub_modes: Sequence[str]) -> None:
        self.en_run_modes = sub_modes

    @staticmethod
    def get_default_mode() -> None:
        return None
