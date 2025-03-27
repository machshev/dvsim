# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""A wrapper for loading hjson files as used by dvsim's FlowCfg."""

import pprint
from collections.abc import Iterator, Mapping, MutableMapping


class ConfigStore(MutableMapping):
    """Config store that acts as object and mapping."""

    def __init__(self, data: Mapping | None = None) -> None:
        """Initialise the store."""
        self._data = {}

        if data:
            self._data.update(data)

    def __str__(self) -> str:
        """Get string representation."""
        return pprint.pformat(self._data)

    def __getattr__(self, name: str) -> object:
        """Get a value from the store."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if name in self._data:
            return self._data[name]

        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: object) -> None:
        """Set a config value."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        self._data[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete key from store."""
        if name.startswith("_"):
            super().__delattr__(name)
            return

        del self._data[name]

    def __delitem__(self, key: str) -> None:
        """Delete item from store."""
        del self._data[key]

    def __getitem__(self, key: str) -> object:
        """Get an item from the store."""
        return self._data[key]

    def __iter__(self) -> Iterator[object]:
        """Get an iterator."""
        return self._data.__iter__()

    def __len__(self) -> int:
        """Get the number of items in the store."""
        return len(self._data)

    def __setitem__(self, key: str, value: object) -> None:
        """Set a config value."""
        self._data[key] = value
