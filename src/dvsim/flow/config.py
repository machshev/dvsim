# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models describing the flow configuration schema.

These models validate the merged hjson data produced by
`dvsim.flow.hjson.load_hjson` before it is merged into a flow config. The
config namespace is deliberately open: projects define arbitrary additional
keys which are used as wildcard substitution variables (e.g. `{dv_root}` or
`{tl_aw}` in the OpenTitan configs). Unknown keys are therefore allowed, but
their values must be of a type that the wildcard substitution and merge
machinery understands.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

__all__ = (
    "FlowConfig",
    "OverrideConfig",
)

# Value types that hjson can produce and that `set_target_attribute` /
# `find_and_substitute_wildcards` know how to merge and expand.
_EXTRA_VALUE_TYPES = (str, int, float, bool, Path, list, dict)


class OverrideConfig(BaseModel):
    """A single entry of the `overrides` list.

    See `dvsim.flow.bootstrap.process_overrides`, which requires exactly these two keys.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Name of the config attribute to override."""
    value: str | int | float | bool
    """Value replacing the attribute's current value (must match its type)."""


class FlowConfig(BaseModel):
    """Schema for the config keys common to all dvsim flows.

    The field defaults serve as the config defaults for flows that hold this
    model as their config state (see `SimCfg`). Flows that instead merge the
    hjson data into their instance `__dict__` must dump with
    `exclude_unset=True` so only the keys actually present in the hjson data
    get merged.
    """

    model_config = ConfigDict(extra="allow")

    flow: str
    """The flow this config drives (e.g. "sim"). Selects the flow config class."""

    name: str = ""
    """Name of the DUT / config."""
    project: str = ""
    """Name of the wider project."""
    variant: str = ""
    """Optional variant of the config (e.g. "masked")."""
    tool: str | None = None
    """EDA tool to run the flow with (may instead come from --tool)."""

    proj_root: str = ""
    """Root of the project repository (injected by the config factory)."""
    self_dir: Path | None = None
    """Directory of the config file being loaded (injected by the factory)."""
    rel_path: str = ""
    """Results path relative to `proj_root`."""
    scratch_path: str = ""
    """Scratch area for this config's build and run artifacts."""
    scratch_base_path: str = ""
    """Base scratch area for the current branch."""

    use_cfgs: list[str | Mapping[str, Any]] = Field(default_factory=list)
    """Child config files (or inline configs) making this a primary config."""
    exports: list[Mapping[str, str | int | float | bool] | str] = Field(default_factory=list)
    """Variables exported to the environment of the launched jobs."""
    overrides: list[OverrideConfig] = Field(default_factory=list)
    """Config attribute overrides applied before wildcard expansion."""

    # Reporting / publishing.
    results_html_name: str = ""
    """Filename of the generated HTML results page."""
    results_server: str = ""
    """Results publishing server."""
    results_server_cmd: str = ""
    """Command used to upload results to the server."""
    results_server_prefix: str = ""
    """URL scheme prefix for the results server (e.g. "gs://")."""
    doc_server: str = ""
    """Documentation server."""
    repo_server: str = ""
    """Source repository server (e.g. "github.com/lowrisc/opentitan")."""
    book: str = ""
    """Base URL of the project book (documentation site)."""
    revision: str = ""
    """Revision string displayed in the results (may use {eval_cmd})."""
    results_title: str = ""
    """Title of the results report."""
    sanitize_publish_results: bool = False
    """Whether to sanitize the published results."""

    @model_validator(mode="after")
    def _check_extra_value_types(self) -> Self:
        """Check unknown keys hold values the merge/expansion machinery supports."""
        for key, value in (self.model_extra or {}).items():
            if value is None or not isinstance(value, _EXTRA_VALUE_TYPES):
                msg = (
                    f"Key {key!r} has value {value!r} of type "
                    f"{type(value).__name__}, which is not supported by the "
                    "config merge / wildcard expansion machinery. Supported "
                    "types: str, int, float, bool, Path, list, dict."
                )
                raise ValueError(msg)
        return self
