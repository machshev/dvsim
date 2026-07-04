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

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from dvsim.logging import log

__all__ = (
    "FlowConfig",
    "OverrideConfig",
    "apply_config_expansion",
    "config_attr",
    "config_wildcard_namespace",
    "is_config_key",
    "load_flow_config",
    "merge_flow_config",
    "process_config_overrides",
)

# Value types that hjson can produce and that `set_target_attribute` /
# `find_and_substitute_wildcards` know how to merge and expand.
_EXTRA_VALUE_TYPES = (str, int, float, bool, Path, list, dict)


class OverrideConfig(BaseModel):
    """A single entry of the `overrides` list.

    See `process_config_overrides`, which requires exactly these two keys.
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


def load_flow_config(path: str, hjson_data: Mapping, model_cls: type[FlowConfig]) -> FlowConfig:
    """Validate merged cfg hjson data into a flow config model.

    Args:
        path: config file the data was loaded from (used in error messages).
        hjson_data: merged hjson data as returned by `load_hjson`.
        model_cls: the flow config model to validate against.

    Returns:
        The validated config model.

    Raises:
        RuntimeError: if the data does not match the schema.

    """
    try:
        return model_cls.model_validate(dict(hjson_data))
    except ValidationError as err:
        msg = f"{path!r}: flow config does not match the schema:\n{err}"
        raise RuntimeError(msg) from err


def is_config_key(config: FlowConfig, name: str) -> bool:
    """Whether `name` is a key managed by the config model."""
    return name in type(config).model_fields or name in (config.model_extra or {})


def config_attr(flow, name: str):
    """Fall back to the flow's config model for an attribute read.

    Intended to implement a flow's `__getattr__`: it is only consulted when
    normal attribute lookup fails, so runtime instance attributes (including
    ones shadowing config keys) take precedence. Only config keys are
    delegated - the model's own API is not exposed.
    """
    config = flow.__dict__.get("config")
    if (
        config is not None
        and not (name.startswith("__") and name.endswith("__"))
        and is_config_key(config, name)
    ):
        return getattr(config, name)

    msg = f"{type(flow).__name__!r} object has no attribute {name!r}"
    raise AttributeError(msg)


def merge_flow_config(
    flow,
    hjson_data: Mapping,
    *,
    model_cls: type[FlowConfig],
    cli_seeds: Mapping[str, list] | None = None,
) -> None:
    """Load the hjson data into the flow's typed config model.

    The validated model becomes the flow's config state (`flow.config`) and
    attribute reads fall through to it (see `config_attr`), with the schema
    field defaults serving as the config defaults.

    The 'cli_seeds' are command-line seeded list options that are folded
    into the config: the CLI values come first, matching the historic merge
    order where hjson values were appended to the CLI-seeded lists.
    """
    try:
        flow.config = load_flow_config(flow.flow_cfg_file, hjson_data, model_cls)
    except RuntimeError as err:
        log.error(str(err))
        sys.exit(1)

    # Drop the flow's instance defaults for keys the config model manages -
    # they would otherwise shadow the config.
    for key in [k for k in flow.__dict__ if is_config_key(flow.config, k)]:
        del flow.__dict__[key]

    for key, seed in (cli_seeds or {}).items():
        setattr(flow.config, key, [*seed, *getattr(flow.config, key)])


def process_config_overrides(flow) -> None:
    """Apply the typed overrides from the flow's config model."""
    overrides_seen = {}
    for override in flow.config.overrides:
        if override.name in overrides_seen:
            log.error(
                'Override for key "%s" already exists!\nOld: %s\nNew: %s',
                override.name,
                overrides_seen[override.name],
                override.value,
            )
            sys.exit(1)
        overrides_seen[override.name] = override.value
        _config_override(flow, override.name, override.value)


def _config_override(flow, ov_name: str, ov_value: object) -> None:
    """Override a single attribute, preferring runtime state over config."""
    in_instance = ov_name in flow.__dict__
    if in_instance:
        orig_value = flow.__dict__[ov_name]
    elif is_config_key(flow.config, ov_name):
        orig_value = getattr(flow.config, ov_name)
    else:
        log.error('Override key "%s" not found in the cfg!', ov_name)
        sys.exit(1)

    if not isinstance(ov_value, type(orig_value)):
        log.error(
            'The type of override value "%s" for "%s" '
            'doesn\'t match the type of original value "%s"',
            ov_value,
            ov_name,
            orig_value,
        )
        sys.exit(1)

    log.debug('Overriding "%s" value "%s" with "%s"', ov_name, orig_value, ov_value)
    if in_instance:
        flow.__dict__[ov_name] = ov_value
    else:
        setattr(flow.config, ov_name, ov_value)


def config_wildcard_namespace(flow) -> dict:
    """Merge the config model into the wildcard substitution namespace.

    Runtime instance attributes shadow config values of the same name.
    """
    namespace = flow.config.model_dump()
    namespace.update(flow.__dict__)
    return namespace


def apply_config_expansion(flow, expanded: Mapping) -> None:
    """Split an expanded namespace back into config and runtime state.

    Keys that live in the flow's instance `__dict__` (runtime state,
    including shadowed config keys) are updated in place; everything else is
    config data and is re-validated into a fresh config model.
    """
    model_cls = type(flow.config)
    instance_keys = set(flow.__dict__)
    cfg_data = {k: v for k, v in expanded.items() if k not in instance_keys}
    flow.__dict__.update((k, v) for k, v in expanded.items() if k in instance_keys)

    try:
        flow.config = model_cls.model_validate(cfg_data)
    except ValidationError as err:
        log.error(
            "%r: config is no longer schema-valid after wildcard expansion:\n%s",
            flow.flow_cfg_file,
            err,
        )
        sys.exit(1)
