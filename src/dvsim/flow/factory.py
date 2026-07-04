# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Factory for loading flow configurations from hjson config files."""

import pathlib
import sys
from collections.abc import Mapping

import hjson

import dvsim.instrumentation.runtime as instrumentation
from dvsim.flow.cdc import CdcCfg
from dvsim.flow.formal import FormalCfg
from dvsim.flow.group import Flow, FlowGroup
from dvsim.flow.hjson import load_hjson
from dvsim.flow.lint import LintCfg
from dvsim.flow.rdc import RdcCfg
from dvsim.flow.syn import SynCfg
from dvsim.logging import log
from dvsim.sim.flow import SimCfg
from dvsim.utils import rm_path, subst_wildcards

FLOW_HANDLERS = {
    "cdc": CdcCfg,
    "formal": FormalCfg,
    "lint": LintCfg,
    "rdc": RdcCfg,
    "sim": SimCfg,
    "syn": SynCfg,
}


def _load_cfg(path, initial_values):
    """Worker function for make_flow_group.

    initial_values is passed to load_hjson (see documentation there).

    Returns a pair (cls, hjson_data) on success or raises a RuntimeError on
    failure.

    """
    # Set the `self_dir` template variable to the path of the currently
    # processed Hjson file.
    assert "self_dir" in initial_values
    initial_values["self_dir"] = pathlib.Path(path).parent

    # Start by loading up the hjson file and any included files
    hjson_data = load_hjson(path, initial_values)

    # Look up the value of flow in the loaded data. This is a required field,
    # and tells us what sort of flow config to make.
    flow = hjson_data.get("flow")
    if flow is None:
        msg = (
            f'{path!r}: No value for the "flow" key. Are you sure '
            "this is a dvsim configuration file?"
        )
        raise RuntimeError(
            msg,
        )

    found_cls = FLOW_HANDLERS.get(flow)
    if found_cls is None:
        msg = '{}: Configuration file sets "flow" to {!r}, but this is not a known flow (known: {}).'.format(
            path,
            flow,
            ", ".join(sorted(FLOW_HANDLERS)),
        )
        raise RuntimeError(
            msg,
        )

    return (found_cls, hjson_data)


def _make_child_cfg(path, args, initial_values):
    try:
        cls, hjson_data = _load_cfg(path, initial_values)
    except RuntimeError as err:
        log.exception(str(err))
        sys.exit(1)

    # Since this is a child configuration (from some primary configuration),
    # make sure that we aren't ourselves a primary configuration. We don't need
    # multi-level hierarchies and this avoids circular dependencies.
    if "use_cfgs" in hjson_data:
        msg = (
            f"{path}: Configuration file has use_cfgs, but is "
            "itself included from another configuration."
        )
        raise RuntimeError(
            msg,
        )

    return cls(path, hjson_data, args)


def _conv_inline_cfg_to_hjson(idict: Mapping, args) -> str | None:
    """Dump a temp hjson file in the scratch space from an inline cfg dict."""
    name = idict.get("name", None)
    if not name:
        log.error(
            "In-line entry in use_cfgs list does not contain a "
            '"name" key (will be skipped!):\n%s',
            idict,
        )
        return None

    # Check if temp cfg file already exists
    temp_cfg_file = args.scratch_root + "/." + args.branch + "__" + name + "_cfg.hjson"

    # Create the file and dump the dict as hjson
    log.verbose('Dumping inline cfg "%s" in hjson to:\n%s', name, temp_cfg_file)

    try:
        pathlib.Path(temp_cfg_file).write_text(hjson.dumps(idict, for_json=True))

    except Exception as e:
        log.exception(
            'Failed to hjson-dump temp cfg file"%s" for "%s"(will be skipped!) due to:\n%s',
            temp_cfg_file,
            name,
            e,
        )
        return None

    # Return the temp cfg file created
    return temp_cfg_file


def _load_child_cfg(entry, primary: Flow, args, initial_values) -> Flow | None:
    """Load a child configuration of a primary cfg."""
    if type(entry) is str:
        # Treat this as a file entry. Substitute wildcards in cfg_file
        # files since we need to process them right away.
        cfg_file = subst_wildcards(entry, primary.wildcard_namespace(), ignore_error=True)
        return _make_child_cfg(cfg_file, args, initial_values)

    if type(entry) is dict:
        # Treat this as a cfg expanded in-line
        temp_cfg_file = _conv_inline_cfg_to_hjson(entry, args)
        if not temp_cfg_file:
            return None
        child = _make_child_cfg(temp_cfg_file, args, initial_values)

        # Delete the temp_cfg_file once the instance is created
        log.verbose("Deleting temp cfg file:\n%s", temp_cfg_file)
        rm_path(temp_cfg_file, ignore_error=True)
        return child

    log.error(
        'Type of entry "%s" in the "use_cfgs" key is invalid: %s',
        entry,
        str(type(entry)),
    )
    sys.exit(1)


def make_flow_group(path, args, proj_root) -> FlowGroup:
    """Make a flow group by loading the config file at path.

    args is the arguments passed to the dvsim.py tool and proj_root is the top
    of the project. A config with a `use_cfgs` key groups the listed child
    configs; any other config forms a group of one.

    """
    initial_values = {
        "proj_root": proj_root,
        "self_dir": pathlib.Path(path).parent,
    }
    if args.tool is not None:
        initial_values["tool"] = args.tool

    try:
        cls, hjson_data = _load_cfg(path, initial_values)
    except RuntimeError as err:
        log.exception(str(err))
        sys.exit(1)

    primary = cls(path, hjson_data, args)

    cfgs = [primary]
    if primary.is_primary_cfg:
        cfgs = []
        for entry in hjson_data["use_cfgs"]:
            child_ivs = initial_values.copy()
            child_ivs["flow"] = hjson_data["flow"]
            child = _load_child_cfg(entry, primary, args, child_ivs)
            if child is None:
                continue

            # Sanity check to make sure the child is the same class as the
            # primary: we don't yet support heterogeneous primary
            # configurations.
            if type(primary) is not type(child):
                log.error(
                    f"{path}: Loading child configuration at "
                    f"{child.flow_cfg_file!r}, but the resulting flow types "
                    f"don't match: ({type(primary).__name__} vs. "
                    f"{type(child).__name__}).",
                )
                sys.exit(1)

            cfgs.append(child)

    # Configure the report path for instrumentation
    reports_dir = pathlib.Path(primary.scratch_base_path) / "reports"
    instrumentation.set_report_path(reports_dir / "metrics.json")

    return FlowGroup(primary=primary, cfgs=cfgs, args=args)
