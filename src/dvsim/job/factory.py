# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for constructing job specs from flow configurations.

Flows construct JobSpec models from their hjson-driven configuration. The
helpers in this module are flow-agnostic: they operate on a duck-typed flow
configuration (see FlowConfigLike) and a plain wildcard-substitution
namespace, so each flow can construct its job specs directly without
depending on a shared flow base class.
"""

import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from dvsim.job.data import JobSpec, WorkspaceConfig
from dvsim.job.status import JobStatus
from dvsim.logging import log
from dvsim.report.data import IPMeta, ToolMeta
from dvsim.utils import find_and_substitute_wildcards

__all__ = (
    "FlowConfigLike",
    "construct_job_cmd",
    "is_equivalent_job_spec",
    "job_full_name",
    "job_namespace",
    "new_job_spec",
    "resolve_wildcards",
)

T = TypeVar("T")


class FlowConfigLike(Protocol):
    """The parts of a flow configuration that job spec construction uses."""

    name: str
    tool: str | None
    gui: bool
    interactive: bool
    branch: str
    commit: str
    commit_short: str
    revision: str
    workspace_cfg: WorkspaceConfig

    def wildcard_namespace(self) -> Mapping:
        """Return the flat mapping used for wildcard substitution."""
        ...


def resolve_wildcards(value: T, namespace: Mapping) -> T:
    """Recursively substitute wildcards in a config value."""
    return find_and_substitute_wildcards(
        obj=value,
        wildcard_values=namespace,
        ignored_wildcards=None,
        ignore_error=False,
    )


def job_full_name(cfg: FlowConfigLike, qual_name: str) -> str:
    """Construct a job's full name from its qualified name.

    The full name disambiguates across multiple cfgs being run (example:
    'aes:default', 'uart:default' builds). The cfg's variant, if it defines
    one, is folded into the name.
    """
    variant = getattr(cfg, "variant", None)
    if not (isinstance(variant, str) or variant is None):
        raise TypeError("Unexpected type for variant")

    suffix = f"_{variant}" if variant else ""
    return f"{cfg.name}{suffix}:{qual_name}"


def job_namespace(
    cfg: FlowConfigLike,
    *,
    attrs: Iterable[str],
    mode_dict: Mapping | None = None,
    **per_job: object,
) -> dict:
    """Construct the wildcard substitution namespace for a job.

    The namespace is the cfg's own wildcard namespace overlaid with the
    given job attrs — looked up in the mode dict first and the cfg's
    namespace second — plus any additional per-job values given as keyword
    arguments. An AttributeError is raised if a mandatory attr cannot be
    found in either source.

    The returned namespace holds raw (unsubstituted) values; job spec
    fields are computed from it with resolve_wildcards().
    """
    ns = dict(cfg.wildcard_namespace())
    ns["flow"] = cfg.name
    ns["gui"] = cfg.gui

    for attr in ("dry_run", "exports", "flow_makefile", *attrs):
        if mode_dict is not None and attr in mode_dict:
            ns[attr] = mode_dict[attr]
        elif attr not in ns:
            msg = f"Attribute {attr!r} not found for {cfg.name!r}."
            raise AttributeError(msg)

    ns.update(per_job)
    return ns


def construct_job_cmd(
    *,
    makefile: str,
    target: str,
    dry_run: bool,
    cmd_attrs: Mapping[str, object],
    cmds_list_vars: Sequence[str] = (),
) -> str:
    """Construct the command that will eventually be launched.

    The 'cmd_attrs' map make variable names to their (already resolved)
    values. The 'cmds_list_vars' name the attributes that are to be treated
    as "list of commands": lists that need to be joined with '&&' instead
    of a space.
    """
    cmd = f"make -f {makefile} {target}"
    if dry_run is True:
        cmd += " -n"
    for attr in sorted(cmd_attrs):
        value = cmd_attrs[attr]
        if type(value) is list:
            # Join attributes that are list of commands with '&&' to chain
            # them together when executed as a Make target's recipe.
            separator = " && " if attr in cmds_list_vars else " "
            value = separator.join(item.strip() for item in value)
        if type(value) is bool:
            value = int(value)
        if type(value) is str:
            value = shlex.quote(value.strip())
        cmd += f" {attr}={value}"
    return cmd


def new_job_spec(
    cfg: FlowConfigLike,
    *,
    job_type: str,
    target: str,
    name: str,
    qual_name: str,
    cmd: str,
    odir: str | Path,
    gui: bool,
    dry_run: bool,
    exports: Iterable[Mapping[str, str]],
    extra_exports: Mapping[str, str] | None = None,
    seed: int | None = None,
    weight: int = 1,
    dependencies: Sequence[JobSpec] = (),
    needs_all_dependencies_passing: bool = True,
    timeout_mins: float | None = None,
    renew_odir: bool = False,
    pass_patterns: Sequence[str] = (),
    fail_patterns: Sequence[str] = (),
    pre_launch: Callable[[], None] | None = None,
    post_finish: Callable[[JobStatus], None] | None = None,
) -> JobSpec:
    """Construct a job spec, filling in the cfg-derived boilerplate.

    The 'exports' are the key-value pairs to be exported to the subprocess'
    environment, given as a list of dicts (as loaded from the HJson); they
    are flattened to a single dict, with 'extra_exports' merged on top.

    The 'weight' represents the weight with which a job of this type is
    scheduled. Weights are roughly inversely proportional to the average
    runtime of the job type. The lower the runtime, the higher the chance
    it gets scheduled. It is useful to customize this only for job types
    that may coexist at a time.

    In GUI mode no timeout is applied.
    """
    # At this point, the configuration should have populated its tool field
    # (either from a command line argument or a value in the hjson that was
    # loaded. If not, we don't know what to do.
    if cfg.tool is None:
        msg = (
            "No tool selected in job configuration. It must either be "
            "specified in the hjson or passed with the --tool argument."
        )
        raise RuntimeError(msg)

    merged_exports = {k: str(v) for item in exports for k, v in item.items()}
    if extra_exports:
        merged_exports.update(extra_exports)

    def noop_pre_launch() -> None:
        """Perform additional pre-launch activities (callback)."""

    def noop_post_finish(_status: JobStatus) -> None:
        """Perform additional post-finish activities (callback)."""

    return JobSpec(
        name=name,
        job_type=job_type,
        target=target,
        # TODO: for now we always use the default configured backend, but it might be good
        # to allow different jobs to run on different backends in the future?
        backend=None,
        # Use the configured tool to determine the resources (licenses) that
        # are required. For now, we just assume that the tool itself is the
        # only resource needed.
        resources={cfg.tool.upper(): 1},
        seed=seed,
        full_name=job_full_name(cfg, qual_name),
        qual_name=qual_name,
        block=IPMeta(
            name=cfg.name,
            variant=getattr(cfg, "variant", None),
            commit=cfg.commit,
            commit_short=cfg.commit_short,
            branch=cfg.branch,
            url="",
            revision_info=cfg.revision,
        ),
        tool=ToolMeta(
            name=cfg.tool,
            version="",
        ),
        workspace_cfg=cfg.workspace_cfg,
        dependencies=[d.full_name for d in dependencies],
        needs_all_dependencies_passing=needs_all_dependencies_passing,
        weight=weight,
        timeout_mins=(None if gui else timeout_mins),
        cmd=cmd,
        exports=merged_exports,
        dry_run=dry_run,
        interactive=cfg.interactive,
        odir=Path(odir),
        renew_odir=renew_odir,
        log_path=Path(f"{odir}/{target}.log"),
        pre_launch=pre_launch if pre_launch is not None else noop_pre_launch,
        post_finish=post_finish if post_finish is not None else noop_post_finish,
        pass_patterns=pass_patterns,
        fail_patterns=fail_patterns,
    )


def is_equivalent_job_spec(existing: JobSpec, candidate: JobSpec) -> bool:
    """Check if two job specs would result in equivalent dispatched jobs.

    Determines if 'candidate' and 'existing' would behave exactly the same
    way when deployed. If so, then there is no point in keeping both. The
    caller can choose to discard 'candidate' and pick 'existing' instead. To
    do so, we check the final resolved 'cmd' & the exports. The 'name' field
    will be unique to each, so we take that out of the comparison.
    """
    # Check if the cmd field is identical.
    if candidate.cmd != existing.cmd.replace(existing.name, candidate.name):
        return False

    # Check if exports have identical set of keys.
    if candidate.exports.keys() != existing.exports.keys():
        return False

    # Check if exports have identical values.
    for key, val in candidate.exports.items():
        existing_val = existing.exports[key]
        if type(existing_val) is str:
            existing_val = existing_val.replace(existing.name, candidate.name)
        if val != existing_val:
            return False

    log.verbose('Job "%s" is equivalent to "%s"', existing.name, candidate.name)
    return True
