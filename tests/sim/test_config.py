# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test the sim flow configuration schema."""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from dvsim.sim.config import load_sim_flow_config

__all__ = ()


MINIMAL_CFG = {
    "name": "uart",
    "flow": "sim",
    "tool": "vcs",
    "proj_root": "/proj",
}


def validate(path: str, hjson_data: Mapping) -> dict:
    """Validate hjson data and dump back only the keys that were present."""
    return load_sim_flow_config(path, hjson_data).model_dump(exclude_unset=True)


def test_minimal_cfg_roundtrips() -> None:
    """Only keys present in the input appear in the validated output."""
    validated = validate("<test>", MINIMAL_CFG)

    assert validated == MINIMAL_CFG


def test_structural_sections_are_validated_and_roundtrip() -> None:
    """Mode / test / regression / override entries survive validation intact."""
    data = {
        **MINIMAL_CFG,
        "build_modes": [{"name": "waves", "is_sim_mode": 1, "build_opts": ["-debug"]}],
        "run_modes": [{"name": "gui", "run_opts": ["-gui"]}],
        "tests": [{"name": "uart_smoke", "uvm_test_seq": "uart_smoke_vseq", "reseed": 5}],
        "regressions": [{"name": "smoke", "tests": ["uart_smoke"], "reseed": 1}],
        "overrides": [{"name": "scratch_path", "value": "/scratch/x"}],
        "exports": [{"SCRATCH_PATH": "{scratch_path}"}, {"VCS_LICENSE_WAIT": 1}],
    }

    validated = validate("<test>", data)

    assert validated == data


def test_project_specific_keys_are_allowed() -> None:
    """Unknown keys (wildcard substitution variables) pass through verbatim."""
    data = {
        **MINIMAL_CFG,
        "tl_aw": 32,
        "aes_model_core": "lowrisc:model:aes:1.0",
        "fusesoc_cores_root_dirs": ["--cores-root {proj_root}/hw"],
        "self_dir": Path("/proj/hw/ip/uart/dv"),
    }

    validated = validate("<test>", data)

    assert validated == data


def test_unknown_key_with_unsupported_value_type_is_rejected() -> None:
    """Extra keys must hold types the merge / wildcard machinery understands."""
    with pytest.raises(RuntimeError, match="bad_key"):
        validate("<test>", {**MINIMAL_CFG, "bad_key": None})


def test_typo_in_test_entry_is_rejected() -> None:
    """Unknown keys in a tests entry are caught at validation time."""
    data = {
        **MINIMAL_CFG,
        "tests": [{"name": "uart_smoke", "uvm_test_sequence": "oops"}],
    }

    with pytest.raises(RuntimeError, match="uvm_test_sequence"):
        validate("<test>", data)


def test_test_entry_requires_name() -> None:
    """A tests entry without a name is rejected."""
    with pytest.raises(RuntimeError, match="name"):
        validate("<test>", {**MINIMAL_CFG, "tests": [{"reseed": 3}]})


def test_malformed_override_is_rejected() -> None:
    """Override entries must be exactly {name, value}."""
    data = {
        **MINIMAL_CFG,
        "overrides": [{"name": "scratch_path", "value": "/scratch/x", "extra": True}],
    }

    with pytest.raises(RuntimeError, match="overrides"):
        validate("<test>", data)


def test_wrong_type_for_known_key_is_rejected() -> None:
    """Known keys are type checked."""
    with pytest.raises(RuntimeError, match="run_opts"):
        validate("<test>", {**MINIMAL_CFG, "run_opts": "-not-a-list"})


def _opentitan_root() -> Path:
    """Path to an OpenTitan checkout to validate real configs against."""
    return Path(os.environ.get("OPENTITAN_ROOT", "~/base/opentitan")).expanduser()


@pytest.mark.skipif(
    not (_opentitan_root() / "hw/top_earlgrey/dv/top_earlgrey_sim_cfgs.hjson").exists(),
    reason="OpenTitan checkout not available (set OPENTITAN_ROOT)",
)
def test_schema_against_opentitan_top_earlgrey() -> None:
    """Validate the schema against the full top_earlgrey sim cfg tree.

    Every cfg reachable from top_earlgrey_sim_cfgs.hjson must validate and
    round-trip unchanged (validation must not alter the config data the
    SimCfg consumes).
    """
    from dvsim.flow.hjson import load_hjson
    from dvsim.utils import subst_wildcards

    proj_root = _opentitan_root()
    primary_path = proj_root / "hw/top_earlgrey/dv/top_earlgrey_sim_cfgs.hjson"

    def check(path: Path, initial_values: dict) -> dict:
        data = load_hjson(str(path), initial_values)
        validated = validate(str(path), data)
        assert validated == dict(data), f"{path}: validation altered the config data"
        return data

    primary = check(
        primary_path,
        {"proj_root": str(proj_root), "self_dir": primary_path.parent},
    )

    child_paths = [
        subst_wildcards(entry, primary, ignore_error=True) for entry in primary.get("use_cfgs", [])
    ]
    assert child_paths, "expected use_cfgs in the primary config"

    for child_path in child_paths:
        check(
            Path(child_path),
            {
                "proj_root": str(proj_root),
                "self_dir": Path(child_path).parent,
                "flow": primary["flow"],
            },
        )
