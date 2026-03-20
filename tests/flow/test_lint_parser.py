# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Test lint flow configuration parser."""

from pathlib import Path

import hjson
import pytest
from hamcrest import assert_that, calling, equal_to, has_length, raises
from pydantic import ValidationError

from dvsim.config import ProjectConfig
from dvsim.linting.config import LintBlockConfig, MessageBucket
from dvsim.linting.parser import (
    _resolve_templates,
    load_lint_config_from_dict,
    parse_lint_flow_config,
)


def test_parse_minimal_lint_config(tmp_path: Path) -> None:
    """Test parsing a minimal lint block configuration."""
    config_data = {
        "name": "test_lint",
        "dut": "test",
        "fusesoc_core": "test:core",
        "build_dir": "/tmp/build",  # noqa: S108
        "build_log": "/tmp/build/test.log",  # noqa: S108
        "build_cmd": "fusesoc",
        "build_opts": ["run"],
        "report_severities": ["info", "warning", "error"],
        "fail_severities": ["warning", "error"],
        "message_buckets": [{"category": "lint", "severity": "error", "label": "Errors"}],
        "rel_path": "lint",
    }

    config_file = tmp_path / "test_lint.hjson"
    config_file.write_text(hjson.dumps(config_data))

    config = parse_lint_flow_config(config_file)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("test_lint"))
    assert_that(config.report_severities, equal_to(["info", "warning", "error"]))
    assert_that(config.fail_severities, equal_to(["warning", "error"]))


def test_parse_full_lint_config(tmp_path: Path) -> None:
    """Test parsing a full lint block configuration."""
    config_data = {
        "name": "aes_lint",
        "dut": "aes",
        "fusesoc_core": "lowrisc:dv:aes_sim",
        "additional_fusesoc_argument": "--mapping=lowrisc:systems:top_earlgrey:0.1",
        "build_dir": "{scratch_path}/{build_mode}",
        "build_log": "{build_dir}/{tool}.log",
        "build_cmd": "fusesoc",
        "build_opts": ["--cores-root {proj_root}/hw", "run"],
        "is_style_lint": False,
        "report_severities": ["info", "warning", "error"],
        "fail_severities": ["warning", "error"],
        "message_buckets": [
            {"category": "flow", "severity": "info", "label": ""},
            {"category": "flow", "severity": "warning", "label": ""},
            {"category": "flow", "severity": "error", "label": ""},
            {"category": "lint", "severity": "info", "label": ""},
            {"category": "lint", "severity": "warning", "label": ""},
            {"category": "lint", "severity": "error", "label": ""},
        ],
        "rel_path": "hw/ip/aes/lint",
    }

    config_file = tmp_path / "aes_lint.hjson"
    config_file.write_text(hjson.dumps(config_data))

    config = parse_lint_flow_config(config_file)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("aes_lint"))
    assert_that(config.dut, equal_to("aes"))
    assert_that(config.fusesoc_core, equal_to("lowrisc:dv:aes_sim"))
    assert_that(config.build_opts, has_length(2))
    assert_that(config.message_buckets, has_length(6))
    assert_that(config.message_buckets[0].category, equal_to("flow"))
    assert_that(config.message_buckets[0].severity, equal_to("info"))


def test_parse_style_lint_config(tmp_path: Path) -> None:
    """Test parsing a style lint block configuration."""
    config_data = {
        "name": "verible_style_lint",
        "dut": "test",
        "fusesoc_core": "test:core",
        "build_dir": "/tmp/build",  # noqa: S108
        "build_log": "/tmp/build/veriblelint.log",  # noqa: S108
        "build_cmd": "verible-verilog-lint",
        "build_opts": ["--rules_config={proj_root}/.verible.cfg"],
        "is_style_lint": True,
        "report_severities": ["warning", "error"],
        "fail_severities": ["error"],
        "message_buckets": [
            {"category": "lint", "severity": "warning", "label": "Style Warnings"},
            {"category": "lint", "severity": "error", "label": "Style Errors"},
        ],
        "tool": "veriblelint",
        "rel_path": "lint/style",
    }

    config_file = tmp_path / "style_lint.hjson"
    config_file.write_text(hjson.dumps(config_data))

    config = parse_lint_flow_config(config_file)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("verible_style_lint"))
    assert_that(config.is_style_lint, equal_to(True))
    assert_that(config.tool, equal_to("veriblelint"))


def test_load_lint_config_from_dict() -> None:
    """Test loading block configuration from a dictionary."""
    config_dict = {
        "name": "dict_test",
        "dut": "test_dut",
        "fusesoc_core": "test:core",
        "build_dir": "/tmp/build",  # noqa: S108
        "build_log": "/tmp/build/test.log",  # noqa: S108
        "build_cmd": "fusesoc",
        "build_opts": ["run"],
        "report_severities": ["error"],
        "fail_severities": ["error"],
        "message_buckets": [
            {"category": "lint", "severity": "error", "label": "Errors"},
        ],
        "rel_path": "lint/test",
    }

    config = load_lint_config_from_dict(config_dict)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("dict_test"))
    assert_that(config.message_buckets, has_length(1))
    assert_that(config.message_buckets[0].category, equal_to("lint"))


def test_message_bucket_model() -> None:
    """Test the MessageBucket pydantic model."""
    bucket = MessageBucket(category="flow", severity="warning", label="Flow Warnings")

    assert_that(bucket.category, equal_to("flow"))
    assert_that(bucket.severity, equal_to("warning"))
    assert_that(bucket.label, equal_to("Flow Warnings"))


def test_message_bucket_default_label() -> None:
    """Test MessageBucket with default label."""
    bucket = MessageBucket(category="lint", severity="info")

    assert_that(bucket.category, equal_to("lint"))
    assert_that(bucket.severity, equal_to("info"))
    assert_that(bucket.label, equal_to(""))


def test_parse_missing_file_raises() -> None:
    """Test that parsing a non-existent file raises FileNotFoundError."""
    assert_that(
        calling(parse_lint_flow_config).with_args("/nonexistent/file.hjson"),
        raises(FileNotFoundError),
    )


def test_parse_invalid_hjson_raises(tmp_path: Path) -> None:
    """Test that parsing invalid hjson raises RuntimeError."""
    config_file = tmp_path / "invalid.hjson"
    config_file.write_text("{invalid hjson content")

    # Should raise RuntimeError for invalid hjson syntax
    with pytest.raises(RuntimeError) as exc_info:
        parse_lint_flow_config(config_file)

    # Verify the error message mentions the file and parsing failure
    assert str(config_file) in str(exc_info.value)
    assert "parse" in str(exc_info.value).lower()


def test_missing_required_field_raises() -> None:
    """Test that missing required fields raise ValidationError."""
    config_dict = {
        # Missing required 'name' field
    }

    # ValidationError should be raised directly by Pydantic
    with pytest.raises(ValidationError) as exc_info:
        load_lint_config_from_dict(config_dict)

    # Verify the error mentions the missing 'name' field
    assert "name" in str(exc_info.value)


def test_extra_fields_forbidden() -> None:
    """Test that extra fields are forbidden in the configuration."""
    config_dict = {
        "name": "test",
        "flow": "lint",
        "custom_field": "custom_value",
    }

    # Should raise ValidationError for unknown fields
    with pytest.raises(ValidationError) as exc_info:
        load_lint_config_from_dict(config_dict)

    # Verify the error mentions the unexpected field
    assert "custom_field" in str(exc_info.value)


def test_integration_with_example_hjson() -> None:
    """Integration test with a realistic lint block hjson file."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    config_file = fixtures_dir / "example_lint.hjson"

    config = parse_lint_flow_config(config_file)

    # Build expected config
    expected = LintBlockConfig(
        name="aes_dv_lint",
        dut="aes",
        fusesoc_core="lowrisc:dv:aes_sim",
        additional_fusesoc_argument="--mapping=lowrisc:systems:top_earlgrey:0.1",
        build_dir="{scratch_path}/{build_mode}",
        build_log="{build_dir}/{tool}.log",
        build_cmd="fusesoc",
        build_opts=[
            "--cores-root {proj_root}/hw",
            "run",
            "--target={flow}",
            "--tool={tool}",
            "--work-root={build_dir}/fusesoc-work",
        ],
        is_style_lint=False,
        report_severities=["info", "warning", "error"],
        fail_severities=["warning", "error"],
        message_buckets=[
            MessageBucket(category="flow", severity="info", label="Flow Info"),
            MessageBucket(category="flow", severity="warning", label="Flow Warnings"),
            MessageBucket(category="flow", severity="error", label="Flow Errors"),
            MessageBucket(category="lint", severity="info", label="Lint Info"),
            MessageBucket(category="lint", severity="warning", label="Lint Warnings"),
            MessageBucket(category="lint", severity="error", label="Lint Errors"),
        ],
        tool="ascentlint",
        rel_path="hw/ip/aes/dv/lint/{tool}",
    )

    assert_that(config.model_dump(), equal_to(expected.model_dump()))


def test_resolve_templates_string() -> None:
    """Test template resolution in strings."""
    project_config = ProjectConfig(
        proj_root=Path("/home/user/project").absolute(),
        tool="ascentlint",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    result = _resolve_templates("{proj_root}/hw/data/file.hjson", project_config)

    assert_that(result, equal_to("/home/user/project/hw/data/file.hjson"))


def test_resolve_templates_absolute_path() -> None:
    """Test that resolved template creates absolute path."""
    project_config = ProjectConfig(
        proj_root=Path("/home/user/project").absolute(),
        tool="ascentlint",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    resolved = _resolve_templates("{proj_root}/hw/data/file.hjson", project_config)
    path = Path(resolved)

    assert_that(path.is_absolute(), equal_to(True))


def test_resolve_templates_list() -> None:
    """Test template resolution in lists."""
    project_config = ProjectConfig(
        proj_root=Path("/home/user/project").absolute(),
        tool="ascentlint",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    result = _resolve_templates(
        ["{proj_root}/file1.hjson", "{proj_root}/file2.hjson"], project_config
    )

    assert_that(
        result,
        equal_to(["/home/user/project/file1.hjson", "/home/user/project/file2.hjson"]),
    )


def test_resolve_templates_dict() -> None:
    """Test template resolution in dictionaries."""
    project_config = ProjectConfig(
        proj_root=Path("/home/user/project").absolute(),
        tool="ascentlint",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    result = _resolve_templates(
        {
            "path": "{proj_root}/hw",
            "tool": "{tool}",
            "build_dir": "{scratch_path}/{tool}",
        },
        project_config,
    )

    assert_that(
        result,
        equal_to(
            {
                "path": "/home/user/project/hw",
                "tool": "ascentlint",
                "build_dir": "/tmp/scratch/ascentlint",  # noqa: S108
            }
        ),
    )


def test_parse_with_template_imports() -> None:
    """Test parsing config with template variables in import paths."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    config_file = fixtures_dir / "template_main.hjson"

    project_config = ProjectConfig(
        proj_root=fixtures_dir.absolute(),
        tool="tool",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    config = parse_lint_flow_config(config_file, project_config)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("main_lint"))
    assert_that(config.dut, equal_to("test_dut"))
    assert_that(config.build_dir, equal_to("/tmp/scratch"))  # noqa: S108


def test_parse_with_chained_template_resolution() -> None:
    """Test that templates in later imports can reference values from earlier imports."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    config_file = fixtures_dir / "chain_main.hjson"

    project_config = ProjectConfig(
        proj_root=fixtures_dir.absolute(),
        tool="tool",
        scratch_path=Path("/tmp/scratch").absolute(),  # noqa: S108
    )

    config = parse_lint_flow_config(config_file, project_config)

    assert_that(isinstance(config, LintBlockConfig), equal_to(True))
    assert_that(config.name, equal_to("chain_test"))
    # tool_path should have lint_root resolved
    # lint_root is defined as {proj_root}/lint in chain_base
    # So tool_path should be {proj_root}/lint/tools = fixtures_dir/lint/tools
    expected_tool_path = f"{fixtures_dir.absolute()}/lint/tools"
    assert_that(config.model_dump().get("tool_path"), equal_to(expected_tool_path))
