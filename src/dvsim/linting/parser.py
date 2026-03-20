# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for lint flow hjson configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hjson
from pydantic import ValidationError

from dvsim.linting.config import LintBatchConfig, LintBlockConfig, MessageBucket
from dvsim.logging import log
from dvsim.utils.wildcards import find_and_substitute_wildcards

if TYPE_CHECKING:
    from dvsim.config import ProjectConfig


def _extract_wildcards(
    data: dict[str, Any], wildcard_values: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract string/int/bool values from config to use as wildcard values.

    If values contain template variables, resolves them using the provided
    wildcard_values before extracting. This allows chained template resolution
    where later imports can reference variables defined in earlier imports.

    Args:
        data: Configuration dictionary
        wildcard_values: Existing wildcard values to use for resolution

    Returns:
        Dictionary of field names to their resolved values

    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str | int | bool | Path):
            # If value contains templates and we have wildcard_values, resolve it first
            if isinstance(value, str) and "{" in value and wildcard_values:
                try:
                    value = find_and_substitute_wildcards(
                        value, wildcard_values=wildcard_values, ignore_error=True
                    )
                except ValueError:
                    # Skip if circular reference or other error
                    continue
            result[key] = value
    return result


def _resolve_templates(value: Any, project_config: ProjectConfig) -> Any:  # noqa: ANN401
    """Recursively resolve template variables using wildcard substitution.

    Args:
        value: Value to resolve (can be str, list, dict, or other)
        project_config: Project configuration for template resolution

    Returns:
        Value with templates resolved

    """
    # Build wildcard values from project config
    wildcard_values = project_config.to_context()

    # If value is a dict, extract all simple values as wildcards
    # This allows fields to reference other fields in the same config
    if isinstance(value, dict):
        wildcard_values.update(_extract_wildcards(value, wildcard_values))

    return find_and_substitute_wildcards(
        value,
        wildcard_values=wildcard_values,
        ignore_error=True,  # Allow unresolved wildcards for validation
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deeply merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary whose values take precedence

    Returns:
        Merged dictionary

    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_and_merge_imports(
    hjson_path: Path,
    project_config: ProjectConfig | None = None,
    visited: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a config file and recursively merge all imports.

    Args:
        hjson_path: Path to the configuration file
        project_config: Project configuration for template resolution
        visited: Set of already visited paths to detect circular imports

    Returns:
        Merged configuration dictionary

    Raises:
        FileNotFoundError: If a config file doesn't exist
        RuntimeError: If there's a circular import or parsing error

    """
    if visited is None:
        visited = set()

    hjson_path = hjson_path.resolve()

    if hjson_path in visited:
        msg = f"Circular import detected: {hjson_path}"
        raise RuntimeError(msg)

    visited.add(hjson_path)

    if not hjson_path.exists():
        msg = f"Configuration file not found: {hjson_path}"
        raise FileNotFoundError(msg)

    # Load the config file
    try:
        with hjson_path.open() as f:
            config_data = hjson.load(f, use_decimal=True)
    except hjson.HjsonDecodeError as e:
        msg = f"Failed to parse hjson file {hjson_path}: {e}"
        raise RuntimeError(msg) from e
    except OSError as e:
        msg = f"Failed to read hjson file {hjson_path}: {e}"
        raise RuntimeError(msg) from e

    # Start with empty config
    merged_config: dict[str, Any] = {}

    # Process imports first (so main config can override)
    if "import_cfgs" in config_data and project_config is not None:
        import_paths = config_data["import_cfgs"]
        if not isinstance(import_paths, list):
            import_paths = [import_paths]

        for import_path_str in import_paths:
            log.debug(f"Processing import: {import_path_str}")
            log.debug(f"Project config: proj_root={project_config.proj_root}")

            # Build wildcard context from project config + merged data + current file
            # This allows imports to reference values defined in the same file
            wildcard_values = project_config.to_context()
            wildcard_values.update(_extract_wildcards(merged_config, wildcard_values))
            wildcard_values.update(_extract_wildcards(config_data, wildcard_values))

            # Resolve template variables in import path
            import_path_resolved = find_and_substitute_wildcards(
                import_path_str,
                wildcard_values=wildcard_values,
                ignore_error=True,
            )
            log.debug(f"After template resolution: {import_path_resolved}")

            # Resolve relative paths relative to the current config file
            import_path = Path(import_path_resolved)
            if not import_path.is_absolute():
                import_path = (hjson_path.parent / import_path).resolve()
                log.debug(f"Resolved relative path: {import_path}")

            # Recursively load and merge the import
            imported_config = _load_and_merge_imports(import_path, project_config, visited.copy())
            merged_config = _deep_merge(merged_config, imported_config)

    # Merge the main config (overriding imports)
    return _deep_merge(merged_config, config_data)


def parse_lint_flow_config(
    hjson_path: str | Path, project_config: ProjectConfig | None = None
) -> LintBatchConfig | LintBlockConfig:
    """Parse a lint flow hjson configuration file.

    Loads the configuration file and recursively merges all imports before
    validation. Resolves template variables using the provided project config.
    Automatically detects whether the config is a batch config (with use_cfgs)
    or a block config (individual lint job).

    Args:
        hjson_path: Path to the hjson configuration file
        project_config: Project configuration for template resolution

    Returns:
        LintBatchConfig if config has use_cfgs, LintBlockConfig otherwise

    Raises:
        FileNotFoundError: If the hjson file doesn't exist
        ValidationError: If the hjson data doesn't match the expected schema
        RuntimeError: If there are other errors parsing the hjson

    Example:
        >>> from dvsim.config import ProjectConfig
        >>> proj_cfg = ProjectConfig(
        ...     proj_root=Path("/path/to/project"),
        ...     tool="ascentlint",
        ...     scratch_path=Path("/tmp/scratch")
        ... )
        >>> config = parse_lint_flow_config("path/to/lint_cfg.hjson", proj_cfg)
        >>> print(config.name)

    """
    hjson_path = Path(hjson_path)

    # Load and merge all imports
    hjson_data = _load_and_merge_imports(hjson_path, project_config)

    # Resolve template variables in all values
    if project_config is not None:
        hjson_data = _resolve_templates(hjson_data, project_config)

    # Convert message_buckets and use_cfgs from dicts to objects
    # Pass project_config and base_path for merging block imports
    _convert_nested_objects(hjson_data, project_config, hjson_path.parent)

    # Remove flow field - it's only used for detection, not stored in the model
    hjson_data.pop("flow", None)

    # Remove import_cfgs - no longer needed after merging
    hjson_data.pop("import_cfgs", None)

    # Determine if this is a batch config or block config
    has_use_cfgs = "use_cfgs" in hjson_data and hjson_data["use_cfgs"]

    # Validate against the appropriate pydantic model
    try:
        if has_use_cfgs:
            return LintBatchConfig(**hjson_data)
        return LintBlockConfig(**hjson_data)
    except ValidationError as e:
        config_type = "batch" if has_use_cfgs else "block"
        msg = f"Lint {config_type} config validation failed for {hjson_path}:\n{e}"
        raise RuntimeError(msg) from e


def _merge_block_imports(
    block_dict: dict[str, Any],
    project_config: ProjectConfig | None,
    base_path: Path,
) -> dict[str, Any]:
    """Merge imports for a single block configuration.

    Args:
        block_dict: Block configuration dictionary
        project_config: Project configuration for template resolution
        base_path: Base path for resolving relative import paths

    Returns:
        Merged block configuration dictionary

    """
    merged = {}

    # Process imports if present
    if "import_cfgs" in block_dict and project_config is not None:
        import_paths = block_dict["import_cfgs"]
        if not isinstance(import_paths, list):
            import_paths = [import_paths]

        for import_path_str in import_paths:
            # Resolve template variables in import path
            import_path_str = _resolve_templates(import_path_str, project_config)

            # Resolve relative paths
            import_path = Path(import_path_str)
            if not import_path.is_absolute():
                import_path = (base_path / import_path).resolve()

            # Load and merge the import
            imported_config = _load_and_merge_imports(import_path, project_config, set())
            merged = _deep_merge(merged, imported_config)

    # Merge the block config itself (overriding imports)
    return _deep_merge(merged, block_dict)


def _convert_nested_objects(
    data: dict, project_config: ProjectConfig | None = None, base_path: Path | None = None
) -> None:
    """Convert nested dicts to Pydantic objects in-place.

    Args:
        data: Configuration dict to process
        project_config: Project configuration for template resolution
        base_path: Base path for resolving relative import paths

    """
    # Convert message_buckets
    if "message_buckets" in data:
        data["message_buckets"] = [
            MessageBucket(**bucket) if isinstance(bucket, dict) else bucket
            for bucket in data["message_buckets"]
        ]

    # Convert use_cfgs - merge imports for each block first
    if "use_cfgs" in data:
        converted_use_cfgs = []
        for cfg in data["use_cfgs"]:
            if isinstance(cfg, dict):
                # Merge imports for this block
                if base_path is not None:
                    merged_cfg = _merge_block_imports(cfg, project_config, base_path)
                else:
                    merged_cfg = cfg

                # Resolve templates in the merged config
                if project_config is not None:
                    merged_cfg = _resolve_templates(merged_cfg, project_config)

                # Remove flow and import_cfgs from block
                merged_cfg.pop("flow", None)
                merged_cfg.pop("import_cfgs", None)

                # Recursively convert nested objects in block configs
                _convert_nested_objects(merged_cfg, project_config, base_path)
                converted_use_cfgs.append(LintBlockConfig(**merged_cfg))
            else:
                converted_use_cfgs.append(cfg)
        data["use_cfgs"] = converted_use_cfgs


def load_lint_config_from_dict(
    config_dict: dict, project_config: ProjectConfig | None = None
) -> LintBatchConfig | LintBlockConfig:
    """Load a lint flow configuration from a dictionary.

    Args:
        config_dict: Dictionary containing lint flow configuration
        project_config: Project configuration for template resolution

    Returns:
        LintBatchConfig if config has use_cfgs, LintBlockConfig otherwise

    Raises:
        ValidationError: If the dictionary doesn't match the expected schema

    Example:
        >>> config_dict = {"name": "test_lint"}
        >>> config = load_lint_config_from_dict(config_dict)

    """
    # Resolve template variables in all values
    if project_config is not None:
        config_dict = _resolve_templates(config_dict, project_config)

    _convert_nested_objects(config_dict)

    # Remove flow field - it's only used for detection, not stored in the model
    config_dict.pop("flow", None)

    # Remove import_cfgs - should be empty after merging
    config_dict.pop("import_cfgs", None)

    has_use_cfgs = "use_cfgs" in config_dict and config_dict["use_cfgs"]

    if has_use_cfgs:
        return LintBatchConfig(**config_dict)
    return LintBlockConfig(**config_dict)
