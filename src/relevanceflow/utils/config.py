"""Configuration loading utilities for RelevanceFlow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """Raised when RelevanceFlow configuration is invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML configuration.

    Raises
    ------
    ConfigurationError
        If the file does not exist or does not contain a mapping.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    if not config_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {config_path}"
        ) from exc

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")

    return config


def load_configs(
    config_dir: str | Path = "configs",
) -> dict[str, dict[str, Any]]:
    """Load all RelevanceFlow YAML configuration files.

    Parameters
    ----------
    config_dir:
        Directory containing YAML configuration files.

    Returns
    -------
    dict[str, dict[str, Any]]
        Configuration keyed by filename stem.
    """
    directory = Path(config_dir)

    if not directory.exists():
        raise ConfigurationError(f"Configuration directory does not exist: {directory}")

    config_files = sorted(directory.glob("*.yaml"))

    if not config_files:
        raise ConfigurationError(f"No YAML configuration files found in: {directory}")

    configs: dict[str, dict[str, Any]] = {}

    for config_file in config_files:
        configs[config_file.stem] = load_yaml_config(config_file)

    return configs


def get_env(
    name: str,
    *,
    required: bool = False,
    default: str | None = None,
) -> str | None:
    """Read an environment variable.

    Parameters
    ----------
    name:
        Environment variable name.

    required:
        Whether the variable must exist and be non-empty.

    default:
        Value returned when the variable is not defined.

    Returns
    -------
    str | None
        Environment variable value.
    """
    value = os.getenv(name, default)

    if required and (value is None or not value.strip()):
        raise ConfigurationError(f"Required environment variable is missing: {name}")

    return value


def get_config_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Retrieve a nested configuration value.

    Example
    -------
    ``get_config_value(config, "training", "random_seed")``
    """
    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current
