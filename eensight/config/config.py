# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from glob import iglob
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Set

import jinja2
from kedro.config import (
    AbstractConfigLoader,
    BadConfigException,
    MissingConfigException,
)
from omegaconf import OmegaConf

SUPPORTED_EXTENSIONS = [".yml", ".yaml"]

logger = logging.getLogger("ConfigLoader")


########################################################################################
# Utility functions
# From https://github.com/kedro-org/kedro/blob/0.18.1/kedro/config/common.py
########################################################################################


def _remove_duplicates(items: Iterable[Any]):
    """Remove duplicates while preserving the order."""
    unique_items = []  # type: List[str]
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
        else:
            logger.warning(
                f"Duplicate environment detected! "
                f"Skipping re-loading from configuration path: {item}"
            )
    return unique_items


def _check_duplicate_keys(
    processed_files: Dict[Path, AbstractSet[str]], filepath: Path, conf: Dict[str, Any]
) -> None:
    duplicates = []

    for processed_file, keys in processed_files.items():
        overlapping_keys = conf.keys() & keys

        if overlapping_keys:
            sorted_keys = ", ".join(sorted(overlapping_keys))
            if len(sorted_keys) > 100:
                sorted_keys = sorted_keys[:100] + "..."
            duplicates.append(f"{processed_file}: {sorted_keys}")

    if duplicates:
        dup_str = "\n- ".join(duplicates)
        raise ValueError(f"Duplicate keys found in {filepath} and:\n- {dup_str}")


def _path_lookup(conf_path: Path, patterns: Iterable[str]) -> Set[Path]:
    """Return a set of all configuration files from ``conf_path`` or
    its subdirectories, which satisfy a given list of glob patterns.
    """
    config_files = set()
    conf_path = conf_path.resolve()

    for pattern in patterns:
        # `Path.glob()` ignores the files if pattern ends with "**",
        # therefore iglob is used instead
        for each in iglob(str(conf_path / pattern), recursive=True):
            path = Path(each).resolve()
            if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
                config_files.add(path)

    return config_files


def _lookup_config_filepaths(
    conf_path: Path,
    patterns: Iterable[str],
    processed_files: Set[Path],
) -> List[Path]:
    config_files = _path_lookup(conf_path, patterns)

    seen_files = config_files & processed_files
    if seen_files:
        logger.warning(
            "Config file(s): %s already processed, skipping loading...",
            ", ".join(str(seen) for seen in sorted(seen_files)),
        )
        config_files -= seen_files

    return sorted(config_files)


##########################################################################################
# ConfigLoader
##########################################################################################


class ConfigLoader(AbstractConfigLoader):
    """Recursively scan directories (config paths) contained in ``conf_source`` for
    configuration files with a ``yaml`` or ``yml`` extension, load them, and return them
    in the form of a config dictionary.

    The first processed config path is the ``base`` directory inside ``conf_source``. The
    optional ``env`` argument can be used to specify a subdirectory of ``conf_source`` to
    process as a config path after ``base``. When the same top-level key appears in any 2
    config files located in the same (sub)directory, a ``ValueError`` is raised. When the
    same key appears in any 2 config files located in different (sub)directories, the last
    processed config path takes precedence and overrides this key.

    Args:
        conf_source (str): Path to use as root directory for loading configuration.
        env (str, optional): Environment that will take precedence over `base_env`. Defaults
            to None.
        runtime_params (dict, optional): Extra parameters passed to a Kedro run. Defaults to
            None.
        config_patterns: Regex patterns that specify the naming convention for configuration
            files so they can be loaded. Can be customised by supplying config_patterns as
            in `CONFIG_LOADER_ARGS` in `settings.py`.
        base_env (str, optional): Keyword-only argument for the name of the base environment.
            This is used in the `conf_paths` property method to construct the configuration
            paths. Defaults to "base".
        default_run_env (str, optional): Keyword-only argument for the name of the default run
            environment. This is used in the `conf_paths` property method to construct the
            configuration paths. Can be overriden by supplying the `env` argument. Defaults to
            "local".
        globals_pattern (str, optional): Keyword-only argument specifying a glob pattern. Files
            that match this pattern will be loaded as files containing global values for variable
            interpolation. Defaults to None.
    """

    def __init__(
        self,
        conf_source: str,
        env: str = None,
        runtime_params: Dict[str, Any] = None,
        config_patterns: Dict[str, List[str]] = None,
        *,
        base_env: str = "base",
        default_run_env: str = "local",
        globals_pattern: Optional[str] = None,
    ):
        super().__init__(
            conf_source=conf_source, env=env, runtime_params=runtime_params or {}
        )
        self.base_env = base_env
        self.default_run_env = default_run_env
        self.globals_pattern = globals_pattern

        self.config_patterns = {
            "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
            "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
            "credentials": ["credentials*", "credentials*/**", "**/credentials*"],
            "logging": ["logging*", "logging*/**", "**/logging*"],
        }
        self.config_patterns.update(config_patterns or {})

        self.conf_paths = _remove_duplicates(
            [
                Path(conf_source) / self.base_env,
                Path(conf_source) / (env or default_run_env),
            ]
        )

        if globals_pattern is not None:
            globals_dict = OmegaConf.to_container(
                OmegaConf.merge(
                    *[
                        OmegaConf.load(filename)
                        for filename in _path_lookup(
                            Path(conf_source) / base_env, [globals_pattern]
                        )
                    ]
                )
            )
            self.runtime_params.update({"globals": globals_dict})

    def __getitem__(self, key):
        return self.get(*self.config_patterns[key])

    def __repr__(self):  # pragma: no cover
        return (
            f"ConfigLoader(conf_source={self.conf_source}, env={self.env}, "
            f"config_patterns={self.config_patterns}), "
            f"globals_pattern='{self.globals_pattern}')"
        )

    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load an individual config file.

        Args:
            config_file (Path): Path to a config file to process.

        Raises:
            BadConfigException: If configuration is poorly formatted and
                cannot be loaded.

        Returns:
            dict: Parsed configuration as a dictionary.
        """
        try:
            conf = OmegaConf.create(
                jinja2.Template(config_file.read_text(), autoescape=True).render(
                    app=self.runtime_params.get("app", {}),
                    env=self.runtime_params.get("env", {}),
                    globals=self.runtime_params.get("globals", {}),
                )
            )
        except Exception as exc:
            raise BadConfigException(
                f"Couldn't load config file: {config_file}"
            ) from exc
        else:
            keys_to_drop = []
            for key in conf:
                if key.startswith("_"):
                    keys_to_drop.append(key)
            for key in keys_to_drop:
                del conf[key]
            return conf

    def _load_configs(self, config_filepaths: List[Path]) -> Dict[str, Any]:
        """Recursively load all configuration files, which satisfy
        a given list of glob patterns from a specific path.
        """
        aggregate_config = OmegaConf.create()
        seen_file_to_keys = {}

        for config_filepath in config_filepaths:
            single_config = self._load_config_file(config_filepath)
            _check_duplicate_keys(seen_file_to_keys, config_filepath, single_config)
            seen_file_to_keys[config_filepath] = single_config.keys()
            aggregate_config = OmegaConf.merge(aggregate_config, single_config)

        return aggregate_config

    def _get_config_from_patterns(
        self,
        patterns: Iterable[str] = None,
    ) -> Dict[str, Any]:

        if not patterns:
            raise ValueError(
                "`patterns` must contain at least one glob pattern to match filenames against."
            )

        config = OmegaConf.create()
        processed_files = set()  # type: Set[Path]

        for conf_path in self.conf_paths:
            if not conf_path.is_dir():
                raise ValueError(
                    f"Given configuration path either does not exist "
                    f"or is not a valid directory: {str(conf_path)}"
                )

            config_filepaths = _lookup_config_filepaths(
                conf_path, patterns, processed_files
            )

            new_conf = self._load_configs(config_filepaths)

            common_keys = config.keys() & new_conf.keys()
            if common_keys:
                sorted_keys = ", ".join(sorted(common_keys))
                msg = (
                    "Config from path `%s` will override the following "
                    "existing top-level config keys: %s"
                )
                logger.info(msg, conf_path, sorted_keys)

            config = OmegaConf.merge(config, new_conf)
            processed_files |= set(config_filepaths)

        if not processed_files:
            raise MissingConfigException(
                f"No files found in {self.conf_paths} matching the glob "
                f"pattern(s): {patterns}"
            )

        return OmegaConf.to_container(config, resolve=True)

    def get(self, *patterns: str) -> Dict[str, Any]:
        """Recursively scan for configuration files, load and merge them, and
        return them in the form of a config dictionary.

        Args:
            *patterns: Glob patterns to match. Files with names matching any of
                the specified patterns will be processed.

        Returns:
            Dict[str, Any]:  A dict with the combined configuration from all configuration files.

        Note:
            Any keys that start with `_` will be ignored.
        """
        return self._get_config_from_patterns(patterns=list(patterns))
