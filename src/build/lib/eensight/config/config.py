# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/config/config.py

from functools import reduce
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Union

from feature_encoders.utils import as_list
from kedro.config import BadConfigException, ConfigLoader
from kedro.config.config import MissingConfigException
from omegaconf import DictConfig, OmegaConf


def _check_duplicate_keys(
    processed_files: Dict[Path, AbstractSet[str]], filepath: Path, conf: Dict[str, Any]
):
    """Check duplicate keys.

    Copied from https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/config/config.py
    so that to avoid importing a private function from kedro.
    """
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


class OmegaConfigLoader(ConfigLoader):
    """Recursively scan the directories specified in ``conf_paths`` for
    configuration files with a ``yaml`` or ``yml`` extension, load them,
    and return them in the form of a config dictionary.

    When the same top-level key appears in any 2 config files located in
    the same ``conf_path`` (sub)directory, a ``ValueError`` is raised.
    When the same key appears in any 2 config files located in different
    ``conf_path`` directories, the last processed config path takes
    precedence and overrides this key.

    Args:
        conf_paths (str or list of str): Non-empty path or list of paths
            to configuration directories.
        globals_pattern (str, optional): Optional keyword-only argument
            specifying a glob pattern. Files that match this pattern will
            be loaded as files containing global values for variable
            interpolation. Defaults to None.

    Raises:
        ValueError: If ``conf_paths`` is empty.
    """

    def __init__(
        self,
        conf_paths: Union[str, Iterable[str]],
        globals_pattern: Optional[str] = None,
    ):
        super().__init__(conf_paths)
        self.globals_pattern = globals_pattern
        if globals_pattern is not None:
            self._register_global_resolvers()

    def _register_global_resolvers(self):
        processed_files = set()
        global_dict = OmegaConf.create()

        for conf_path in self.conf_paths:
            conf_path = Path(conf_path)
            if not conf_path.is_dir():
                raise ValueError(
                    f"Given configuration path either does not exist "
                    f"or is not a valid directory: {conf_path}"
                )

            filepaths = self._lookup_config_filepaths(
                conf_path, as_list(self.globals_pattern), processed_files
            )

            for global_conf in filepaths:
                global_dict = OmegaConf.merge(global_dict, OmegaConf.load(global_conf))

            processed_files |= set(filepaths)

        OmegaConf.register_new_resolver(
            "globals",
            lambda key: reduce(DictConfig.get, key.split("."), global_dict),
            replace=True,
        )

    @staticmethod
    def _load_config_file(config_file: Path) -> Dict[str, Any]:
        """Load an individual config file using `OmegaConf` as a backend.

        Args:
            config_file (Path): Path to a config file to process.

        Raises:
            BadConfigException: If configuration is poorly formatted and
                cannot be loaded.

        Returns:
            omegaconf.DictConfig: Parsed configuration as a DictConfig.
        """
        try:
            conf = OmegaConf.load(config_file)
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

        Args:
            config_filepaths (list of Path): Configuration files sorted in
                the order of precedence.

        Raises:
            ValueError: If two or more configuration files contain the same
                key(s).
            BadConfigException: If configuration is poorly formatted and
                cannot be loaded.

        Returns:
            omegaconf.DictConfig: Resulting configuration dictionary as a DictConfig.
        """
        aggregate_config = OmegaConf.create()
        seen_file_to_keys = {}

        for config_filepath in config_filepaths:
            single_config = self._load_config_file(config_filepath)
            _check_duplicate_keys(seen_file_to_keys, config_filepath, single_config)
            seen_file_to_keys[config_filepath] = single_config.keys()
            aggregate_config.update(single_config)

        return aggregate_config

    def get(self, *patterns: str) -> Dict[str, Any]:
        """Recursively scan for configuration files, load and merge them, and
        return them in the form of a config dictionary.

        Args:
            *patterns: Glob patterns to match. Files with names matching any of
                the specified patterns will be processed.

        Raises:
            ValueError: If two or more configuration files inside the same config
                path (or its subdirectories) contain the same top-level key.
            MissingConfigException: If no configuration files exist within a specified
                config path.
            BadConfigException: If configuration is poorly formatted and cannot be loaded.

        Returns:
            Dict[str, Any]:  A dict with the combined configuration from all configuration files.

        Note:
            Any keys that start with `_` will be ignored.
        """
        if not patterns:
            raise ValueError(
                "`patterns` must contain at least one glob pattern to match filenames against."
            )

        config = OmegaConf.create()
        processed_files = set()

        for conf_path in self.conf_paths:
            if not Path(conf_path).is_dir():
                raise ValueError(
                    f"Given configuration path either does not exist "
                    f"or is not a valid directory: {conf_path}"
                )

            config_filepaths = self._lookup_config_filepaths(
                Path(conf_path), patterns, processed_files
            )
            new_conf = self._load_configs(config_filepaths)

            common_keys = config.keys() & new_conf.keys()
            if common_keys:
                sorted_keys = ", ".join(sorted(common_keys))
                msg = (
                    "Config from path `%s` will override the following "
                    "existing top-level config keys: %s"
                )
                self.logger.info(msg, conf_path, sorted_keys)

            config.update(new_conf)
            processed_files |= set(config_filepaths)

        if not processed_files:
            raise MissingConfigException(
                f"No files found in {self.conf_paths} matching the glob "
                f"pattern(s): {list(patterns)}"
            )
        return OmegaConf.to_container(config, resolve=True)
