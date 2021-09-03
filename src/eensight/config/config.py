# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro/blob/0.17.4/kedro/config/config.py

from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from kedro.config import BadConfigException, ConfigLoader
from kedro.config.config import _check_duplicate_keys
from omegaconf import DictConfig, OmegaConf

from eensight.utils import as_list


class OmegaConfigLoader(ConfigLoader):
    """Recursively scan the directories specified in ``conf_paths`` for
    configuration files with a ``yaml`` or ``yml`` extension, load them,
    and return them in the form of a config dictionary.

    When the same top-level key appears in any 2 config files located in
    the same ``conf_path`` (sub)directory, a ``ValueError`` is raised. This
    behaviour can be overridden by providing a list of values to the
    parameter ``merge_keys``. For these keys, the contents are merged, but if
    a sub-key is duplicate, a ``ValueError`` will be raised.

    When the same key appears in any 2 config files located in different
    ``conf_path`` directories, the last processed config path takes
    precedence and overrides this key.
    """

    def __init__(
        self,
        conf_paths: Union[str, Iterable[str]],
        globals_pattern: Optional[Union[str, Iterable[str]]] = None,
        merge_keys: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Instantiate a ConfigLoader.

        Args:
            conf_paths: str or list of str
                Non-empty path or list of paths to configuration directories.
            globals_pattern: str or list of str, default=None
                Optional argument specifying one or more glob patterns for the
                configuration files containing global values.
            merge_keys: str or list of str, default=None
                Top-level keys for which contents from different config files
                located in the same ``conf_path`` (sub)directory should be merged.

        Raises:
            ValueError: If ``conf_paths`` is empty.
        """
        super().__init__(conf_paths)
        self.globals_pattern = as_list(globals_pattern)
        self.merge_keys = as_list(merge_keys)

        if globals_pattern is not None:
            self._register_global_resolvers()

    @staticmethod
    def _load_config_file(config_file: Path) -> Dict[str, Any]:
        """Load an individual config file using `OmegaConf` as a backend.

        Args:
            config_file: Path to a config file to process.

        Raises:
            BadConfigException: If configuration is poorly formatted and
                cannot be loaded.

        Returns:
            Parsed configuration.
        """
        try:
            conf = OmegaConf.load(config_file)
        except AttributeError as exc:
            raise BadConfigException(
                f"Couldn't load config file: {config_file}"
            ) from exc
        else:
            return conf

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
                conf_path, self.globals_pattern, processed_files
            )

            for global_conf in filepaths:
                global_dict = OmegaConf.merge(global_dict, OmegaConf.load(global_conf))

            processed_files |= set(filepaths)

        OmegaConf.register_new_resolver(
            "globals",
            lambda key: reduce(DictConfig.get, key.split("."), global_dict),
            replace=True,
        )

    def _load_configs(self, config_filepaths: List[Path]) -> Dict[str, Any]:
        """Recursively load all configuration files, which satisfy
        a given list of glob patterns from a specific path.

        Args:
            config_filepaths: Configuration files sorted in the order of precedence.

        Raises:
            ValueError: If 2 or more configuration files contain the same key(s).
            BadConfigException: If configuration is poorly formatted and
                cannot be loaded.

        Returns:
            Resulting configuration dictionary.
        """

        aggregate_config = {}
        seen_file_to_keys = defaultdict(list)

        for config_filepath in config_filepaths:
            single_config = self._load_config_file(config_filepath)
            single_config = OmegaConf.to_container(single_config, resolve=True)

            for key, contents in single_config.items():
                if (key in self.merge_keys) and isinstance(contents, dict):
                    _check_duplicate_keys(seen_file_to_keys, config_filepath, contents)
                    seen_file_to_keys[config_filepath].extend(contents.keys())
                    if key in aggregate_config:
                        aggregate_config[key].update(contents)
                    else:
                        aggregate_config[key] = contents
                elif key in self.merge_keys:
                    raise ValueError("Keys in `merge_keys` cannot be flat.")
                else:
                    _check_duplicate_keys(
                        seen_file_to_keys, config_filepath, {key: None}
                    )
                    seen_file_to_keys[config_filepath].append(key)
                    aggregate_config.update({key: contents})

        return aggregate_config
