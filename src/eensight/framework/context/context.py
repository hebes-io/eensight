# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/framework/context/context.py

import os
from pathlib import Path
from typing import Any, Dict, Union
from warnings import warn

import feature_encoders.settings
from kedro.config import ConfigLoader, MissingConfigException
from kedro.framework.context import KedroContext, KedroContextError
from kedro.framework.context.context import (
    _convert_paths_to_absolute_posix,
    _validate_layers_for_transcoding,
)
from kedro.framework.hooks import get_hook_manager
from kedro.io import DataCatalog
from kedro.versioning import Journal
from omegaconf import OmegaConf

import eensight.settings


class CustomContext(KedroContext):
    """Create a context object.

    ``KedroContext`` is the base class which holds the configuration and
    Kedro's main functionality. ``CustomContext`` extends ``KedroContext``
    to allow additional configuration to pass from the ``KedroSession``.

    Args:
        package_name (str): Package name for the Kedro project the context
            is created for.
        project_path (Union[Path, str]): Project path to define the context for.
        env (str, optional): Optional argument for additional configuration
            environment to be used for running the pipelines. If not specified,
            it defaults to "local". Defaults to None.
        extra_params (Dict[str, Any], optional): Optional dictionary containing
            extra project parameters. If specified, will update (and therefore
            take precedence over) the parameters retrieved from the project
            configuration.
    """

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        env: str = None,
        extra_params: Dict[str, Any] = None,
    ):
        super().__init__(package_name, project_path, env=env, extra_params=extra_params)

    def _get_config_loader(self) -> ConfigLoader:
        """A hook for changing the creation of a ConfigLoader instance.

        Returns:
            ConfigLoader: Instance of `ConfigLoader`.

         Raises:
            KedroContextError: If an incorrect ``ConfigLoader`` is registered.
        """
        feature_path = feature_encoders.settings.CONF_PATH
        resources_path = str(eensight.settings.RESOURCES_PATH.resolve())

        base_path = os.path.join(eensight.settings.CONF_ROOT, "base")
        if not os.path.isabs(base_path):
            base_path = os.path.join(resources_path, base_path)

        local_path = os.path.join(eensight.settings.CONF_ROOT, self.env)
        if not os.path.isabs(local_path):
            local_path = os.path.join(resources_path, local_path)

        conf_paths = [feature_path, base_path, local_path]

        hook_manager = get_hook_manager()
        config_loader = (
            hook_manager.hook.register_config_loader(  # pylint: disable=no-member
                conf_paths=conf_paths,
                env=self.env,
                extra_params=self._extra_params,
            )
        )
        if not isinstance(config_loader, ConfigLoader):
            raise KedroContextError(
                f"Expected an instance of `ConfigLoader`, "
                f"got `{type(config_loader).__name__}` instead."
            )
        return config_loader

    @property
    def params(self) -> Dict[str, Any]:
        """Read-only property referring to Kedro's parameters for this context.

        Returns:
            Dict[str, Any]: Parameters defined in configuration file(s) with the
                addition of any extra parameters passed at initialization.
        """
        try:
            # '**/parameters*' reads modular pipeline configs
            params = self.config_loader.get(
                "parameters*", "parameters*/**", "**/parameters*"
            )
        except MissingConfigException as exc:
            warn(f"Parameters not found in your Kedro project config.\n{str(exc)}")
            params = {}

        params = OmegaConf.create(params)
        extra_params = OmegaConf.create(self._extra_params or {})
        params = OmegaConf.merge(params, extra_params)
        return OmegaConf.to_container(params)

    def _get_catalog(
        self,
        save_version: str = None,
        journal: Journal = None,
        load_versions: Dict[str, str] = None,
    ) -> DataCatalog:
        """A hook for changing the creation of a DataCatalog instance.

        Raises:
            KedroContextError: If incorrect ``DataCatalog`` is registered for the project.

        Returns:
            DataCatalog: A populated DataCatalog instance.
        """
        feed_dict = self._get_feed_dict()

        selected_catalog = feed_dict["parameters"].pop(
            "catalog", eensight.settings.DEFAULT_CATALOG
        )

        catalog_is_partial = feed_dict["parameters"].pop(
            "partial_catalog",
            False or (selected_catalog == eensight.settings.DEFAULT_CATALOG),
        )

        catalog_search = [
            f"catalogs/{selected_catalog}.*",
            f"catalogs/{selected_catalog}/**",
            f"catalogs/**/{selected_catalog}.*",
        ]

        if catalog_is_partial:
            catalog_search.append("templates/_base.*")

        conf_catalog = self.config_loader.get(*catalog_search)

        # remove site_name and versioned
        conf_catalog.pop("site_name", None)
        conf_catalog.pop("versioned", None)

        # capture rebind_names and location
        rebind_names = conf_catalog.pop("rebind_names", {})
        location = conf_catalog.pop("location", {})

        # turn relative paths in conf_catalog into absolute paths
        # before initializing the catalog
        data_root = eensight.settings.DATA_ROOT
        if not os.path.isabs(data_root):
            resources_path = str(eensight.settings.RESOURCES_PATH.resolve())
            data_root = os.path.join(resources_path, data_root)

        conf_catalog = _convert_paths_to_absolute_posix(
            project_path=Path(data_root), conf_dictionary=conf_catalog
        )
        conf_creds = self._get_config_credentials()

        hook_manager = get_hook_manager()
        catalog = hook_manager.hook.register_catalog(  # pylint: disable=no-member
            catalog=conf_catalog,
            credentials=conf_creds,
            load_versions=load_versions,
            save_version=save_version,
            journal=journal,
        )

        if not isinstance(catalog, DataCatalog):
            raise KedroContextError(
                f"Expected an instance of `DataCatalog`, "
                f"got `{type(catalog).__name__}` instead."
            )

        # add model, feature and parameters to the catalog
        selected_base_model = feed_dict["parameters"].pop(
            "base_model", eensight.settings.DEFAULT_BASE_MODEL
        )

        catalog.add_feed_dict(feed_dict)
        catalog.add_feed_dict(dict(rebind_names=rebind_names))
        catalog.add_feed_dict(dict(location=location))

        conf_model = self.config_loader.get(
            f"base_models/{selected_base_model}.*",
            f"base_models/{selected_base_model}/**",
            f"**/base_models/{selected_base_model}.*",
        )
        catalog.add_feed_dict(dict(model_config=conf_model))

        conf_features = self.config_loader.get(
            "features*", "features/**", "**/features*"
        )
        catalog.add_feed_dict(dict(feature_map=conf_features))

        if catalog.layers:
            _validate_layers_for_transcoding(catalog)

        hook_manager = get_hook_manager()
        hook_manager.hook.after_catalog_created(  # pylint: disable=no-member
            catalog=catalog,
            conf_catalog=conf_catalog,
            conf_creds=conf_creds,
            feed_dict=feed_dict,
            save_version=save_version,
            load_versions=load_versions,
            run_id=self.run_id or save_version,
        )
        return catalog
