# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, Union
from warnings import warn

import inject
from kedro.config import MissingConfigException
from kedro.framework.context import KedroContext, KedroContextError
from kedro.framework.context.context import (
    _convert_paths_to_absolute_posix,
    _validate_layers_for_transcoding,
)
from kedro.framework.hooks import get_hook_manager
from kedro.io import DataCatalog
from kedro.versioning import Journal
from mergedeep import merge

from .validation import parse_model_config


class CustomContext(KedroContext):
    """``CustomContext`` is the base class which holds the configuration and
    Kedro's main functionality.
    """

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        env: str = None,
        extra_params: Dict[str, Any] = None,
    ):
        """Create a context object

        Args:
            package_name: Package name for the Kedro project the context is created for.
            project_path: Project path to define the context for.
            env: Optional argument for configuration default environment to be used
                for running the pipelines. If not specified, it defaults to "local".
            extra_params: Optional dictionary containing extra project parameters.
                If specified, will update (and therefore take precedence over)
                the parameters retrieved from the project configuration.

        Raises:
            KedroContextError: If there is a mismatch between Kedro project version
                and package version.

        """
        super().__init__(package_name, project_path, env=env, extra_params=extra_params)

    @property
    def params(self) -> Dict[str, Any]:
        """Read-only property referring to Kedro's parameters for this context.

        Returns:
            Parameters defined in configuration file with the addition of any
                extra parameters passed at initialization.
        """
        try:
            selected_params = inject.instance("selected_params")
            if selected_params is None:
                conf_params = {}
            else:
                conf_params = self.config_loader.get(
                    f"parameters/{selected_params}.*",
                    f"parameters/{selected_params}*/**",
                    f"**/parameters/{selected_params}.*",
                )
        except MissingConfigException as exc:
            warn(f"Parameters not found in your Kedro project config.\n{str(exc)}")
            conf_params = {}

        conf_params = merge({}, conf_params, self._extra_params or {})
        return conf_params

    def _get_catalog(
        self,
        save_version: str = None,
        journal: Journal = None,
        load_versions: Dict[str, str] = None,
    ) -> DataCatalog:
        """A hook for changing the creation of a DataCatalog instance.

        Returns:
            DataCatalog

        Raises:
            KedroContextError: Incorrect ``DataCatalog`` registered for the project.
        """
        selected_catalog = inject.instance("selected_catalog")
        selected_model = inject.instance("selected_model")

        if selected_catalog is None:
            raise KedroContextError("No catalog configuration has been selected.")

        conf_catalog = self.config_loader.get(
            f"catalog*/{selected_catalog}.*",
            f"catalog*/{selected_catalog}*/**",
            f"**/catalog*/{selected_catalog}.*",
        )

        rebind_names = {}
        if "rebind_names" in conf_catalog:
            rebind_names = conf_catalog.pop("rebind_names")

        location = {}
        if "location" in conf_catalog:
            location = conf_catalog.pop("location")

        # turn relative paths in conf_catalog into absolute paths
        # before initializing the catalog
        conf_catalog = _convert_paths_to_absolute_posix(
            project_path=self.project_path, conf_dictionary=conf_catalog["sources"]
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

        feed_dict = self._get_feed_dict()
        catalog.add_feed_dict(feed_dict)
        catalog.add_feed_dict(dict(rebind_names=rebind_names))
        catalog.add_feed_dict(dict(location=location))

        if selected_model is not None:
            conf_model = self.config_loader.get(
                f"models/{selected_model}.*",
                f"models/{selected_model}*/**",
                f"**/models/{selected_model}.*",
            )
            model_structure = parse_model_config(conf_model)
            catalog.add_feed_dict(dict(model_structure=model_structure))

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
