# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, Union

from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from pluggy import PluginManager


class Context(KedroContext):
    """Kedro context object.

    Args:
        package_name (str): Package name for the Kedro project the context is
            created for.
        project_path (pathlib.Path or str): Project path to define the context for.
        config_loader (kedro.config.ConfigLoader): The loader for the project's
            configuration files.
        hook_manager: The ``PluginManager`` to activate hooks, supplied by the session.
        env: Optional argument for configuration default environment to be used
            for running the pipeline. If not specified, it defaults to "local".
        extra_params: Optional dictionary containing extra project parameters.
            If specified, will update (and therefore take precedence over)
            the parameters retrieved from the project configuration.
    """

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        config_loader: ConfigLoader,
        hook_manager: PluginManager,
        env: str = None,
        extra_params: Dict[str, Any] = None,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            package_name=package_name,
            project_path=project_path,
            config_loader=config_loader,
            hook_manager=hook_manager,
            env=env,
            extra_params=extra_params,
        )

    def _get_feed_dict(self) -> Dict[str, Any]:
        """Get parameters and return the feed dictionary."""
        params = self.params
        params.pop("globals", None)
        app_data = params.pop("app", None)
        feed_dict = {"parameters": params, "app": app_data}

        def _add_param_to_feed_dict(param_name, param_value):
            """This recursively adds parameter paths to the `feed_dict`,
            whenever `param_value` is a dictionary itself, so that users can
            specify specific nested parameters in their node inputs.
            """
            key = f"params:{param_name}"
            feed_dict[key] = param_value

            if isinstance(param_value, dict):
                for key, val in param_value.items():
                    _add_param_to_feed_dict(f"{param_name}.{key}", val)

        for param_name, param_value in params.items():
            _add_param_to_feed_dict(param_name, param_value)

        return feed_dict
