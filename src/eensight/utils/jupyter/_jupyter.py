# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from dynaconf.validator import ValidationError
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog

from eensight.settings import PROJECT_PATH


def load_catalog(catalog, partial_catalog=False, model=None, env="local"):
    """A utility function that loads the data catalog."""
    path = Path(PROJECT_PATH)
    bootstrap_project(path)

    extra_params = {"catalog": catalog, "catalog_is_partial": partial_catalog}
    if model is not None:
        extra_params["model"] = model

    session = KedroSession.create(
        package_name="eensight",
        project_path=PROJECT_PATH,
        save_on_close=False,
        env=env,
        extra_params=extra_params,
    )
    context = session.load_context()
    catalog = context.catalog

    if not isinstance(catalog, DataCatalog):
        raise ValidationError(
            f"Expected an instance of `DataCatalog`, "
            f"got `{type(catalog).__name__}` instead."
        )

    return catalog
