# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import pandas as pd
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.io import DataCatalog

from eensight.config import OmegaConfigLoader
from eensight.settings import CONF_ROOT, PROJECT_PATH


def load_catalog(catalog, model=None, parameters=None, env="local"):
    """A utility function that loads the data catalog in a way that is similar
    to that of the KedroContext.
    """
    conf_paths = [
        os.path.join(PROJECT_PATH, CONF_ROOT, "base"),
        os.path.join(PROJECT_PATH, CONF_ROOT, env),
    ]

    config_loader = OmegaConfigLoader(
        conf_paths,
        globals_pattern=["globals*", "globals*/**", "**/globals*"],
        merge_keys=["rebind_names", "sources"],
    )

    conf_catalog = config_loader.get(
        f"catalog*/{catalog}*", f"catalog*/{catalog}*/**", f"**/catalog*/{catalog}*"
    )

    rebind_names = {}
    if "rebind_names" in conf_catalog:
        rebind_names = conf_catalog.pop("rebind_names")

    location = {}
    if "location" in conf_catalog:
        location = conf_catalog.pop("location")

    conf_catalog = _convert_paths_to_absolute_posix(
        project_path=PROJECT_PATH, conf_dictionary=conf_catalog
    )

    catalog = DataCatalog.from_config(conf_catalog["sources"])

    catalog.add_feed_dict(dict(rebind_names=rebind_names))
    catalog.add_feed_dict(dict(location=location))

    if model is not None:
        model = config_loader.get(
            f"models/{model}*", f"models/{model}*/**", f"**/models/{model}*"
        )
        catalog.add_feed_dict(dict(models=model))

    if parameters is not None:
        parameters = config_loader.get(
            f"parameters/{parameters}*",
            f"parameters/{parameters}*/**",
            f"**/parameters/{parameters}*",
        )
        catalog.add_feed_dict(dict(parameters=parameters))
        params = pd.json_normalize(parameters, sep=".")
        params = params.rename(columns={col: "params:" + col for col in params.columns})
        catalog.add_feed_dict(params.to_dict(orient="records")[0])

    return catalog
