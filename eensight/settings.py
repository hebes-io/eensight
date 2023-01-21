import importlib.resources

from kedro.io import DataCatalog

import eensight

from .config import ConfigLoader
from .framework.context import Context
from .hooks import ActivityHooks, MLflowHooks

with importlib.resources.path(eensight, "__main__.py") as main_path:
    PROJECT_PATH = main_path.resolve().parent

# Instantiated project hooks.
HOOKS = (
    ActivityHooks(),
    MLflowHooks(),
)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.shelvestore import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Class that manages Kedro's library components.
CONTEXT_CLASS = Context

# Directory that holds configuration.
CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = ConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# CONFIG_LOADER_ARGS = {
#       "config_patterns": {
#           "spark" : ["spark*/"],
#           "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
#       }
# }
CONFIG_LOADER_ARGS = {
    "globals_pattern": "*globals.*",
}

# Class that manages the Data Catalog.
DATA_CATALOG_CLASS = DataCatalog
