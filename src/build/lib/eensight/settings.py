# -*- coding: utf-8 -*-

import os
from pathlib import Path

from kedro.framework.session.store import ShelveStore

from eensight.framework.context import CustomContext
from eensight.hooks import ProjectHooks

# Instantiate and list your project hooks here
HOOKS = (ProjectHooks(),)

# Define where to store data from a KedroSession
SESSION_STORE_CLASS = ShelveStore

# Define keyword arguments to be passed to `SESSION_STORE_CLASS` constructor
SESSION_STORE_ARGS = {"path": "./sessions"}

# Define custom context class. Defaults to `KedroContext`
CONTEXT_CLASS = CustomContext

# Define the configuration folder. Defaults to `conf`
CONF_ROOT = "conf"

# Define the data folder. Defaults to `data`
DATA_ROOT = "data"

# Define the default site name
DEFAULT_CATALOG = "demo"

# Define the default base model
DEFAULT_BASE_MODEL = "towt"

# Define the default run configuration file
DEFAULT_RUN_CONFIG = "default"

SOURCE_PATH = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent
PROJECT_PATH = Path(SOURCE_PATH).resolve().parent
