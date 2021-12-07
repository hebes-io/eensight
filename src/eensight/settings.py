# -*- coding: utf-8 -*-

import importlib.resources

from environs import Env
from kedro.framework.session.store import ShelveStore

import eensight
from eensight.framework.context import CustomContext
from eensight.hooks import ProjectHooks

env = Env()
env.read_env()

# Instantiate and list your project hooks here
HOOKS = (ProjectHooks(),)

# Define where to store data from a KedroSession
SESSION_STORE_CLASS = ShelveStore

# Define keyword arguments to be passed to `SESSION_STORE_CLASS` constructor
SESSION_STORE_ARGS = {"path": "./sessions"}

# Define custom context class. Defaults to `KedroContext`
CONTEXT_CLASS = CustomContext

with env.prefixed("EENSIGHT_"):
    # Define the configuration folder. Defaults to `conf`
    CONF_ROOT = env.str("CONF", "conf")
    # Define the data folder. Defaults to `data`
    DATA_ROOT = env.str("DATA", "data")
    # Define the default site name
    DEFAULT_CATALOG = env.str("CATALOG", "demo")
    # Define the default base model
    DEFAULT_BASE_MODEL = env.str("BASE_MODEL", "towt")
    # Define the default run configuration file
    DEFAULT_RUN_CONFIG = env.str("RUN_CONFIG", "default")
    # Define the resources path
    RESOURCES_PATH = env.path("RESOURCES_PATH")

with importlib.resources.path(eensight, "__main__.py") as main_path:
    SOURCE_PATH = main_path.resolve().parents[1]
    PROJECT_PATH = SOURCE_PATH.parent
