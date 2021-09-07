# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from pathlib import Path

import hydra
import inject
from kedro.framework.session import KedroSession
from kedro.utils import load_obj
from omegaconf import DictConfig, OmegaConf

from eensight.utils import as_list

from .settings import DEFAULT_CATALOG, DEFAULT_MODEL, DEFAULT_PARAMETERS, PROJECT_PATH

warnings.filterwarnings("ignore", category=DeprecationWarning)


def dot_string_to_dict(in_dict):
    tree = {}
    for key, value in in_dict.items():
        t = tree
        parts = key.split(".")
        for part in parts[:-1]:
            t = t.setdefault(part, {})
        t[parts[-1]] = value
    return tree


@hydra.main(config_path="hydra", config_name="run_config")
def run(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)

    runner = cfg.get("runner") or "SequentialRunner"
    runner_class = load_obj(runner, "kedro.runner")

    parameters = cfg.get("parameters") or DEFAULT_PARAMETERS
    catalog = cfg.get("catalog") or DEFAULT_CATALOG
    model = cfg.get("model") or DEFAULT_MODEL

    def bind_values(binder):
        binder.bind("selected_params", parameters)
        binder.bind("selected_catalog", catalog)
        binder.bind("selected_model", model)

    inject.configure(bind_values)

    extra_params = {}
    params = cfg.get("params")
    if params is not None:
        for key, value in params.items():
            extra_params.update(dot_string_to_dict({key: value}))

    env = cfg.get("env") or "local"
    is_async = bool(cfg.get("async"))

    package_name = str(Path(__file__).resolve().parent.name)

    with KedroSession.create(
        package_name,
        project_path=PROJECT_PATH,
        env=env,
        extra_params=extra_params,
        save_on_close=False,
    ) as session:
        session.run(
            runner=runner_class(is_async=is_async),
            tags=as_list(cfg.get("tags")),
            node_names=as_list(cfg.get("nodes")),
            from_nodes=as_list(cfg.get("from_nodes")),
            to_nodes=as_list(cfg.get("to_nodes")),
            from_inputs=as_list(cfg.get("from_inputs")),
            to_outputs=as_list(cfg.get("to_outputs")),
            load_versions=cfg.get("load_versions"),
            pipeline_name=cfg.get("pipeline"),
        )
