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

from .settings import DEFAULT_CATALOG, DEFAULT_MODEL, DEFAULT_PARAMETERS, PROJECT_PATH

warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_string(value):
    """Split string by comma."""
    return [item.strip() for item in value.split(",") if item.strip()]


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

    catalog = cfg.get("catalog") or DEFAULT_CATALOG
    model = cfg.get("model") or DEFAULT_MODEL
    parameters = cfg.get("parameters") or DEFAULT_PARAMETERS
    pipeline_name = cfg.get("pipeline")
    runner = cfg.get("runner") or "SequentialRunner"
    runner_class = load_obj(runner, "kedro.runner")
    is_async = bool(cfg.get("async"))
    env = cfg.get("env") or "local"

    from_inputs = (
        None
        if cfg.get("from_inputs") is None
        else cfg.get("from_inputs")
        if isinstance(cfg.get("from_inputs"), list)
        else split_string(cfg.get("from_inputs"))
    )
    to_outputs = (
        None
        if cfg.get("to_outputs") is None
        else cfg.get("to_outputs")
        if isinstance(cfg.get("to_outputs"), list)
        else split_string(cfg.get("to_outputs"))
    )
    from_nodes = (
        None
        if cfg.get("from_nodes") is None
        else cfg.get("from_nodes")
        if isinstance(cfg.get("from_nodes"), list)
        else split_string(cfg.get("from_nodes"))
    )
    to_nodes = (
        None
        if cfg.get("to_nodes") is None
        else cfg.get("to_nodes")
        if isinstance(cfg.get("to_nodes"), list)
        else split_string(cfg.get("to_nodes"))
    )
    node_names = (
        None
        if cfg.get("nodes") is None
        else cfg.get("nodes")
        if isinstance(cfg.get("nodes"), list)
        else split_string(cfg.get("nodes"))
    )
    tags = (
        None
        if cfg.get("tags") is None
        else cfg.get("tags")
        if isinstance(cfg.get("tags"), list)
        else split_string(cfg.get("tags"))
    )

    extra_params = {}
    params = cfg.get("params")
    if params is not None:
        for key, value in params.items():
            extra_params.update(dot_string_to_dict({key: value}))

    package_name = str(Path(__file__).resolve().parent.name)

    def bind_values(binder):
        binder.bind("selected_params", parameters)
        binder.bind("selected_catalog", catalog)
        binder.bind("selected_model", model)

    inject.configure(bind_values)

    with KedroSession.create(
        package_name,
        project_path=PROJECT_PATH,
        env=env,
        extra_params=extra_params,
        save_on_close=False,
    ) as session:
        session.run(
            runner=runner_class(is_async=is_async),
            tags=tags,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=cfg.get("load_versions"),
            pipeline_name=pipeline_name,
        )
