# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import os
from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple

import click
from environs import Env
from kedro.framework.cli.utils import (
    KedroCliError,
    _reformat_load_versions,
    env_option,
    split_string,
)
from kedro.framework.session import KedroSession
from kedro.utils import load_obj
from omegaconf import OmegaConf

from eensight.config import OmegaConfigLoader
from eensight.framework.cli.catalog import catalog
from eensight.framework.cli.pipeline import pipeline
from eensight.framework.cli.starter import resources
from eensight.settings import (
    CONF_ROOT,
    DEFAULT_BASE_MODEL,
    DEFAULT_RUN_CONFIG,
    PROJECT_PATH,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

CATALOG_HELP = """The name of the catalog to use. Catalogs will be searched
in `conf/base/catalogs`."""
PARTIAL_ARG_HELP = """A flag to indicate whether the selected catalog includes
information only about the raw input data."""
BASE_MODEL_ARG_HELP = """The name of the base model configuration to use. If not
set, the `seetings.DEFAULT_BASE_MODEL` will be used."""
PIPELINE_ARG_HELP = """Name of the modular pipeline to run. If not set, all
pipelines will run."""
RUNNER_ARG_HELP = """Specify a runner that you want to run the pipeline with.
Available runners: `SequentialRunner`, `ParallelRunner` and `ThreadRunner`.
The default is `SequentialRunner`. This option cannot be used together with 
--parallel."""
PARALLEL_ARG_HELP = """Flag to run the pipeline using the `ParallelRunner`.
If not specified, use the `SequentialRunner`. This flag cannot be used together
with --runner."""
ASYNC_ARG_HELP = """Flag to load and save node inputs and outputs asynchronously
with threads. If not specified, load and save datasets synchronously."""
FROM_INPUTS_HELP = """A (comma separated) list of dataset names which should be
used as a starting point."""
TO_OUTPUTS_HELP = """A (comma separated) list of dataset names which should be
used as an end point."""
FROM_NODES_HELP = """A (comma separated) list of node names which should be used
as a starting point."""
TO_NODES_HELP = """A (comma separated) list of node names which should be used as
an end point."""
NODE_ARG_HELP = """Run only nodes with specified names. Option can be used 
multiple times."""
TAG_ARG_HELP = """Construct the pipeline using only nodes which have this 
tag attached. Option can be used multiple times, which results in a pipeline 
constructed from nodes having any of those tags."""
LOAD_VERSION_HELP = """Specify a particular dataset version (timestamp) for loading.
Option can be used multiple times, each for a particular dataset. The expected form of
`load_version` is`dataset_name:YYYY-MM-DDThh.mm.ss.sssZ`"""
EXTRA_PARAMS_ARG_HELP = """Specify extra parameters that you want to pass
to the context initializer. It is expected to be a dot-list: "a.aa.aaa=1, a.aa.bbb=2"
https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-dot-list"""
RUN_CONFIG_HELP = """Specify a YAML configuration file to load the run command 
arguments from. If command line arguments are also provided, they will override 
the loaded ones."""


def _get_values_as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    if values:
        return tuple(chain.from_iterable(value.split(",") for value in values))
    else:
        return values


@click.group(context_settings=CONTEXT_SETTINGS, name="eensight")
def cli():
    """Command line tool for running the eensight package."""


@cli.command("run")
@click.option(
    "--catalog",
    "-c",
    type=str,
    default=None,
    help=CATALOG_HELP,
)
@click.option(
    "--partial-catalog", "-pc", is_flag=True, multiple=False, help=PARTIAL_ARG_HELP
)
@click.option("--base-model", "-bm", type=str, default=None, help=BASE_MODEL_ARG_HELP)
@click.option("--pipeline", "-ppl", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--runner",
    "-r",
    type=str,
    default=None,
    help=RUNNER_ARG_HELP,
)
@click.option("--parallel", "-p", is_flag=True, multiple=False, help=PARALLEL_ARG_HELP)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_string
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_string
)
@click.option("--tag", "-t", "tags", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--load-version",
    "-lv",
    "load_versions",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_reformat_load_versions,
)
@click.option(
    "--params",
    "extra_params",
    type=str,
    default="",
    help=EXTRA_PARAMS_ARG_HELP,
    callback=split_string,
)
@click.option(
    "--run-config", "-rc", type=str, default=DEFAULT_RUN_CONFIG, help=RUN_CONFIG_HELP
)
def run(
    catalog,
    partial_catalog,
    base_model,
    pipeline,
    runner,
    parallel,
    is_async,
    env,
    from_inputs,
    to_outputs,
    from_nodes,
    to_nodes,
    tags,
    node_names,
    load_versions,
    extra_params,
    run_config,
):
    """Run the eensight pipelines using the selected catalog data."""
    if env is None:
        env = "local"

    resources_path = str(Env().path("EENSIGHT_RESOURCES_PATH").resolve())
    config_loader = OmegaConfigLoader(
        [os.path.join(resources_path, CONF_ROOT, "base", "run_config")]
    )
    run_config = config_loader.get(
        f"{run_config}.*", f"{run_config}*/**", f"**/{run_config}.*"
    )

    run_args = {
        "base_model": base_model,
        "pipeline": pipeline,
        "runner": runner,
        "from_inputs": from_inputs,
        "to_outputs": to_outputs,
        "from_nodes": from_nodes,
        "to_nodes": to_nodes,
        "tags": tags,
        "node_names": node_names,
        "load_versions": load_versions,
        "extra_params": extra_params,
    }

    run_args = {
        key: run_config.get(key) if not value else value
        for key, value in run_args.items()
    }

    """Run the pipeline."""
    base_model = run_args["base_model"] or DEFAULT_BASE_MODEL

    if parallel and run_args["runner"]:
        raise KedroCliError(
            "Both --parallel and --runner options cannot be used together. "
            "Please use either --parallel or --runner."
        )

    runner = run_args["runner"] or "SequentialRunner"
    if parallel:
        runner = "ParallelRunner"
    runner_class = load_obj(runner, "kedro.runner")

    # extra_params is expected to be a dot-list,
    # https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-dot-list
    if run_args["extra_params"]:
        extra_params = OmegaConf.to_container(
            OmegaConf.from_dotlist(run_args["extra_params"])
        )
    else:
        extra_params = {}

    extra_params["catalog"] = catalog
    extra_params["partial_catalog"] = partial_catalog
    extra_params["base_model"] = base_model

    package_name = str(Path(__file__).resolve().parent.name)
    with KedroSession.create(
        package_name,
        project_path=PROJECT_PATH,
        env=env,
        extra_params=extra_params,
        save_on_close=False,
    ) as session:
        session.run(
            pipeline_name=run_args["pipeline"],
            runner=runner_class(is_async=is_async),
            tags=_get_values_as_tuple(run_args["tags"]),
            node_names=_get_values_as_tuple(run_args["node_names"]),
            from_nodes=run_args["from_nodes"],
            to_nodes=run_args["to_nodes"],
            from_inputs=run_args["from_inputs"],
            to_outputs=run_args["to_outputs"],
            load_versions=run_args["load_versions"],
        )


cli.add_command(catalog)
cli.add_command(pipeline)
cli.add_command(resources)
