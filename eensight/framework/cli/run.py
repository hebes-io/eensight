# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import click
import mlflow
from click_help_colors import HelpColorsGroup
from kedro.framework.cli.utils import env_option
from kedro.framework.session import KedroSession
from kedro.utils import load_obj
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from omegaconf import OmegaConf

from eensight.framework.startup import bootstrap_project
from eensight.settings import PROJECT_PATH

from .help import (
    AUTO_HELP,
    BATCH_HELP,
    CONFIG_FILE_HELP,
    DISABLE_TRACKING_HELP,
    EXP_NAME_HELP,
    FEATURES_FORMAT_HELP,
    FEATURES_LOAD_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    INPUT_URI_HELP,
    LABELS_FORMAT_HELP,
    LABELS_LOAD_HELP,
    LOCAL_FILE_URI_PREFIX,
    NAMESPACE_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    RUN_ID_HELP,
    RUNNER_ARG_HELP,
    SITE_ID_HELP,
    STORE_URI_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    TRACK_URI_HELP,
)

logger = logging.getLogger("eensight")
warnings.filterwarnings("ignore", category=DeprecationWarning)


##########################################################################################
# Utility functions
##########################################################################################


def _uri_to_string(uri):
    if uri and (urlparse(uri).scheme == "file"):
        return uri
    if uri and os.path.exists(uri):
        return Path(uri).absolute().resolve().as_posix()
    return uri


def _tuple_to_dict(ctx, param, values) -> Dict[str, str]:
    """Reformat data structure from tuple to dictionary"""
    if values:
        return dict(map(lambda x: x.split("="), values))
    return dict()


def _split_string(ctx, param, value):
    """Split string by comma."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_kedro_session(env, extra_params, run_args):
    bootstrap_project(PROJECT_PATH)
    with KedroSession.create(
        package_name="eensight",
        project_path=PROJECT_PATH,
        env=env,
        extra_params=extra_params,
        save_on_close=False,
    ) as session:
        session.run(**run_args)


##########################################################################################
# CLI
##########################################################################################


@click.group(
    name="eensight",
    cls=HelpColorsGroup,
    help_headers_color="green",
    help_options_color="yellow",
)
def cli_run():
    pass


@cli_run.command("run", help="Run a eensight pipeline, where PIPELINE is its name.")
@click.argument("pipeline", type=str)
@click.option("--site-id", "-si", type=str, default=None, help=SITE_ID_HELP)
@click.option(
    "--store-uri",
    "-su",
    type=str,
    default=None,
    envvar="EENSIGHT_STORE_URI",
    help=STORE_URI_HELP,
)
@click.option(
    "--namespace",
    "-ns",
    type=click.Choice(["train", "test", "apply"]),
    default=None,
    help=NAMESPACE_HELP,
)
@click.option(
    "--autoencode",
    "-ac",
    is_flag=True,
    show_default=True,
    default=False,
    help=AUTO_HELP,
)
@click.option(
    "--input-uri",
    "-iu",
    type=str,
    default=None,
    help=INPUT_URI_HELP,
)
@click.option("--batch", "-b", type=str, default=None, help=BATCH_HELP)
@click.option(
    "--no-tracking",
    "-nt",
    "disable_tracking",
    is_flag=True,
    help=DISABLE_TRACKING_HELP,
)
@click.option(
    "--experiment",
    "-ex",
    "experiment_name",
    type=str,
    default=None,
    help=EXP_NAME_HELP,
)
@click.option(
    "--tracking-uri",
    "-tu",
    type=str,
    default=None,
    envvar="MLFLOW_TRACKING_URI",
    help=TRACK_URI_HELP,
)
@click.option(
    "--run-id",
    "-ri",
    type=str,
    default=None,
    help=RUN_ID_HELP,
)
@click.option(
    "--features-format",
    "-ff",
    type=click.Choice(["csv", "parquet", "json"]),
    show_default=True,
    default="csv",
    help=FEATURES_FORMAT_HELP,
)
@click.option(
    "--labels-format",
    "-lf",
    type=click.Choice(["csv", "parquet", "json"]),
    show_default=True,
    default="csv",
    help=LABELS_FORMAT_HELP,
)
@click.option(
    "--f-load-arg",
    "-fla",
    "features_load_args",
    type=str,
    multiple=True,
    callback=_tuple_to_dict,
    help=FEATURES_LOAD_HELP,
)
@click.option(
    "--l-load-arg",
    "-lla",
    "labels_load_args",
    type=str,
    multiple=True,
    callback=_tuple_to_dict,
    help=LABELS_LOAD_HELP,
)
@env_option
@click.option(
    "--param",
    "-p",
    "extra_params",
    type=click.UNPROCESSED,
    multiple=True,
    help=PARAMS_ARG_HELP,
)
@click.option(
    "--from-inputs",
    "-fi",
    type=click.UNPROCESSED,
    default="",
    help=FROM_INPUTS_HELP,
    callback=_split_string,
)
@click.option(
    "--to-outputs",
    "-to",
    type=click.UNPROCESSED,
    default="",
    help=TO_OUTPUTS_HELP,
    callback=_split_string,
)
@click.option(
    "--from-nodes",
    "-fn",
    type=click.UNPROCESSED,
    default="",
    help=FROM_NODES_HELP,
    callback=_split_string,
)
@click.option(
    "--to-nodes",
    "-tn",
    type=click.UNPROCESSED,
    default="",
    help=TO_NODES_HELP,
    callback=_split_string,
)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--runner",
    "-r",
    type=str,
    default=None,
    help=RUNNER_ARG_HELP,
)
@click.option(
    "--tag",
    "-t",
    "tags",
    type=str,
    multiple=True,
    callback=_tuple_to_dict,
    help=TAG_ARG_HELP,
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help=CONFIG_FILE_HELP,
)
def run(
    pipeline: str,
    site_id: str,
    store_uri: str,
    namespace: str,
    autoencode: bool,
    input_uri: str,
    batch: str,
    disable_tracking: str,
    experiment_name: str,
    tracking_uri: str,
    run_id: str,
    features_format: str,
    labels_format: str,
    features_load_args: dict,
    labels_load_args: dict,
    env: str,
    extra_params: list,
    from_inputs: list,
    to_outputs: list,
    from_nodes: list,
    to_nodes: list,
    node_names: list,
    runner: str,
    tags: dict,
    config: Path,
):
    """Run the eensight pipelines using the provided input data."""

    if (pipeline == "evaluate") and (namespace == "train"):
        msg = "The `evaluate` pipeline supports only the `test` and `apply` namespaces."
        click.secho(click.style(msg, fg="bright_red"))
        raise click.Abort()

    if (pipeline == "adjust") and (namespace != "apply"):
        msg = "The `adjust` pipeline supports only the `apply` namespace."
        click.secho(click.style(msg, fg="bright_red"))
        raise click.Abort()

    config = OmegaConf.load(config) if config else {}

    env = env or config.get("env") or "local"
    site_id = site_id or config.get("site_id")

    if site_id is None:
        msg = (
            "`site_id` must be provided through either the cli or the `--config` file."
        )
        click.secho(click.style(msg, fg="bright_red"))
        raise click.Abort()

    store_uri = _uri_to_string(store_uri or config.get("store_uri"))
    if store_uri is None:
        msg = "`store_uri` must be provided through either the cli or the `--config` file."
        click.secho(click.style(msg, fg="bright_red"))
        raise click.Abort()

    input_uri = _uri_to_string(input_uri or config.get("input_uri"))
    namespace = namespace or config.get("namespace") or "train"

    disable_tracking = (
        disable_tracking
        or (namespace == "apply")
        or ((namespace == "test") and (run_id is None))
    )

    if not disable_tracking:
        experiment_name = experiment_name or config.get("experiment_name") or "Default"
        tracking_uri = tracking_uri or config.get("tracking_uri")

        if tracking_uri:
            tracking_uri = _uri_to_string(tracking_uri)
        elif os.path.exists(store_uri):
            tracking_uri = f"{LOCAL_FILE_URI_PREFIX}/{store_uri}/{site_id}/mlruns"
        else:
            msg = "`tracking_uri` must be provided if `store_uri` is not a local directory."
            click.secho(click.style(msg, fg="bright_red"))
            raise click.Abort()

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        expt = client.get_experiment_by_name(experiment_name)
        if expt is not None:
            experiment_id = expt.experiment_id
        else:
            logger.info(
                f"Experiment with name {experiment_name} not found. Creating it."
            )
            experiment_id = client.create_experiment(name=experiment_name)

        tags = {
            **config.get("tags", {}),
            **tags,
            "site_id": site_id,
            "namespace": namespace,
            "autoencoding": autoencode,
        }

    features_load_args = features_load_args or config.get("features_load_args", {})
    labels_load_args = labels_load_args or config.get("labels_load_args", {})

    for load_args in [features_load_args, labels_load_args]:
        if ("sep" in load_args) and (load_args["sep"] == ","):
            del load_args["sep"]

    extra_params = [
        "=".join([p.split("=")[0].replace("-", "_"), p.split("=")[1]])
        for p in extra_params
    ]
    extra_params = {
        **config.get("extra_params", {}),
        **OmegaConf.to_container(OmegaConf.from_dotlist(extra_params)),
    }

    for key in extra_params.keys():
        if key in ("globals", "app"):
            msg = (
                f"`{key}` is a reserved word and cannot be used as a key in `--param`."
            )
            click.secho(click.style(msg, fg="bright_red"))
            raise click.Abort()

    extra_params.update(
        {
            "app": {
                "store_uri": store_uri,
                "input_uri": input_uri,
                "site_id": site_id,
                "namespace": namespace,
                "run_id": None,
                "batch": batch,
                "features": {
                    "format": features_format,
                    "load_args": features_load_args,
                },
                "labels": {"format": labels_format, "load_args": labels_load_args},
            }
        }
    )

    runner = load_obj(
        runner or config.get("runner") or "SequentialRunner", "kedro.runner"
    )
    run_args = {
        "from_nodes": from_nodes,
        "to_nodes": to_nodes,
        "from_inputs": from_inputs,
        "to_outputs": to_outputs,
        "node_names": node_names,
        "runner": runner(is_async=True),
        "pipeline_name": pipeline,
        "tags": [
            namespace,
            "_".join([namespace, "autoenc" if autoencode else "default"]),
        ],
    }

    run_args = {
        key: config.get(key) if not value else value for key, value in run_args.items()
    }

    if not disable_tracking:
        with mlflow.start_run(
            experiment_id=experiment_id, run_id=run_id, tags=resolve_tags(tags)
        ) as run:
            run_id = run.info.run_id
            run_name = run.info.run_name
            extra_params["app"]["run_id"] = run_id
            _run_kedro_session(env, extra_params, run_args)
    else:
        run_id = None
        run_name = None
        _run_kedro_session(env, extra_params, run_args)

    click.echo(f"Run ID: {run_id}, Run name (for MLflow UI): {run_name}")
