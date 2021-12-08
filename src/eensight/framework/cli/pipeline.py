# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import difflib
import os

import click
import rich
import yaml
from environs import Env
from kedro.framework.cli.utils import env_option
from kedro.framework.project import pipelines
from rich.table import Table

from eensight.config import OmegaConfigLoader
from eensight.settings import CONF_ROOT, PROJECT_PATH

PREPROCESS_HELP = """Merges (if necessary) and validates input data, identifies potential
data drift, identifies potential outliers, evaluates data adequacy,
and imputes missing values.\n"""
DAYTYPE_HELP = """Finds consumption profile prototypes and estimates a distance metric that
translates calendar information (month of year and day of week) to daily 
or sub-daily consumption profile similarity. Prototypes are a small set
of daily or sub-daily profiles that adequately summarize the available data.\n"""
BASELINE_HELP = """Optimizes a predictive model for either in-sample or out-of-sample
performance, fits the optimized model on the available training data,
and evaluates its performance in-sample.\n"""
VALIDATE_HELP = """Cross-validates the optimized predictive model and builds a
conformal predictor to construct uncertainty intervals.\n"""
PREDICT_HELP = """Uses the optimized predictive model and the conformal model of the `validate`
stage to generate predictions on pre- and post-retrofit data, adding uncertainty
intervals with user-provided confidence levels.\n"""
COMPARE_HELP = """Estimates cumulative savings given the optimized predictive model and
post-retrofit data, while adding uncertainty intervals with user-provided
confidence levels.\n"""


def dotter(x, key="", dots={}):
    if isinstance(x, dict):
        for k in x.keys():
            dotter(x[k], "%s.%s" % (key, k) if key else k)
    else:
        dots[key] = x
    return dots


def as_dot_list(x):
    x = dotter(x)
    return "\n".join([f"{key}: {val}" for key, val in x.items()])


@click.group(name="eensight")
def pipeline_cli():
    pass


@pipeline_cli.group()
def pipeline():
    """Commands for working with pipelines."""


@pipeline.command("list")
def list_pipelines():
    """List all pipelines defined in eensight's registry.py file."""
    click.echo(yaml.dump(sorted(pipelines)))


@pipeline.command("describe")
@click.argument("name")
@env_option
def describe_pipeline(name, env):
    """Describe a pipeline by providing a pipeline name."""
    descriptions = {
        "preprocess": PREPROCESS_HELP,
        "daytype": DAYTYPE_HELP,
        "baseline": BASELINE_HELP,
        "validate": VALIDATE_HELP,
        "predict": PREDICT_HELP,
        "compare": COMPARE_HELP,
    }

    resources_path = str(Env().path("EENSIGHT_RESOURCES_PATH").resolve())

    base_path = os.path.join(CONF_ROOT, "base")
    if not os.path.isabs(base_path):
        base_path = os.path.join(resources_path, base_path)

    local_path = os.path.join(CONF_ROOT, env)
    if not os.path.isabs(local_path):
        local_path = os.path.join(resources_path, local_path)

    config_loader = OmegaConfigLoader([base_path, local_path])

    table = Table(box=rich.box.SIMPLE, show_header=False)
    table.add_column(style="bold #2070b2")
    table.add_column()

    if name in descriptions:
        params = config_loader.get(
            f"{name}*", f"parameters*/{name}*", f"**/parameters*/{name}**"
        )
        table.add_row("Description", descriptions[name])
        table.add_row("Parameters", as_dot_list(params))
        click.secho(rich.print(table))
    else:
        error_msg = f"Pipeline '{name}' not recognised"
        matches = difflib.get_close_matches(name, descriptions.keys())
        if matches:
            suggestions = ", ".join(matches)  # type: ignore
            error_msg += f" - did you mean one of these instead: {suggestions}"
        click.secho(error_msg, fg="bright_red")
