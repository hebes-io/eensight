# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import difflib

import click
import rich
import yaml
from kedro.framework.project import pipelines
from rich.table import Table

NAME_HELP = """The name of the pipeline to describe."""

PREPROCESS_HELP = """Validates input data, identifies outliers, evaluates
data adequacy and imputes missing values.\n"""
DAYTYPE_HELP = """Finds consumption profile prototypes and estimates a distance metric
that translates calendar information to consumption profile similarity.\n"""
BASELINE_HELP = """Fits a predictive model on the available training data, evaluates
its performance in-sample, cross-validates the model, and (if test data is available)
evalutes the model on the test data and identifies periods where the test data distribution
has shifted compared to the training data.\n"""


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
@click.option(
    "--name",
    "-n",
    type=str,
    default="__default__",
    help=NAME_HELP,
)
def describe_pipeline(name):
    """Describe a pipeline by providing a pipeline name. Defaults to all pipelines."""
    descriptions = {
        "preprocess": PREPROCESS_HELP,
        "daytype": DAYTYPE_HELP,
        "baseline": BASELINE_HELP,
    }

    table = Table(show_header=True, header_style="bold #2070b2", box=rich.box.SIMPLE)
    table.add_column("Pipeline name", justify="right")
    table.add_column("Pipeline description")

    if name == "__default__":
        for pipe in descriptions:
            table.add_row(pipe, descriptions[pipe])
        click.secho(rich.print(table))
    elif name in descriptions:
        table.add_row(name, descriptions[name])
        click.secho(rich.print(table))
    else:
        error_msg = f"Pipeline '{name}' not recognised"
        matches = difflib.get_close_matches(name, descriptions.keys())
        if matches:
            suggestions = ", ".join(matches)  # type: ignore
            error_msg += f" - did you mean one of these instead: {suggestions}"
        click.secho(error_msg, fg="bright_red")
