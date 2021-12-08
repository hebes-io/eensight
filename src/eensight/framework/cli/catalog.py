# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import re
from collections import defaultdict
from functools import reduce

import click
import rich
import yaml
from environs import Env
from kedro.framework.cli.utils import KedroCliError, env_option, split_string
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from omegaconf import DictConfig, OmegaConf
from rich.table import Table

from eensight.settings import CONF_ROOT, DEFAULT_CATALOG, PROJECT_PATH

NAME_HELP = """The name of the catalog to load."""
PARTIAL_ARG_HELP = """Flag to indicate whether the catalog includes information
only about the raw input data."""
PIPELINE_HELP = """One or more names (separated by comma) for the pipeline(s) to
show catalog artifacts for. If not set, artifacts for all pipelines will be shown."""


def _map_type_to_datasets(datasets, datasets_meta):
    """Build dictionary with a dataset type as a key and list of
    datasets of the specific type as a value.
    """
    mapping = defaultdict(list)
    for dataset in datasets:
        is_param = dataset.startswith("params:") or dataset == "parameters"
        if not is_param:
            ds_type = datasets_meta[dataset].__class__.__name__
            if dataset not in mapping[ds_type]:
                mapping[ds_type].append(dataset)
    return mapping


def _echo_text(txt, color="bright_yellow"):
    return click.style(txt, fg=color)


def _collect_location():
    country = None
    province = None
    state = None
    country = click.prompt(
        _echo_text(
            "Enter the name of the country where the site is located (press Enter to skip)"
        ),
        show_default=False,
        default="",
    )
    country = country or None
    if country:
        province = click.prompt(
            _echo_text(
                "Enter the name of the province of the site's location (press Enter to skip)"
            ),
            show_default=False,
            default="",
            type=str,
        )
        province = province or None
        state = click.prompt(
            _echo_text(
                "Enter the name of the state of the site's location (press Enter to skip)"
            ),
            show_default=False,
            default="",
            type=str,
        )
        state = state or None
    return country, province, state


def _collect_rebind_names():
    timestamp = click.prompt(
        _echo_text(
            "Enter the name of the feature that includes date & time information (press "
            "Enter to keep default)"
        ),
        show_default=True,
        default="timestamp",
        type=str,
    )
    consumption = click.prompt(
        _echo_text(
            "Enter the name of the feature for energy consumption (press Enter to keep default)"
        ),
        show_default=True,
        default="consumption",
        type=str,
    )
    temperature = click.prompt(
        _echo_text(
            "Enter the name of the feature for outdoor temperature (press Enter to keep default)"
        ),
        show_default=True,
        default="temperature",
        type=str,
    )
    holiday = None
    if click.confirm(_echo_text("Is there a feature with holiday information?")):
        holiday = click.prompt(
            _echo_text(
                "Enter the name of the feature for holidays (press Enter to keep default)"
            ),
            show_default=True,
            default="holiday",
            type=str,
        )
    return timestamp, consumption, temperature, holiday


@click.group(name="eensight")
def catalog_cli():
    pass


@catalog_cli.group()
def catalog():
    """Commands for working with data catalogs."""


@catalog.command("create")
def create_catalog():
    "Create a new catalog and add it in `conf/base/catalogs`."
    site_name = click.prompt(
        _echo_text(
            "Enter a name for the building site (all whitespace will be "
            "replaced by underscores)"
        ),
        type=str,
    )
    site_name = re.sub(r"\s+", "_", site_name)
    versioned = click.confirm(_echo_text("Should pipeline artifacts be versioned?"))
    country, province, state = _collect_location()
    timestamp, consumption, temperature, holiday = _collect_rebind_names()
    single_file = click.confirm(_echo_text("Is all the data in the same file?"))
    if single_file:
        click.secho("It is assumed that the file is a csv file", fg="bright_white")
        root_input_name = click.prompt(
            _echo_text("Enter the name of the csv file (without the .csv extension)"),
            type=str,
        )
    else:
        click.secho(
            "It is assumed that data is partitioned in many csv files all located "
            "in the same directory",
            fg="bright_white",
        )

    has_post_data = click.confirm(
        _echo_text("Is there a post-intervention version of the data too?")
    )

    conf_root = CONF_ROOT
    resources_path = str(Env().path("EENSIGHT_RESOURCES_PATH").resolve())

    base_path = os.path.join(conf_root, "base")
    if not os.path.isabs(base_path):
        base_path = os.path.join(resources_path, base_path)

    base_cfg = OmegaConf.load(os.path.join(base_path, "templates", "_base.yaml"))

    if single_file:
        root_input_cfg = OmegaConf.load(
            os.path.join(base_path, "templates", "_single.yaml")
        )
        root_input_cfg.root_input_name = root_input_name
    else:
        root_input_cfg = OmegaConf.load(
            os.path.join(base_path, "templates", "_multiple.yaml")
        )

    root_input_cfg.site_name = site_name
    root_input_cfg.versioned = versioned
    root_input_cfg.location.country = country
    root_input_cfg.location.province = province
    root_input_cfg.location.state = state
    root_input_cfg.rebind_names.timestamp = timestamp
    root_input_cfg.rebind_names.consumption = consumption
    root_input_cfg.rebind_names.temperature = temperature
    root_input_cfg.rebind_names.holiday = holiday

    globals_cfg = OmegaConf.load(os.path.join(base_path, "globals.yaml"))

    OmegaConf.register_new_resolver(
        "globals",
        lambda key: reduce(DictConfig.get, key.split("."), globals_cfg),
        replace=True,
    )

    catalog = OmegaConf.merge(root_input_cfg, base_cfg)
    OmegaConf.resolve(catalog)

    keys_to_drop = ["site_name", "versioned", "root_input_name"]
    if not has_post_data:
        for key in catalog:
            if key.startswith("apply.") and (not has_post_data):
                keys_to_drop.append(key)
            else:
                continue

    for key in keys_to_drop:
        catalog.pop(key, None)

    path = pathlib.Path(os.path.join(base_path, "catalogs", site_name))
    path.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(path, "catalog.yaml")

    override = True
    if os.path.isfile(filepath):
        override = click.confirm(
            _echo_text("Existing catalog found. Do you want to override it?")
        )

    if not override:
        click.secho("No catalog created", fg="bright_red")
    else:
        with open(filepath, "w") as f:
            f.write(OmegaConf.to_yaml(catalog))
        click.secho(f"Saved `catalog.yaml` at {path}", fg="bright_white")


# adapted from https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/framework/cli/catalog.py
@catalog.command("describe")
@click.argument("name", default=DEFAULT_CATALOG)
@click.option(
    "--partial-catalog", "-pc", is_flag=True, multiple=False, help=PARTIAL_ARG_HELP
)
@click.option(
    "--pipeline",
    "-ppl",
    type=str,
    default="",
    help=PIPELINE_HELP,
    callback=split_string,
)
@env_option
def list_catalog(name, partial_catalog, pipeline, env):
    "List all artifacts in the selected catalog."
    session = KedroSession.create(
        package_name="eensight",
        project_path=PROJECT_PATH,
        save_on_close=False,
        env=env,
        extra_params={"catalog": name, "partial_catalog": partial_catalog},
    )

    context = session.load_context()
    datasets_meta = context.catalog._data_sets
    catalog_ds = set(context.catalog.list())
    target_pipelines = pipeline or pipelines.keys()

    table = Table(show_header=True, header_style="bold #2070b2", box=rich.box.SIMPLE)
    table.add_column("Pipeline name", justify="right")
    table.add_column("Catalog artifacts used by pipeline")

    for pipe in target_pipelines:
        pl_obj = pipelines.get(pipe)
        if pl_obj:
            pipeline_ds = pl_obj.data_sets()
        else:
            existing_pls = ", ".join(sorted(pipelines.keys()))
            raise KedroCliError(
                f"`{pipe}` pipeline not found! Existing pipelines: {existing_pls}"
            )

        unused_ds = catalog_ds - pipeline_ds
        default_ds = pipeline_ds - catalog_ds
        used_ds = catalog_ds - unused_ds

        used_by_type = _map_type_to_datasets(used_ds, datasets_meta)

        if default_ds:
            used_by_type["DefaultDataSet"].extend(default_ds)

        data = {
            "Artifacts by type": {
                key: sorted(val)
                for key, val in used_by_type.items()
                if isinstance(val, list)
            }
        }
        table.add_row(pipe, yaml.dump(data))

    click.secho(rich.print(table))
