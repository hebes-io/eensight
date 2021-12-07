# -*- coding: utf-8 -*-
# Adapted from https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/framework/cli/starters.py


import importlib.resources
import shutil
from pathlib import Path

import click
import yaml
from kedro.framework.cli.starters import _fetch_config_from_user_prompts
from kedro.framework.cli.utils import KedroCliError

import eensight.templates

DIRECTORY_ARG_HELP = """An optional directory inside which the resources repository
should reside. If no value is provided, the current working directory will be used."""


def _clean_pycache(path: Path):
    """Recursively clean all __pycache__ folders from `path`.
    Args:
        path: Existing local directory to clean __pycache__ folders from.
    """
    to_delete = [each.resolve() for each in path.rglob("__pycache__")]

    for each in to_delete:
        shutil.rmtree(each, ignore_errors=True)


@click.group(name="eensight")
def start_cli():
    pass


@start_cli.group()
def resources():
    """Initialize the eensight resources."""


@resources.command("init")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=False, file_okay=False),
    default=None,
    help=DIRECTORY_ARG_HELP,
)
def init(output_dir):
    """Create a repository for the eensight resources."""
    from cookiecutter.generate import generate_context
    from cookiecutter.main import cookiecutter  # for performance reasons

    with importlib.resources.path(
        eensight.templates, "cookiecutter.json"
    ) as cookiecutter_dir:
        cookiecutter_context = generate_context(cookiecutter_dir).get(
            "cookiecutter", {}
        )
        cookiecutter_dir = cookiecutter_dir.parent.resolve()
        prompts_yml = cookiecutter_dir / "prompts.yml"
        try:
            with prompts_yml.open("r") as prompts_file:
                prompts_required = yaml.safe_load(prompts_file)
        except Exception as exc:
            raise KedroCliError(
                "Failed to generate project: could not load prompts.yml. " + str(exc)
            ) from exc

        config = _fetch_config_from_user_prompts(prompts_required, cookiecutter_context)

        cookiecutter_args = {
            "output_dir": output_dir or str(Path.cwd().resolve()),
            "no_input": True,
            "extra_context": config,
        }
        try:
            resources_path = cookiecutter(
                template=str(cookiecutter_dir), **cookiecutter_args
            )
        except Exception as exc:
            raise KedroCliError(
                "Failed to generate resourse repository when running cookiecutter. "
                + str(exc)
            ) from exc

        _clean_pycache(Path(resources_path))
        click.secho(
            f"\nResource repository generated in {resources_path}",
            fg="bright_green",
        )
