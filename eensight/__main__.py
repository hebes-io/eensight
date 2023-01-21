# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click
from kedro.framework.project import configure_project

from .framework.cli.run import run

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS, name="eensight")
def cli():
    """Command line tool for running the eetracker package."""
    pass


cli.add_command(run)


def main():
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    cli()
