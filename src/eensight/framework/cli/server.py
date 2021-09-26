# -*- coding: utf-8 -*-

# Adapted from https://github.com/quantumblacklabs/kedro-viz/blob/v3.16.0/package/kedro_viz/server.py


import webbrowser
from pathlib import Path
from typing import Dict, cast

import uvicorn
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro_viz.api import apps, responses
from kedro_viz.data_access import data_access_manager
from kedro_viz.server import DEFAULT_HOST, DEFAULT_PORT, is_localhost, populate_data


def run_server(
    catalog: str = None,
    partial_catalog: bool = False,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    browser: bool = None,
    load_file: str = None,
    save_file: str = None,
    pipeline_name: str = None,
    env: str = None,
    project_path: str = None,
    autoreload: bool = False,
):
    """Run a uvicorn server with a FastAPI app that either launches API response data from a file
    or from reading data from a real Kedro project.
    Args:
        catalog: the name of the catalog to load
        partial_catalog: whether the catalog includes information only about the raw input data
        host: the host to launch the webserver
        port: the port to launch the webserver
        browser: whether to open the default browser automatically
        load_file: if a valid JSON containing API response data is provided,
            the API of the server is created from the JSON.
        save_file: if provided, the data returned by the API will be saved to a file.
        pipeline_name: the optional name of the pipeline to visualise.
        env: the optional environment of the pipeline to visualise.
            If not provided, it will use Kedro's default, which is "local".
        autoreload: Whether the API app should support autoreload.
        project_path: the optional path of the Kedro project that contains the pipelines
            to visualise. If not supplied, the current working directory will be used.
    """
    if load_file is None:
        path = Path(project_path) if project_path else Path.cwd()
        bootstrap_project(path)

        session = KedroSession.create(
            package_name="eensight",
            project_path=project_path,
            save_on_close=False,
            env=env,
            extra_params={"catalog": catalog, "catalog_is_partial": partial_catalog},
        )
        context = session.load_context()
        catalog = context.catalog
        pipelines = cast(Dict, kedro_pipelines)

        pipelines = (
            pipelines
            if pipeline_name is None
            else {pipeline_name: pipelines[pipeline_name]}
        )
        populate_data(data_access_manager, catalog, pipelines)
        if save_file:
            res = responses.get_default_response()
            Path(save_file).write_text(res.json(indent=4, sort_keys=True))

        app = apps.create_api_app_from_project(path, autoreload)
    else:
        app = apps.create_api_app_from_file(load_file)

    if browser and is_localhost(host):
        webbrowser.open_new(f"http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)
