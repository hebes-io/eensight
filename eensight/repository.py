from dagster import repository

from eensight.pipelines import preprocessing_pipeline


@repository
def eensight():
    """
    The repository definition for this eensight Dagster repository.

    For hints on building your Dagster repository, see our documentation overview on Repositories:
    https://docs.dagster.io/overview/repositories-workspaces/repositories
    """
    pipelines = [preprocessing_pipeline]
    schedules = []
    sensors = []

    return pipelines + schedules + sensors
