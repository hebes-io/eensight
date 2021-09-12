# -*- coding: utf-8 -*-

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from eensight.pipelines import day_typing as dt
from eensight.pipelines import model_selection as ms
from eensight.pipelines import preprocessing as pre


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    preprocess_pipeline = pre.create_pipeline()
    daytype_pipeline = dt.create_pipeline()
    baseline_pipeline = ms.create_pipeline()

    return {
        "preprocess": preprocess_pipeline,
        "daytype": daytype_pipeline,
        "baseline": baseline_pipeline,
        "__default__": preprocess_pipeline + daytype_pipeline + baseline_pipeline,
    }
