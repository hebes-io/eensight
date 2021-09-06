# -*- coding: utf-8 -*-

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

#from eensight.pipelines import day_typing as dd
from eensight.pipelines import preprocessing as pre


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    preprocess_pipeline = pre.create_pipeline()
    #daytype_pipeline = dd.create_pipeline()

    return {
        "preprocess": preprocess_pipeline,
        #"daytype": daytype_pipeline,
        #"day_type": daytype_pipeline,
        "__default__": preprocess_pipeline #+ daytype_pipeline,
    }
