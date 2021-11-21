# -*- coding: utf-8 -*-

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from eensight.pipelines import baseline, compare, daytype, predict, preprocess, validate


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    preprocess_pipeline = preprocess.create_pipeline()
    daytype_pipeline = daytype.create_pipeline()
    baseline_pipeline = baseline.create_pipeline()
    validate_pipeline = validate.create_pipeline()
    predict_pipeline = predict.create_pipeline()
    compare_pipeline = compare.create_pipeline()

    return {
        "preprocess": preprocess_pipeline,
        "daytype": daytype_pipeline,
        "baseline": baseline_pipeline,
        "validate": validate_pipeline,
        "predict": predict_pipeline,
        "compare": compare_pipeline,
        "__default__": preprocess_pipeline
        + daytype_pipeline
        + baseline_pipeline
        + validate_pipeline
        + predict_pipeline
        + compare_pipeline,
    }
