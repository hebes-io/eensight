"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import adjust, evaluate, predict, preprocess


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocess_ppl = preprocess.create_pipeline()
    predict_ppl = predict.create_pipeline()
    evaluate_ppl = evaluate.create_pipeline()
    adjust_ppl = adjust.create_pipeline()

    return {
        "preprocess": preprocess_ppl,
        "predict": predict_ppl,
        "evaluate": evaluate_ppl,
        "adjust": adjust_ppl,
        "all": preprocess_ppl + predict_ppl + evaluate_ppl + adjust_ppl,
        "__default__": preprocess_ppl + predict_ppl,
    }
