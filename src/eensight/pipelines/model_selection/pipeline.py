# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_ensemble, create_ensemble, evaluate_ensemble, optimize_model

baseline_train = Pipeline(
    [
        node(
            func=optimize_model,
            inputs=[
                "model_input_data",
                "model_structure",
                "distance_metrics",
                "params:for_optimize_model",
            ],
            outputs=["composite_model", "opt_scores", "opt_params"],
            name="optimize_model",
        ),
        node(
            func=create_ensemble,
            inputs=["model_input_data", "composite_model", "opt_scores", "opt_params"],
            outputs="ensemble_model",
            name="create_ensemble",
        ),
        node(
            func=evaluate_ensemble,
            inputs=[
                "model_input_data",
                "ensemble_model",
                "params:include_components",
                "params:for_cross_validate",
            ],
            outputs=["model_prediction", "cv_model"],
            name="evaluate_ensemble",
        ),
    ]
)


baseline_train = pipeline(
    baseline_train,
    inputs=["model_structure", "distance_metrics"],  # don't namespace
    outputs=["ensemble_model", "cv_model"],  # don't namespace
    namespace="train",
)


baseline_test = Pipeline(
    [
        node(
            func=apply_ensemble,
            inputs=[
                "model_input_data",
                "ensemble_model",
                "params:include_components",
            ],
            outputs="model_prediction",
            name="apply_ensemble",
        ),
    ]
)

baseline_test = pipeline(
    baseline_test,
    inputs="ensemble_model",  # don't namespace
    namespace="test",
)


def create_pipeline(**kwargs):
    return Pipeline([baseline_train], tags="train") + Pipeline(
        [baseline_test], tags="test"
    )
