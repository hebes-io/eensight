# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_predict

predict = Pipeline(
    [
        node(
            func=apply_predict,
            inputs=[
                "model_input_data",
                "mean_model",
                "conformal_model",
                "params:of_apply_predict",
            ],
            outputs="model_prediction",
            name="apply_predict",
        ),
    ]
)

predict_train = pipeline(
    predict,
    inputs=["mean_model", "conformal_model"],  # don't namespace
    namespace="train",
)

predict_apply = pipeline(
    predict,
    inputs=["mean_model", "conformal_model"],  # don't namespace
    namespace="apply",
)


def create_pipeline(**kwargs):
    return Pipeline([predict_train], tags="train") + Pipeline(
        [predict_apply], tags="apply"
    )
