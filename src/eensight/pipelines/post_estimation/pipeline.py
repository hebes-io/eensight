# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_ensemble

predict_post = Pipeline(
    [
        node(
            func=apply_ensemble,
            inputs=[
                "model_input_data",
                "ensemble_model",
                "params:include_components",
            ],
            outputs=["model_prediction", "savings"],
            name="apply_ensemble",
        ),
    ]
)

predict_post = pipeline(
    predict_post,
    inputs="ensemble_model",  # don't namespace
    namespace="post",
)


def create_pipeline(**kwargs):
    return Pipeline([predict_post], tags="post")
